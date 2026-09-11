"""
api.py
------
FastAPI wrapper around the ticket-triage agent.

Endpoints:
  POST /triage          -> run a new ticket through the agent
  POST /triage/{id}/resume -> resume a ticket paused at the human-approval checkpoint
  GET  /healthz          -> liveness check

Logging: every request writes one structured JSON line to logs/agent.log
capturing input (truncated), tool-relevant outputs, latency, a rough token/
cost proxy, and errors -- the shape a real monitoring pipeline (Datadog,
CloudWatch, a Grafana/Loki stack, etc.) would ingest.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_graph import run_ticket, resume_ticket, AGENT

LOG_PATH = Path(__file__).parent / "logs" / "agent.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

app = FastAPI(title="Web3Geeks Ticket Triage Agent", version="1.0")

# Pending approvals live in the agent's own InMemorySaver checkpointer
# (keyed by ticket_id as the LangGraph thread_id) -- no separate store
# needed here. Production note: swap InMemorySaver for SqliteSaver /
# PostgresSaver in agent_graph.py so approvals survive a restart and
# multiple API workers share state.


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, description="Raw ticket / client message")


class ResumeRequest(BaseModel):
    approved: bool


def _rough_token_estimate(text: str) -> int:
    # proxy metric only -- swap for real usage.input_tokens/output_tokens
    # once classify_ticket/draft_response call a real model.
    return max(1, len(text) // 4)


def _log_event(event: str, **fields):
    record = {"ts": round(time.time(), 3), "event": event, **fields}
    logger.info(json.dumps(record))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/triage")
def triage(req: TicketRequest):
    start = time.time()
    ticket_id = f"T-{uuid.uuid4().hex[:8]}"
    _log_event("request_received", ticket_id=ticket_id, input_preview=req.text[:80])

    try:
        result = run_ticket(req.text, ticket_id=ticket_id)
    except Exception as e:  # last-resort guard so the API never 500s silently
        _log_event("agent_error", ticket_id=ticket_id, error=str(e))
        raise HTTPException(status_code=500, detail="agent_execution_failed")

    latency_ms = round((time.time() - start) * 1000, 2)
    tokens_est = _rough_token_estimate(req.text) + _rough_token_estimate(result.get("draft") or "")

    _log_event(
        "request_completed",
        ticket_id=ticket_id,
        status=result.get("status"),
        category=result.get("category"),
        priority=result.get("priority"),
        needs_approval=result.get("needs_approval", False),
        latency_ms=latency_ms,
        tokens_est=tokens_est,
        error=result.get("error"),
    )

    return {
        "ticket_id": ticket_id,
        "status": result.get("status"),
        "category": result.get("category"),
        "priority": result.get("priority"),
        "needs_approval": result.get("needs_approval", False),
        "draft": result.get("draft"),
        "final_response": result.get("final_response"),
        "latency_ms": latency_ms,
    }


@app.post("/triage/{ticket_id}/resume")
def resume(ticket_id: str, req: ResumeRequest):
    config = {"configurable": {"thread_id": ticket_id}}
    if not AGENT.get_state(config).next:
        raise HTTPException(status_code=404, detail="no_pending_ticket_with_that_id")

    start = time.time()
    result = resume_ticket(ticket_id, req.approved)
    latency_ms = round((time.time() - start) * 1000, 2)

    _log_event("request_resumed", ticket_id=ticket_id, approved=req.approved,
               status=result.get("status"), latency_ms=latency_ms)

    return {
        "ticket_id": ticket_id,
        "status": result.get("status"),
        "final_response": result.get("final_response"),
        "latency_ms": latency_ms,
    }
