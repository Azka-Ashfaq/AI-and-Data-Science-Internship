"""
api.py — FastAPI wrapper for the AFL LangGraph assistant (Day 5, Task 3).

Run:
    pip install fastapi uvicorn
    uvicorn api:api_app --reload --port 8000

Endpoint:
    POST /chat
    body: {"message": "...", "conversation_id": "optional-string"}
    returns: {
        "response": "...",
        "conversation_id": "...",
        "intent": "prediction" | "retrieval" | "factual" | "off_topic",
        "tool_called": "predict_match_winner" | ... | null,
        "prediction_metadata": {...} | null,   # win_probability, feature_drivers, etc. when relevant
        "latency_ms": 123.4
    }

What this adds beyond a bare graph.invoke() call:
  * Session memory keyed by conversation_id — conversation_history and a
    lightweight "last resolved entities" cache are kept server-side, so a
    /chat call only ever needs the new message + the id.
  * Entity carryover for follow-ups (Task 2's proposed fix for the weakest
    "conversational coherence" category): if a turn comes back
    needs_clarification specifically for a missing team/player AND the
    session has a resolved team/player from the previous turn, we retry
    once with that context appended to the query text
    (e.g. "why do you think that" -> "why do you think that (context:
    Collingwood Magpies vs Geelong Cats)"). This is intentionally a
    session-layer patch rather than a graph-internal change, so it can't
    destabilize the routing logic that Task 2's eval suite already
    verified at 100% on the existing 20-case regression set.
  * Structured logging (query, detected intent, tool called, latency,
    turn count) — the same event stream Task 4's monitoring checklist
    is built on.
  * A basic in-memory rate limiter and off-topic-streak-aware abuse
    handling per Task 1.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from graph_app import app as graph_app
from session_logic import run_with_carryover

# --------------------------------------------------------------------
# Structured logging — one JSON-ish line per turn. In production, point
# this handler at your log aggregator (CloudWatch, Datadog, etc.) instead
# of stdout; the fields are already the ones Task 4's checklist tracks.
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("afl_api")

api_app = FastAPI(title="AFL Assistant API", version="1.0.0")


@api_app.on_event("startup")
def _preload_data_and_models():
    """Load CSVs and joblib models once at server startup instead of on
    the first real request. Without this, the first retrieval/prediction
    call after a (re)start pays the full CSV-parse / model-deserialize
    cost inline and can trip the per-call timeout in resilience.py even
    though nothing is actually wrong — every call after the first is fast
    because tools_retrieval.py / tools_prediction.py cache the loaded
    data at module level."""
    t0 = time.monotonic()
    try:
        import tools_retrieval
        tools_retrieval._load_retrieval_data()
        logger.info('"event":"startup_preload","component":"retrieval","status":"ok"')
    except Exception as e:
        logger.warning(f'"event":"startup_preload","component":"retrieval","status":"failed","error":"{e}"')
    try:
        import tools_prediction
        tools_prediction._load_prediction_assets()
        logger.info('"event":"startup_preload","component":"prediction","status":"ok"')
    except Exception as e:
        logger.warning(f'"event":"startup_preload","component":"prediction","status":"failed","error":"{e}"')
    logger.info(f'"event":"startup_preload_done","latency_ms":{round((time.monotonic()-t0)*1000,1)}')


# --------------------------------------------------------------------
# Session store (in-memory; swap for Redis if you run more than one
# worker process, since dict state won't be shared across workers)
# --------------------------------------------------------------------
class Session:
    __slots__ = ("history", "off_topic_streak", "last_entities", "turn_count")

    def __init__(self):
        self.history: list[dict] = []
        self.off_topic_streak: int = 0
        self.last_entities: Dict[str, Any] = {}
        self.turn_count: int = 0


SESSIONS: Dict[str, Session] = defaultdict(Session)

# --------------------------------------------------------------------
# Basic abuse/rate limiting: sliding window per conversation_id AND per
# client IP. This is intentionally simple (in-memory deque of timestamps)
# — swap for a Redis-backed limiter before running multiple workers.
# --------------------------------------------------------------------
RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_MAX_REQUESTS = 20
_request_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=RATE_LIMIT_MAX_REQUESTS + 1))


def _check_rate_limit(key: str) -> bool:
    now = time.monotonic()
    dq = _request_log[key]
    dq.append(now)
    # drop timestamps outside the window
    while dq and now - dq[0] > RATE_LIMIT_WINDOW_S:
        dq.popleft()
    return len(dq) <= RATE_LIMIT_MAX_REQUESTS


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    intent: Optional[str] = None
    tool_called: Optional[str] = None
    prediction_metadata: Optional[Dict[str, Any]] = None
    latency_ms: float


@api_app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    conversation_id = req.conversation_id or uuid.uuid4().hex
    session = SESSIONS[conversation_id]
    session.turn_count += 1

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(conversation_id) or not _check_rate_limit(f"ip:{client_ip}"):
        logger.warning(f'"event":"rate_limited","conversation_id":"{conversation_id}"')
        raise HTTPException(status_code=429, detail="Too many requests — please slow down.")

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")

    start = time.monotonic()

    # Entity-carryover retry for a specific, narrow case: a follow-up that
    # needs a team/player we already resolved last turn. One retry only —
    # this never loops. Shared with streamlit_app.py and eval_suite.py via
    # session_logic.py so all three exercise identical behavior.
    final = run_with_carryover(
        graph_app, req.message, session.history, session.off_topic_streak,
        session.last_entities,
    )

    latency_ms = round((time.monotonic() - start) * 1000, 1)

    # persist session state for the next turn
    session.off_topic_streak = final.get("off_topic_streak", 0) or 0
    if final.get("resolved_entities"):
        session.last_entities = final["resolved_entities"]
    session.history.append({"role": "user", "content": req.message})
    session.history.append({"role": "assistant", "content": final.get("final_response", "")})
    # keep the last ~10 turns only, to bound memory/token growth
    session.history = session.history[-20:]

    prediction_metadata = None
    tool_results = final.get("tool_results")
    if final.get("tool_called") in ("predict_match_winner", "predict_top_player") and isinstance(tool_results, dict):
        prediction_metadata = {
            k: v for k, v in tool_results.items()
            if k in (
                "win_probability", "home_probability", "away_probability",
                "confidence_level", "predicted_winner", "ranked_players",
                "feature_drivers", "disclaimer",
            )
        }

    logger.info(
        '"event":"chat_turn","conversation_id":"%s","turn":%d,"intent":"%s",'
        '"tool_called":"%s","validation_status":"%s","latency_ms":%.1f,'
        '"off_topic_streak":%d,"carryover_used":%s',
        conversation_id, session.turn_count, final.get("detected_intent"),
        final.get("tool_called"), final.get("validation_status"), latency_ms,
        session.off_topic_streak, bool(final.get("_entity_carryover_used")),
    )

    return ChatResponse(
        response=final.get("final_response", ""),
        conversation_id=conversation_id,
        intent=final.get("detected_intent"),
        tool_called=final.get("tool_called"),
        prediction_metadata=prediction_metadata,
        latency_ms=latency_ms,
    )


@api_app.get("/health")
def health():
    return {"status": "ok"}


@api_app.delete("/chat/{conversation_id}")
def clear_session(conversation_id: str):
    SESSIONS.pop(conversation_id, None)
    return {"cleared": conversation_id}