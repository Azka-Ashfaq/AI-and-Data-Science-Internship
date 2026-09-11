"""
tools.py
--------
External tools / data sources used by the ticket-triage agent.

- kb_lookup()      -> real local data source (JSON "database" of support articles)
- score_priority()  -> deterministic priority scorer (business rules)
- send_response()  -> mock outbound-communication tool; simulates timeouts/errors
                       so the agent has to handle a real failure mode.

Swap send_response()'s internals for a real email/Zendesk/Intercom API call in
production -- the interface (input dict -> result dict) stays the same.
"""

import json
import random
import time
from pathlib import Path

KB_PATH = Path(__file__).parent / "data" / "kb.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    _KB = json.load(f)

HIGH_RISK_TOPICS = {"refund_policy", "billing_dispute", "cancellation", "smart_contract_bug"}


class ToolTimeoutError(Exception):
    """Raised when a tool call exceeds its simulated timeout."""


def kb_lookup(ticket_text: str) -> dict:
    """
    Real data-source tool: scores the local KB by keyword overlap and
    returns the best-matching article, or a 'no_match' result.
    """
    text = ticket_text.lower()
    best, best_score = None, 0
    for entry in _KB:
        score = sum(1 for kw in entry["keywords"] if kw in text)
        if score > best_score:
            best, best_score = entry, score

    if best is None or best_score == 0:
        return {"matched": False, "topic": "unknown", "answer": None, "kb_id": None}

    return {
        "matched": True,
        "topic": best["topic"],
        "answer": best["answer"],
        "kb_id": best["id"],
        "requires_approval": best["topic"] in HIGH_RISK_TOPICS,
    }


def score_priority(ticket_text: str) -> str:
    """Deterministic business-rule priority scorer (no model call needed)."""
    text = ticket_text.lower()
    urgent_words = ["urgent", "asap", "production down", "hacked", "exploit", "losing money", "lawsuit"]
    if any(w in text for w in urgent_words):
        return "P1-critical"
    if any(w in text for w in ["refund", "cancel", "overcharged", "billing"]):
        return "P2-high"
    return "P3-normal"


def send_response(ticket_id: str, message: str, fail_rate: float = 0.0) -> dict:
    """
    Mock outbound-communication tool (stands in for an email/helpdesk API call).
    `fail_rate` lets tests deliberately trigger a timeout to exercise the
    agent's error-handling path.
    """
    start = time.time()
    if random.random() < fail_rate:
        time.sleep(0.05)
        raise ToolTimeoutError(f"send_response timed out contacting helpdesk API for {ticket_id}")

    latency_ms = round((time.time() - start) * 1000, 2)
    return {"status": "sent", "ticket_id": ticket_id, "latency_ms": latency_ms, "chars_sent": len(message)}
