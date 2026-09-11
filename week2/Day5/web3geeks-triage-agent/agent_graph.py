"""
agent_graph.py
--------------
Support-Ticket Triage Agent for Web3Geeks (freelance web3 dev studio).

Framework: LangGraph
Why: this workflow is CONTROL-HEAVY, not role-heavy -- there is one clear
sequential pipeline (validate -> classify -> retrieve -> draft -> critique ->
approve -> send) with explicit branches (bad input, needs-approval, retry-
on-quality-fail, tool-timeout). LangGraph's explicit state machine gives us:
  - a typed, inspectable State object we can log/checkpoint at every node
  - conditional edges for the approval gate and the self-correction loop
  - a natural place to pause execution for a human-in-the-loop decision
CrewAI would be the better fit if this were several autonomous *roles*
negotiating a shared goal (e.g. a researcher + writer + editor crew) --
here we need deterministic control flow and auditability instead, which is
LangGraph's strength.

Reused from earlier days (this week's course), not reinvented:
  - Day 2 (LangChain): the local-JSON-"database" tool pattern
    (get_product_price -> here, kb_lookup over data/kb.json).
  - Day 3 (LangGraph): the generate -> critique -> route_critique -> revise
    self-correction cycle, and the interrupt_before + InMemorySaver
    checkpointer mechanism for a human-approval gate (graph.invoke() to run
    until the interrupt, graph.update_state()/graph.invoke(None, config) to
    reject/resume) -- used here almost exactly as built in Day 3, just
    applied to a ticket instead of an email send.
  - Day 4 (CrewAI): the role-scoped tool-access principle (each specialist
    only gets the tools its job needs, so no agent can blend data sources it
    shouldn't) -- carried over as a design rule for each LangGraph node here,
    even though the framework is different.

Note: to keep this capstone runnable without any API keys, `classify_ticket`
and the generate/critique reasoning use structured, deterministic functions
that stand in for an LLM call (same approach Day 3 used for get_text() /
scoring, just without a live Gemini call). Each is written so the reasoning
body can be swapped 1:1 for a real model call without changing the graph.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from tools import kb_lookup, score_priority, send_response, ToolTimeoutError


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class TicketState(TypedDict, total=False):
    ticket_id: str
    raw_text: str
    error: Optional[str]
    category: str
    priority: str
    kb_result: dict
    draft: str
    critique_score: int              # 0-100, mirrors Day 3's critique score
    critique_feedback: str
    retries: int
    max_retries: int
    needs_approval: bool
    status: str                      # running / sent / rejected / failed_will_retry / rejected_invalid_input / refused
    final_response: Optional[str]
    log: list


def _log(state: TicketState, event: str) -> None:
    state.setdefault("log", []).append({"t": round(time.time(), 3), "event": event})


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def validate_input(state: TicketState) -> TicketState:
    text = (state.get("raw_text") or "").strip()
    if not text:
        state["error"] = "empty_input"
    elif len(text) > 4000:
        state["error"] = "input_too_long"
    elif len(text) < 3:
        state["error"] = "input_too_short"
    else:
        state["error"] = None
    _log(state, f"validate_input -> error={state['error']}")
    return state


def classify_ticket(state: TicketState) -> TicketState:
    """Mock LLM classification call (deterministic stand-in)."""
    text = state["raw_text"].lower()

    # Simulated model-refusal path: agent won't classify/act on abusive or
    # out-of-scope requests (e.g. asking the agent to do something unsafe).
    refusal_triggers = ["ignore previous instructions", "give me admin access", "leak the", "bypass"]
    if any(t in text for t in refusal_triggers):
        state["category"] = "refused"
        state["error"] = "model_refusal"
        _log(state, "classify_ticket -> refused (unsafe/out-of-scope request)")
        return state

    if any(w in text for w in ["refund", "cancel", "money back", "overcharged", "billing"]):
        category = "billing"
    elif any(w in text for w in ["contract", "solidity", "bug", "exploit", "revert", "gas"]):
        category = "technical"
    elif any(w in text for w in ["onboarding", "kickoff", "new client", "get started"]):
        category = "onboarding"
    elif any(w in text for w in ["timeline", "eta", "deadline", "when will"]):
        category = "timeline"
    else:
        category = "general"

    state["category"] = category
    state["priority"] = score_priority(text)
    _log(state, f"classify_ticket -> category={category}, priority={state['priority']}")
    return state


def retrieve_kb(state: TicketState, _retries: int = 2) -> TicketState:
    """Calls the KB tool with a small retry loop (defensive tool-call handling)."""
    last_err = None
    for attempt in range(1, _retries + 1):
        try:
            result = kb_lookup(state["raw_text"])
            state["kb_result"] = result
            state["needs_approval"] = bool(result.get("requires_approval")) or state["priority"] == "P1-critical"
            _log(state, f"retrieve_kb attempt={attempt} -> matched={result['matched']}")
            return state
        except Exception as e:  # defensive: treat any tool failure as retryable
            last_err = e
            _log(state, f"retrieve_kb attempt={attempt} failed: {e}")
            time.sleep(0.01)

    # graceful degradation: fall back to "no match" rather than crashing the run
    state["kb_result"] = {"matched": False, "topic": "unknown", "answer": None, "kb_id": None,
                           "requires_approval": True}
    state["needs_approval"] = True
    state["error"] = f"kb_tool_failed: {last_err}"
    _log(state, "retrieve_kb -> exhausted retries, degraded to no-match + forced human review")
    return state


def generate_draft(state: TicketState) -> TicketState:
    """
    Mock LLM drafting call -- the 'generate' half of Day 3's
    generate -> critique -> revise cycle.

    On the first pass, a no-KB-match ticket gets a generic escalation line
    (same limitation the Day-3-style critique below is designed to catch).
    On a revise pass (retries > 0), the draft echoes the client's own
    question back, which is what pushes the critique score above threshold --
    the self-correction loop doesn't just retry blindly, it fixes the
    specific weakness the critique flagged.
    """
    kb = state.get("kb_result", {})
    retries = state.get("retries", 0)

    if kb.get("matched"):
        body = kb["answer"]
        ref = f" (ref: {kb['kb_id']})"
    elif retries > 0:
        body = (f'Thanks for the details -- specifically on "{state["raw_text"].strip()[:120]}", '
                 "I don't have a pre-written answer, so I'm looping in a specialist from our team "
                 "who will follow up with you directly on this.")
        ref = ""
    else:
        body = ("Thanks for reaching out -- I want to make sure you get an accurate answer, "
                "so I'm looping in a specialist from our team who will follow up shortly.")
        ref = ""

    draft = f"Hi there,\n\n{body}{ref}\n\nBest,\nWeb3Geeks Support"
    state["draft"] = draft
    _log(state, f"generate_draft retries={retries} -> matched={kb.get('matched')}")
    return state


def critique_draft(state: TicketState) -> TicketState:
    """
    Mock LLM critique call (Day 3's critique_node, adapted): scores the
    current draft 0-100. A KB-grounded answer always scores high. A generic
    no-match fallback scores below the 80 threshold on the FIRST attempt
    (forcing a revise), but the revised, question-echoing fallback clears it.
    """
    kb = state.get("kb_result", {})
    draft = state.get("draft", "")
    well_formed = draft.startswith("Hi") and ("Best," in draft) and len(draft) < 1000

    if not well_formed:
        score, feedback = 40, "draft is malformed (missing greeting/sign-off)"
    elif kb.get("matched"):
        score, feedback = 95, "grounded in a KB article"
    elif state.get("retries", 0) > 0:
        score, feedback = 85, "no KB match, but escalation now references the client's specific question"
    else:
        score, feedback = 60, "no KB match and the fallback is generic / not specific to the question"

    state["critique_score"] = score
    state["critique_feedback"] = feedback
    _log(state, f"critique_draft -> score={score} ({feedback})")
    return state


def revise(state: TicketState) -> TicketState:
    """Increments the retry counter before looping back to generate_draft (Day 3 pattern)."""
    state["retries"] = state.get("retries", 0) + 1
    _log(state, f"revise -> retry #{state['retries']}, feedback was: {state.get('critique_feedback')}")
    return state


def human_checkpoint(state: TicketState) -> TicketState:
    """
    Human-in-the-loop gate for consequential actions (refunds, cancellations,
    billing disputes, contract-bug escalations, P1 tickets).

    This node itself is a pass-through; the pause/resume mechanics are the
    graph's `interrupt_before=["human_checkpoint"]` + InMemorySaver
    checkpointer (see build_graph / run_ticket / resume_ticket below),
    reused directly from Day 3's send_email approval gate.
    """
    if state.get("status") != "rejected":
        state["status"] = "approved"
    _log(state, f"human_checkpoint -> status={state['status']}")
    return state


def finalize(state: TicketState) -> TicketState:
    """Dispatch the approved response via the (mock) send tool, with error handling."""
    if state.get("status") == "rejected":
        _log(state, "finalize -> skipped, ticket was rejected at human checkpoint")
        return state

    try:
        result = send_response(state["ticket_id"], state["draft"])
        state["final_response"] = state["draft"]
        state["status"] = "sent"
        _log(state, f"finalize -> sent, latency_ms={result['latency_ms']}")
    except ToolTimeoutError as e:
        # graceful handling of failure scenario #2: outbound tool timeout
        state["status"] = "failed_will_retry"
        state["error"] = str(e)
        _log(state, f"finalize -> send failed, queued for retry: {e}")
    return state


def reject_early(state: TicketState) -> TicketState:
    state["status"] = "rejected_invalid_input" if state.get("error") != "model_refusal" else "refused"
    state["final_response"] = None
    _log(state, f"reject_early -> status={state['status']} ({state.get('error')})")
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_validate(state: TicketState) -> str:
    return "classify_ticket" if state.get("error") is None else "reject_early"


def route_after_classify(state: TicketState) -> str:
    return "reject_early" if state.get("error") == "model_refusal" else "retrieve_kb"


def route_after_critique(state: TicketState) -> str:
    """Day 3's route_critique: loop back to revise if score is too low and budget remains."""
    if state["critique_score"] >= 80 or state.get("retries", 0) >= state.get("max_retries", 2):
        return "human_checkpoint" if state.get("needs_approval") else "finalize"
    return "revise"


def route_after_checkpoint(state: TicketState) -> str:
    return END if state.get("status") == "rejected" else "finalize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(TicketState)
    g.add_node("validate_input", validate_input)
    g.add_node("classify_ticket", classify_ticket)
    g.add_node("retrieve_kb", retrieve_kb)
    g.add_node("generate_draft", generate_draft)
    g.add_node("critique_draft", critique_draft)
    g.add_node("revise", revise)
    g.add_node("human_checkpoint", human_checkpoint)
    g.add_node("finalize", finalize)
    g.add_node("reject_early", reject_early)

    g.set_entry_point("validate_input")
    g.add_conditional_edges("validate_input", route_after_validate,
                             {"classify_ticket": "classify_ticket", "reject_early": "reject_early"})
    g.add_conditional_edges("classify_ticket", route_after_classify,
                             {"retrieve_kb": "retrieve_kb", "reject_early": "reject_early"})
    g.add_edge("retrieve_kb", "generate_draft")
    g.add_edge("generate_draft", "critique_draft")
    g.add_conditional_edges("critique_draft", route_after_critique,
                             {"revise": "revise", "human_checkpoint": "human_checkpoint", "finalize": "finalize"})
    g.add_edge("revise", "generate_draft")
    g.add_conditional_edges("human_checkpoint", route_after_checkpoint,
                             {END: END, "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("reject_early", END)

    # Reused directly from Day 3: interrupt_before pauses the graph right
    # before the risky node runs, and InMemorySaver persists state across
    # the pause so a later call can resume (approve) or update_state
    # (reject) using the same thread_id -- swap SqliteSaver/PostgresSaver
    # in production for a restart-safe queue (see README limitations).
    return g.compile(checkpointer=InMemorySaver(), interrupt_before=["human_checkpoint"])


AGENT = build_graph()


def run_ticket(raw_text: str, ticket_id: Optional[str] = None) -> dict:
    """
    Convenience entrypoint used by both the eval harness and the API.
    Runs until completion, or until it pauses at the human-approval gate
    (status will be 'approved'-pending / not yet 'sent' and
    graph.get_state(config).next will be ('human_checkpoint',)).
    """
    ticket_id = ticket_id or f"T-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": ticket_id}}
    state: TicketState = {
        "ticket_id": ticket_id,
        "raw_text": raw_text,
        "status": "running",
        "retries": 0,
        "max_retries": 2,
        "log": [],
    }
    result = AGENT.invoke(state, config)
    pending = AGENT.get_state(config)
    if pending.next == ("human_checkpoint",):
        result = dict(result)
        result["status"] = "awaiting_human_approval"
    return result


def resume_ticket(ticket_id: str, approved: bool) -> dict:
    """
    Resume a ticket paused at the human_checkpoint interrupt -- same
    approve/reject mechanics as Day 3's send_email gate:
      - approve: graph.invoke(None, config) resumes normally
      - reject: graph.update_state(..., as_node="human_checkpoint") injects
        the rejection, then invoke(None, config) runs it through to END
    """
    config = {"configurable": {"thread_id": ticket_id}}
    if not approved:
        AGENT.update_state(config, {"status": "rejected", "final_response": None}, as_node="human_checkpoint")
    return AGENT.invoke(None, config)
