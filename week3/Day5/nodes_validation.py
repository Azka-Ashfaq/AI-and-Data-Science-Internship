"""
nodes_validation.py — Validation, clarification, and fallback nodes (Task 4).
"""

from state import AFLGraphState


def validate_node(state: AFLGraphState) -> dict:
    """Inspect tool_results; set validation_status accordingly."""
    if state.get("validation_status") in (
        "needs_clarification",
        "unsupported_fallback",
        "tool_error",
    ):
        return {}

    tr = state.get("tool_results")
    if tr is None:
        return {
            "validation_status": "tool_error",
            "validation_error": "No tool result was produced.",
        }

    if isinstance(tr, dict):
        status = tr.get("status")
        if status == "error":
            return {
                "validation_status": "tool_error",
                "validation_error": tr.get("error", "Unknown tool error."),
            }
        if status == "ambiguous":
            return {
                "validation_status": "needs_clarification",
                "clarification_prompt": tr.get("error", "Which one did you mean?"),
            }
        if status == "not_found":
            return {
                "validation_status": "unsupported_fallback",
                "validation_error": tr.get("message", "No data found."),
            }
        if status == "unsupported":
            return {
                "validation_status": "unsupported_fallback",
                "validation_error": tr.get("error", "That prediction type is not supported."),
            }
        if status == "success":
            return {"validation_status": "valid"}

    return {"validation_status": "valid"}


def route_from_validation(state: AFLGraphState) -> str:
    status = state.get("validation_status", "valid")
    return {
        "valid": "formatter_node",
        "needs_clarification": "clarify_node",
        "unsupported_fallback": "fallback_node",
        "tool_error": "fallback_node",
    }.get(status, "fallback_node")


def clarify_node(state: AFLGraphState) -> dict:
    prompt = state.get("clarification_prompt") or (
        "Could you clarify which team or player you mean?"
    )
    return {"final_response": prompt}


def fallback_node(state: AFLGraphState) -> dict:
    err = state.get("validation_error", "")
    msg = (
        "I can't help with that specific request. "
        + (f"Reason: {err}" if err else "Please try rephrasing.")
    )
    return {"final_response": msg}


SCOPE_MESSAGE = (
    "I can only answer AFL-related questions — stats, head-to-heads, "
    "and match/player predictions. Please ask me something about the AFL."
)


def refusal_node(state: AFLGraphState) -> dict:
    """Day 5 hardening: escalate wording after repeated off-scope probing
    (tracked by router_node as off_topic_streak) instead of silently
    repeating the same refusal forever. This does not block the user —
    rate limiting for actual abuse is handled at the API layer (see
    api.py) — it just makes the scope boundary explicit."""
    streak = state.get("off_topic_streak", 0) or 0
    if streak >= 3:
        return {
            "final_response": (
                SCOPE_MESSAGE
                + " I've had to say this a few times now — this assistant "
                "is scoped to AFL only and that isn't going to change "
                "within this conversation."
            )
        }
    return {"final_response": SCOPE_MESSAGE}


def direct_answer_node(state: AFLGraphState) -> dict:
    """Simple factual handler."""
    q = state.get("user_query", "").lower()
    if "how many teams" in q:
        answer = "There are currently 18 teams in the AFL competition."
    elif "how many players" in q:
        answer = "Each AFL team fields 18 players on the ground (plus 4 interchange)."
    elif "goal" in q and "behind" in q:
        answer = "A goal is worth 6 points; a behind is worth 1 point."
    else:
        answer = (
            "I don't have a canned answer for that fact, but I can help with "
            "stats, head-to-heads, and predictions."
        )
    return {"final_response": answer}