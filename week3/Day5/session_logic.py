"""
session_logic.py — shared multi-turn entity-carryover logic.

Extracted so api.py, streamlit_app.py, and eval_suite.py all run the exact
same carryover behavior instead of three hand-copied versions drifting
apart. This is the concrete fix for the "weakest category" finding in
Task 2: nodes_*.py accept conversation_history but no graph node reads it,
so a follow-up like "what about his career average" or "why do you think
that" needs a team/player it never got. Here, the *caller* (API/UI/eval
harness) remembers the last resolved entity and retries once with it
appended as an explicit hint.

Handles entity shapes from both node families:
  * nodes_prediction.py -> {"home_team": ..., "away_team": ...} or {"team": ...}
  * nodes_retrieval.py  -> {"team_a": ..., "team_b": ...} or {"player_name": ...}
"""

from typing import Any, Dict


def needs_entity_carryover(final_state: dict) -> bool:
    prompt = (final_state.get("clarification_prompt") or "").lower()
    return final_state.get("validation_status") == "needs_clarification" and (
        "team" in prompt or "player" in prompt
    )


def build_carryover_hint(entities: Dict[str, Any]) -> str:
    if not entities:
        return ""
    if entities.get("home_team") and entities.get("away_team"):
        return f" (context: {entities['home_team']} vs {entities['away_team']})"
    if entities.get("team_a") and entities.get("team_b"):
        return f" (context: {entities['team_a']} vs {entities['team_b']})"
    if entities.get("team"):
        return f" (context: {entities['team']})"
    if entities.get("player_name"):
        return f" (context: {entities['player_name']})"
    return ""


def run_with_carryover(graph_app, query: str, history: list, off_topic_streak: int,
                        last_entities: Dict[str, Any]) -> dict:
    """Invoke the graph once; if the result needs a team/player we already
    resolved last turn, retry exactly once with that entity hinted in the
    query text. Returns the final state to use (retried one if it helped,
    original otherwise), with `_entity_carryover_used` set on it."""
    state = {
        "user_query": query,
        "conversation_history": history,
        "messages": [],
        "off_topic_streak": off_topic_streak,
    }
    final = graph_app.invoke(state)

    if needs_entity_carryover(final) and last_entities:
        hint = build_carryover_hint(last_entities)
        if hint:
            retried = graph_app.invoke({
                "user_query": query + hint,
                "conversation_history": history,
                "messages": [],
                "off_topic_streak": off_topic_streak,
            })
            if retried.get("validation_status") != "needs_clarification":
                retried["_entity_carryover_used"] = True
                return retried

    final["_entity_carryover_used"] = False
    return final