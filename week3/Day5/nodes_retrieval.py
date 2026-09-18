"""
nodes_retrieval.py — Retrieval node that dispatches to Day-3 retrieval tools.

Day 5 hardening: tool calls go through resilience.safe_call so a slow CSV
load or an unexpected pandas exception is bounded to TIMEOUTS["retrieval"]
seconds and surfaced as a uniform tool_error instead of propagating.
"""

import re
import entity_resolution as er
from tools_retrieval import (
    get_team_head_to_head,
    get_player_season_stats,
    get_player_recent_games,
    get_player_career_average,
)
from state import AFLGraphState
from resilience import safe_langchain_tool, TIMEOUTS


def _dispatch(tool_obj, tool_input: dict, tool_called_name: str) -> dict:
    """Run a langchain retrieval tool through the safe_call boundary and
    normalize the result into the shape nodes_validation.py expects."""
    outcome = safe_langchain_tool(tool_obj, tool_input, timeout=TIMEOUTS["retrieval"])
    if outcome["status"] == "error":
        return {
            "tool_called": tool_called_name,
            "tool_results": {"status": "error", "error": outcome["error"]},
        }
    return {"tool_called": tool_called_name, "tool_results": outcome["result"]}


def _extract_teams(query: str) -> list:
    q = query.lower()
    found = []
    for alias in sorted(er.TEAM_ALIASES.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            canonical = er.TEAM_ALIASES[alias]
            if canonical not in found:
                found.append(canonical)
    return found


def _extract_player(query: str) -> str:
    q = query.lower()
    er._init_player_db()
    for alias in sorted(er._player_alias_map.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            return er._player_alias_map[alias]
    return ""


def _extract_year(query: str) -> int:
    m = re.search(r"\b(19|20)\d{2}\b", query)
    return int(m.group(0)) if m else 0


def retrieval_node(state: AFLGraphState) -> dict:
    query = state.get("user_query", "")
    q = query.lower()

    # Head-to-head between two teams
    teams = _extract_teams(query)
    if len(teams) >= 2 and any(
        k in q for k in ["head to head", "head-to-head", "h2h", "record", "history", "vs"]
    ):
        result = _dispatch(
            get_team_head_to_head, {"team_a": teams[0], "team_b": teams[1]},
            "get_team_head_to_head",
        )
        result["resolved_entities"] = {"team_a": teams[0], "team_b": teams[1]}
        return result

    # Player queries
    player = _extract_player(query)
    if player:
        year = _extract_year(query)
        if year and any(k in q for k in ["season", "stats", str(year)]):
            result = _dispatch(
                get_player_season_stats, {"player_name": player, "year": year},
                "get_player_season_stats",
            )
            result["resolved_entities"] = {"player_name": player}
            return result

        if any(k in q for k in ["career", "all-time", "all time"]):
            result = _dispatch(
                get_player_career_average, {"player_name": player},
                "get_player_career_average",
            )
            result["resolved_entities"] = {"player_name": player}
            return result

        # Default: recent game
        result = _dispatch(
            get_player_recent_games, {"player_name": player, "n_games": 1},
            "get_player_recent_games",
        )
        result["resolved_entities"] = {"player_name": player}
        return result

    return {
        "validation_status": "needs_clarification",
        "clarification_prompt": (
            "I couldn't identify the team or player you're asking about. "
            "Could you rephrase with a team or player name?"
        ),
        "tool_called": None,
    }