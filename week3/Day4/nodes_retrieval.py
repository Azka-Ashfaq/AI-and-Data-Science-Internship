"""
nodes_retrieval.py — Retrieval node that dispatches to Day-3 retrieval tools.
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
        result = get_team_head_to_head.invoke({"team_a": teams[0], "team_b": teams[1]})
        return {"tool_called": "get_team_head_to_head", "tool_results": result}

    # Player queries
    player = _extract_player(query)
    if player:
        year = _extract_year(query)
        if year and any(k in q for k in ["season", "stats", str(year)]):
            result = get_player_season_stats.invoke(
                {"player_name": player, "year": year}
            )
            return {"tool_called": "get_player_season_stats", "tool_results": result}

        if any(k in q for k in ["career", "all-time", "all time"]):
            result = get_player_career_average.invoke({"player_name": player})
            return {"tool_called": "get_player_career_average", "tool_results": result}

        # Default: recent game
        result = get_player_recent_games.invoke({"player_name": player, "n_games": 1})
        return {"tool_called": "get_player_recent_games", "tool_results": result}

    return {
        "validation_status": "needs_clarification",
        "clarification_prompt": (
            "I couldn't identify the team or player you're asking about. "
            "Could you rephrase with a team or player name?"
        ),
        "tool_called": None,
    }