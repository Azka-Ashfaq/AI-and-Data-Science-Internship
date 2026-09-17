"""
nodes_prediction.py - Prediction node (Task 3).
"""

import re
import entity_resolution as er
from tools_prediction import predict_match_winner, predict_top_player
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


def _extract_player(query: str):
    q = query.lower()
    try:
        er._init_player_db()
    except Exception:
        pass
    for alias in sorted(er._player_alias_map.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            return er._player_alias_map[alias]
    return None


def _is_top_player_query(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in [
        "top score", "top-score", "topscore", "top scorer",
        "top player", "best player", "best on ground", "bog",
    ])


def prediction_node(state: AFLGraphState) -> dict:
    query = state.get("user_query", "")

    # 1. Top-player prediction for a team
    if _is_top_player_query(query):
        teams = _extract_teams(query)
        if not teams:
            return {
                "validation_status": "needs_clarification",
                "clarification_prompt": "Which team's top player would you like me to predict?",
                "tool_called": None,
            }
        team = teams[0]
        date = er.resolve_match_date(None)
        try:
            result = predict_top_player(team=team, date=date, stat_type="fantasy_score")
        except Exception as e:
            return {
                "validation_status": "tool_error",
                "validation_error": str(e),
                "tool_called": "predict_top_player",
            }
        return {
            "tool_called": "predict_top_player",
            "tool_results": result,
            "resolved_entities": {"team": team, "date": date},
            "feature_drivers": result.get("feature_drivers"),
        }

    # 2. Player-oriented query with unresolved player
    player_query = any(k in query.lower() for k in [
        "play well", "perform", "kick a bag", "score a bag",
    ])
    if player_query and _extract_player(query) is None:
        return {
            "validation_status": "needs_clarification",
            "clarification_prompt": (
                "I need the player's full name and team to predict their "
                "performance. Could you clarify which player you mean?"
            ),
            "tool_called": None,
        }

    # 3. Match-winner prediction
    teams = _extract_teams(query)
    if len(teams) < 2:
        return {
            "validation_status": "needs_clarification",
            "clarification_prompt": (
                "I need two teams to predict a match winner. "
                "Which two teams are playing?"
            ),
            "tool_called": None,
        }
    home, away = teams[0], teams[1]
    date = er.resolve_match_date(None, home, away)
    try:
        result = predict_match_winner(home_team=home, away_team=away, date=date)
    except Exception as e:
        return {
            "validation_status": "tool_error",
            "validation_error": str(e),
            "tool_called": "predict_match_winner",
        }
    return {
        "tool_called": "predict_match_winner",
        "tool_results": result,
        "resolved_entities": {"home_team": home, "away_team": away, "date": date},
        "feature_drivers": result.get("feature_drivers"),
    }