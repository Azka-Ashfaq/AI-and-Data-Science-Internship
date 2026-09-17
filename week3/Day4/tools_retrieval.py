"""
tools_retrieval.py — Structured statistical retrieval tools for AFL Chat & Retrieval (Task 1 & 4).

Wraps the Day 3 structured queries against tabular AFL records:
  - get_team_head_to_head
  - get_player_season_stats
  - get_player_recent_games
  - get_player_career_average

All tools use entity_resolution for nicknames and return structured responses
with clear success/error/clarification signals.
"""

import glob
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from langchain_core.tools import tool

import entity_resolution as er

DATA_DIR = Path(__file__).parent / "afl_datasets"

# Lazy-loaded datasets
_players_info = None
_round_by_round = None
_team_matches = None


def _find_file(keyword: str) -> Path:
    matches = [f for f in DATA_DIR.glob("*.csv") if keyword in f.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No CSV found containing '{keyword}' in {DATA_DIR}")
    return matches[0]


def _load_retrieval_data():
    global _players_info, _round_by_round, _team_matches
    if _team_matches is not None:
        return
        
    players_info = pd.read_csv(_find_file("players_info"))
    round_by_round = pd.read_csv(_find_file("round_by_round"), low_memory=False)
    team_matches = pd.read_csv(_find_file("matches_home_away"))

    def clean_team_name(s):
        s = s.astype(str).str.strip()
        # Map known variations
        mapping = {"w. bulldogs": "Western Bulldogs", "western bulldogs": "Western Bulldogs"}
        return s.str.lower().map(mapping).fillna(s)

    team_matches["team_name"] = clean_team_name(team_matches["team_name"])
    team_matches["opponent"] = clean_team_name(team_matches["opponent"])
    round_by_round["team"] = clean_team_name(round_by_round["team"])
    round_by_round["opponent"] = clean_team_name(round_by_round["opponent"])

    round_by_round["player_id"] = round_by_round["player_id"].astype("Int64")
    players_info["id"] = players_info["id"].astype("Int64")
    team_matches["match_date"] = pd.to_datetime(team_matches["match_date"])
    round_by_round["match_date"] = pd.to_datetime(round_by_round["match_date"])

    players_info["player_name"] = players_info["player_name"].astype(str).str.strip()

    round_by_round = round_by_round.merge(
        players_info[["id", "player_name"]], left_on="player_id", right_on="id", how="left"
    )
    
    _players_info = players_info
    _round_by_round = round_by_round
    _team_matches = team_matches


@tool
def get_team_head_to_head(team_a: str, team_b: str) -> Dict[str, Any]:
    """Get the all-time head-to-head match record between two AFL teams (wins, losses, draws,
    total games played, and the most recent meeting). Use for questions about historical records
    between two teams. Accepts club names or nicknames (e.g., 'Tigers' vs 'Blues').
    """
    _load_retrieval_data()
    ta, err_a = er.resolve_team(team_a)
    tb, err_b = er.resolve_team(team_b)
    
    if not ta:
        return {"status": "error", "error": err_a or f"Could not resolve '{team_a}'."}
    if not tb:
        return {"status": "error", "error": err_b or f"Could not resolve '{team_b}'."}

    games = _team_matches[(_team_matches["team_name"] == ta) & (_team_matches["opponent"] == tb)]
    if games.empty:
        return {"status": "not_found", "message": f"No recorded AFL matches found between {ta} and {tb}."}

    wins = int((games["result"] == "W").sum())
    losses = int((games["result"] == "L").sum())
    draws = int((games["result"] == "D").sum())
    last = games.sort_values("match_date").iloc[-1]

    return {
        "status": "success",
        "team_a": ta,
        "team_b": tb,
        "total_games": len(games),
        "wins_team_a": wins,
        "wins_team_b": losses,
        "draws": draws,
        "last_meeting_date": str(last["match_date"].date()),
        "last_venue": last["venue"],
        "last_score_team_a": int(last["team_score"]),
        "last_score_team_b": int(last["opponent_score"]),
        "last_result": f"{ta} win" if last["result"] == "W" else (f"{tb} win" if last["result"] == "L" else "Draw"),
        "summary": f"Across {len(games)} matches, {ta} has {wins} wins, {tb} has {losses} wins, and {draws} draws. Most recent match on {last['match_date'].date()} at {last['venue']}: {ta} {int(last['team_score'])} vs {tb} {int(last['opponent_score'])}."
    }


@tool
def get_player_season_stats(player_name: str, year: int) -> Dict[str, Any]:
    """Get aggregated stats for an AFL player for a specific season (games, disposals total & avg,
    goals total & avg, fantasy points). Use for queries like 'Dustin Martin 2017 stats'.
    """
    _load_retrieval_data()
    resolved, err, candidates = er.resolve_player(player_name)
    if not resolved:
        return {
            "status": "ambiguous" if candidates else "error",
            "error": err,
            "candidates": candidates
        }

    rows = _round_by_round[(_round_by_round["player_name"] == resolved) & (_round_by_round["year"] == int(year))]
    if rows.empty:
        return {"status": "not_found", "message": f"No recorded matches found for {resolved} in season {year}."}

    games = len(rows)
    tot_disp = int(rows["disposals"].sum())
    avg_disp = round(float(rows["disposals"].mean()), 1)
    tot_goals = int(rows["goals"].sum())
    avg_goals = round(float(rows["goals"].mean()), 1)
    tot_fan = int(rows["fantasy_points"].sum())
    avg_fan = round(float(rows["fantasy_points"].mean()), 1)

    return {
        "status": "success",
        "player_name": resolved,
        "year": int(year),
        "games_played": games,
        "total_disposals": tot_disp,
        "avg_disposals": avg_disp,
        "total_goals": tot_goals,
        "avg_goals": avg_goals,
        "total_fantasy_points": tot_fan,
        "avg_fantasy_points": avg_fan,
        "summary": f"{resolved} in {year} ({games} games): {tot_disp} disposals (avg {avg_disp}), {tot_goals} goals (avg {avg_goals}), {tot_fan} fantasy points (avg {avg_fan})."
    }


@tool
def get_player_recent_games(player_name: str, n_games: int = 1) -> Dict[str, Any]:
    """Get a player's stats for their most recent N games (default 1). Use for queries like
    'how many disposals did X get last round?'.
    """
    _load_retrieval_data()
    resolved, err, candidates = er.resolve_player(player_name)
    if not resolved:
        return {
            "status": "ambiguous" if candidates else "error",
            "error": err,
            "candidates": candidates
        }

    rows = _round_by_round[_round_by_round["player_name"] == resolved].sort_values("match_date", ascending=False)
    if rows.empty:
        return {"status": "not_found", "message": f"No match records found for {resolved}."}

    recent_rows = rows.head(n_games)
    games_list = []
    for _, r in recent_rows.iterrows():
        games_list.append({
            "match_date": str(r["match_date"].date()),
            "round": r["round"],
            "year": int(r["year"]),
            "opponent": r["opponent"],
            "disposals": int(r["disposals"]) if pd.notna(r["disposals"]) else 0,
            "goals": int(r["goals"]) if pd.notna(r["goals"]) else 0,
            "fantasy_points": int(r["fantasy_points"]) if pd.notna(r["fantasy_points"]) else 0
        })

    last_game = games_list[0]
    return {
        "status": "success",
        "player_name": resolved,
        "games": games_list,
        "most_recent_game": last_game,
        "summary": f"{resolved}'s last game was on {last_game['match_date']} (Round {last_game['round']}, {last_game['year']} vs {last_game['opponent']}): {last_game['disposals']} disposals, {last_game['goals']} goals, {last_game['fantasy_points']} fantasy points."
    }


@tool
def get_player_career_average(player_name: str) -> Dict[str, Any]:
    """Get all-time career averages (disposals, goals, fantasy points, total games) for an AFL player.
    Use for career comparisons like 'how does that compare to his career average?'.
    """
    _load_retrieval_data()
    resolved, err, candidates = er.resolve_player(player_name)
    if not resolved:
        return {
            "status": "ambiguous" if candidates else "error",
            "error": err,
            "candidates": candidates
        }

    rows = _round_by_round[_round_by_round["player_name"] == resolved]
    if rows.empty:
        return {"status": "not_found", "message": f"No career matches found for {resolved}."}

    games = len(rows)
    avg_disp = round(float(rows["disposals"].mean()), 1)
    avg_goals = round(float(rows["goals"].mean()), 1)
    avg_fan = round(float(rows["fantasy_points"].mean()), 1)

    return {
        "status": "success",
        "player_name": resolved,
        "career_games": games,
        "career_avg_disposals": avg_disp,
        "career_avg_goals": avg_goals,
        "career_avg_fantasy_points": avg_fan,
        "summary": f"Over {games} career matches, {resolved} averaged {avg_disp} disposals, {avg_goals} goals, and {avg_fan} fantasy points per game."
    }


RETRIEVAL_TOOLS = [get_team_head_to_head, get_player_season_stats, get_player_recent_games, get_player_career_average]
