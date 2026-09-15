"""
predict.py -- callable prediction functions for the AFL match-winner and top-player models.

Loads the trained pipelines + lookup snapshots saved by the Week 3 Day 2 notebook
(model_store/) and exposes two functions ready to be wrapped as agent tools:

    predict_match_winner(home_team, away_team, date, venue=None) -> dict
    predict_top_player(team, date, stat_type="fantasy_score", top_k=5) -> list[dict]

Both raise ValueError with a clear message on bad input (unknown team, unknown player,
date outside the data's known range) rather than failing silently or crashing with a
raw KeyError/IndexError -- important since an LLM agent tool needs to be able to catch
and relay a clean error message to the user.
"""

import joblib
import pandas as pd
from pathlib import Path

STORE = Path(__file__).parent / "model_store"

_match_model = joblib.load(STORE / "match_winner_model.joblib")
_player_model = joblib.load(STORE / "top_player_model.joblib")
_team_snapshot = pd.read_csv(STORE / "team_latest_snapshot.csv", index_col="team_name")
_player_snapshot = pd.read_csv(STORE / "player_latest_snapshot.csv", index_col="player_id")
_known_teams = set(pd.read_csv(STORE / "known_teams.csv")["team_name"])
_known_players = pd.read_csv(STORE / "known_players.csv")["player_name"]

_DATA_MIN_DATE = pd.Timestamp("1983-01-01")
_DATA_MAX_DATE = pd.Timestamp("2025-12-31")


def _validate_team(team_name):
    if team_name not in _known_teams:
        raise ValueError(
            f"Unknown team '{team_name}'. Must be one of the {len(_known_teams)} known team names, "
            f"e.g. {sorted(list(_known_teams))[:3]}..."
        )


def _validate_date(date):
    ts = pd.Timestamp(date)
    if ts < _DATA_MIN_DATE:
        raise ValueError(f"Date {date} is before the data's known range ({_DATA_MIN_DATE.date()}).")
    return ts


def predict_match_winner(home_team, away_team, date, venue=None):
    """
    Predict the winner of an upcoming match.

    Parameters
    ----------
    home_team, away_team : str -- must match a known AFL team name exactly
    date : str | datetime -- the match date, e.g. "2026-04-05"
    venue : str, optional -- venue name; if omitted, venue-specific features are left blank

    Returns
    -------
    dict with keys: winner, probability, home_team, away_team, date
    """
    _validate_team(home_team)
    _validate_team(away_team)
    if home_team == away_team:
        raise ValueError("home_team and away_team must be different teams.")
    match_date = _validate_date(date)

    if home_team not in _team_snapshot.index:
        raise ValueError(f"No historical data available yet for '{home_team}' to build a prediction.")
    if away_team not in _team_snapshot.index:
        raise ValueError(f"No historical data available yet for '{away_team}' to build a prediction.")

    home_row = _team_snapshot.loc[home_team].add_prefix("home_")
    away_row = _team_snapshot.loc[away_team].add_prefix("away_")
    row = pd.concat([home_row, away_row])
    row["venue"] = venue if venue is not None else "Unknown"

    x = pd.DataFrame([row])
    prob_home_win = float(_match_model.predict_proba(x)[0, 1])
    winner = home_team if prob_home_win >= 0.5 else away_team

    return {
        "home_team": home_team, "away_team": away_team, "date": str(match_date.date()),
        "winner": winner, "probability": round(prob_home_win if winner == home_team else 1 - prob_home_win, 3),
    }


def predict_top_player(team, date, stat_type="fantasy_score", top_k=5):
    """
    Rank a team's players by predicted output for their next match.

    Parameters
    ----------
    team : str -- must match a known AFL team name exactly
    date : str | datetime
    stat_type : str -- currently supports "fantasy_score" (the trained model's target)
    top_k : int -- how many ranked players to return

    Returns
    -------
    list of dicts: [{player_id, predicted_score}, ...] sorted descending, length <= top_k
    """
    _validate_team(team)
    _validate_date(date)
    if stat_type != "fantasy_score":
        raise ValueError(f"stat_type '{stat_type}' not supported yet -- only 'fantasy_score' is trained.")

    roster = _player_snapshot[_player_snapshot["team"] == team]
    if roster.empty:
        raise ValueError(f"No recent player data available for '{team}'.")

    feature_cols = [c for c in roster.columns if c != "team"]
    preds = _player_model.predict(roster[feature_cols])
    ranked = (
        roster.assign(predicted_score=preds)
        .sort_values("predicted_score", ascending=False)
        .head(top_k)
    )
    return [
        {"player_id": int(idx), "predicted_score": round(float(row.predicted_score), 1)}
        for idx, row in ranked.iterrows()
    ]
