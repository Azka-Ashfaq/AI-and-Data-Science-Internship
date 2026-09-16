"""
tools.py — structured retrieval tools for the AFL chat agent.

Design decision (Task 2): ALL retrieval here is structured lookup (pandas queries against the
cleaned AFL tables), not semantic/vector search. Reasoning:

  1. The available dataset (afl_datasets.zip) is entirely structured tabular data — player bios,
     round-by-round stats, seasonal stats, team match results. There is no free-text corpus
     (no match reports, no articles, no commentary) to build a vector store over.
  2. Even if such text existed, sports STATS specifically (disposals, goals, head-to-head records)
     must never come from fuzzy semantic retrieval — a vector search can return a plausible-sounding
     but wrong number. Exact stats belong in a deterministic query that either returns the real row
     or a clear "not found," never an approximation.

If match reports/commentary become available later, a semantic layer should be added ONLY for
qualitative content (e.g. "what did commentators say about the game") — never for numeric stats,
which should keep going through these structured tools.

Every tool below returns a real value pulled directly from the dataset, or a clear "not found"
message — never a guess.
"""

import glob
import os
import pandas as pd
import numpy as np
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Load & clean data (same rules as Day 1 / Day 2 — see those notebooks for the full writeup)
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("AFL_DATA_DIR", "afl_datasets")


def _find_file(keyword):
    matches = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv")) if keyword in f.lower()]
    if not matches:
        raise FileNotFoundError(f"No CSV found containing '{keyword}' in {DATA_DIR}")
    return matches[0]


def _load_data():
    players_info = pd.read_csv(_find_file("players_info"))
    round_by_round = pd.read_csv(_find_file("round_by_round"), low_memory=False)
    team_matches = pd.read_csv(_find_file("matches_home_away"))

    canonical_team = {"w. bulldogs": "Western Bulldogs", "western bulldogs": "Western Bulldogs"}

    def clean_team_name(s):
        s = s.str.strip()
        return s.str.lower().map(canonical_team).fillna(s)

    team_matches["team_name"] = clean_team_name(team_matches["team_name"])
    team_matches["opponent"] = clean_team_name(team_matches["opponent"])
    round_by_round["team"] = clean_team_name(round_by_round["team"])
    round_by_round["opponent"] = clean_team_name(round_by_round["opponent"])

    round_by_round["player_id"] = round_by_round["player_id"].astype("Int64")
    players_info["id"] = players_info["id"].astype("Int64")
    team_matches["match_date"] = pd.to_datetime(team_matches["match_date"])
    round_by_round["match_date"] = pd.to_datetime(round_by_round["match_date"])

    for df in (players_info, round_by_round, team_matches):
        df.drop_duplicates(inplace=True)

    round_by_round = round_by_round.merge(
        players_info[["id", "player_name"]], left_on="player_id", right_on="id", how="left"
    )
    return players_info, round_by_round, team_matches


PLAYERS_INFO, ROUND_BY_ROUND, TEAM_MATCHES = _load_data()
KNOWN_TEAMS = sorted(TEAM_MATCHES["team_name"].unique())
KNOWN_PLAYERS = sorted(ROUND_BY_ROUND["player_name"].dropna().unique())


# ---------------------------------------------------------------------------
# Helpers (fuzzy-ish exact-ish name matching so "richmond" matches "Richmond Tigers")
# ---------------------------------------------------------------------------
def _resolve_team(name: str) -> str | None:
    if name in KNOWN_TEAMS:
        return name
    lower = name.strip().lower()
    for t in KNOWN_TEAMS:
        if lower == t.lower() or lower in t.lower():
            return t
    return None


def _resolve_player(name: str) -> str | None:
    if name in KNOWN_PLAYERS:
        return name
    lower = name.strip().lower()
    exact = [p for p in KNOWN_PLAYERS if p.lower() == lower]
    if exact:
        return exact[0]
    partial = [p for p in KNOWN_PLAYERS if lower in p.lower()]
    if len(partial) == 1:
        return partial[0]
    return None


# ---------------------------------------------------------------------------
# Tool 1 — head-to-head team record
# ---------------------------------------------------------------------------
@tool
def get_team_head_to_head(team_a: str, team_b: str) -> str:
    """Get the all-time head-to-head match record between two AFL teams (wins, losses, draws,
    total games played, and the most recent meeting). Use this for any question about how two
    teams have performed against each other historically. Team names should be the club name,
    e.g. 'Richmond Tigers', 'Carlton Blues' — partial names like 'Richmond' also work.
    """
    ta, tb = _resolve_team(team_a), _resolve_team(team_b)
    if ta is None:
        return f"Could not find a team matching '{team_a}'. Known teams include: {KNOWN_TEAMS[:5]}..."
    if tb is None:
        return f"Could not find a team matching '{team_b}'. Known teams include: {KNOWN_TEAMS[:5]}..."

    games = TEAM_MATCHES[(TEAM_MATCHES["team_name"] == ta) & (TEAM_MATCHES["opponent"] == tb)]
    if games.empty:
        return f"No recorded matches found between {ta} and {tb}."

    wins = int((games["result"] == "W").sum())
    losses = int((games["result"] == "L").sum())
    draws = int((games["result"] == "D").sum())
    last = games.sort_values("match_date").iloc[-1]

    return (
        f"Head-to-head, {ta} vs {tb} ({len(games)} games on record): "
        f"{ta} won {wins}, lost {losses}, drew {draws}. "
        f"Most recent meeting: {last['match_date'].date()} at {last['venue']}, "
        f"{ta} {int(last['team_score'])} - {int(last['opponent_score'])} {tb} ({last['result']} for {ta})."
    )


# ---------------------------------------------------------------------------
# Tool 2 — player season stats
# ---------------------------------------------------------------------------
@tool
def get_player_season_stats(player_name: str, year: int) -> str:
    """Get a player's aggregated stats (games played, total & average disposals, goals, and
    fantasy points) for one specific season. Use this for any question about a player's season
    or yearly totals/averages, e.g. 'how many goals did X kick in 2023'.
    """
    player = _resolve_player(player_name)
    if player is None:
        return f"Could not find a player matching '{player_name}' in the dataset."

    rows = ROUND_BY_ROUND[(ROUND_BY_ROUND["player_name"] == player) & (ROUND_BY_ROUND["year"] == year)]
    if rows.empty:
        return f"No recorded games for {player} in {year}."

    games = len(rows)
    return (
        f"{player}, {year} season ({games} games): "
        f"disposals total {int(rows['disposals'].sum())} (avg {rows['disposals'].mean():.1f}), "
        f"goals total {int(rows['goals'].sum())} (avg {rows['goals'].mean():.1f}), "
        f"fantasy points total {int(rows['fantasy_points'].sum())} (avg {rows['fantasy_points'].mean():.1f})."
    )


# ---------------------------------------------------------------------------
# Tool 3 — player's recent game(s) — needed for "last round" / follow-up questions
# ---------------------------------------------------------------------------
@tool
def get_player_recent_games(player_name: str, n_games: int = 1) -> str:
    """Get a player's stats for their most recent N games, most recent first (default 1 = just
    their last game). Use this for questions like 'how many disposals did X have last round?' or
    a follow-up like 'what about the round before that?' (increase n_games and look further back
    in the returned list).
    """
    player = _resolve_player(player_name)
    if player is None:
        return f"Could not find a player matching '{player_name}' in the dataset."

    rows = ROUND_BY_ROUND[ROUND_BY_ROUND["player_name"] == player].sort_values("match_date", ascending=False)
    if rows.empty:
        return f"No recorded games found for {player}."

    rows = rows.head(n_games)
    lines = [f"{player} — most recent {len(rows)} game(s):"]
    for _, r in rows.iterrows():
        lines.append(
            f"  {r['match_date'].date()} R{r['round']} vs {r['opponent']}: "
            f"{int(r['disposals']) if pd.notna(r['disposals']) else 'N/A'} disposals, "
            f"{int(r['goals']) if pd.notna(r['goals']) else 'N/A'} goals, "
            f"{int(r['fantasy_points']) if pd.notna(r['fantasy_points']) else 'N/A'} fantasy points"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4 — player career average — needed for "how does that compare to his career average?"
# ---------------------------------------------------------------------------
@tool
def get_player_career_average(player_name: str) -> str:
    """Get a player's career averages (across every recorded game) for disposals, goals, and
    fantasy points, plus total games played. Use this for any question comparing a recent
    performance to the player's career norm.
    """
    player = _resolve_player(player_name)
    if player is None:
        return f"Could not find a player matching '{player_name}' in the dataset."

    rows = ROUND_BY_ROUND[ROUND_BY_ROUND["player_name"] == player]
    if rows.empty:
        return f"No recorded games found for {player}."

    return (
        f"{player} — career averages across {len(rows)} games: "
        f"disposals {rows['disposals'].mean():.1f}, goals {rows['goals'].mean():.1f}, "
        f"fantasy points {rows['fantasy_points'].mean():.1f}."
    )


ALL_TOOLS = [get_team_head_to_head, get_player_season_stats, get_player_recent_games, get_player_career_average]


if __name__ == "__main__":
    # Quick self-test against real data, no LLM involved — run with: python tools.py
    print(get_team_head_to_head.invoke({"team_a": "Richmond", "team_b": "Carlton"}))
    print()
    print(get_player_season_stats.invoke({"player_name": "Dustin Martin", "year": 2017}))
    print()
    print(get_player_recent_games.invoke({"player_name": "Dustin Martin", "n_games": 3}))
    print()
    print(get_player_career_average.invoke({"player_name": "Dustin Martin"}))
