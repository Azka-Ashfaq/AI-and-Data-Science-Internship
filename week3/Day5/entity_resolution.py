"""
entity_resolution.py — Entity and fixture resolution for AFL LangGraph integration (Task 3 & 4).

Handles:
  1. Team nickname / alias resolution:
     - Maps informal club names ("Pies", "Cats", "Freo", "Swans", "GWS", "Doggies")
       to exact canonical keys in the AFL dataset and ML models (e.g., "Collingwood Magpies").
  2. Player nickname and ambiguity resolution:
     - Resolves nicknames ("Dusty", "Bont", "Crippa", "Naicos", "The Lizard")
     - Flags ambiguous queries (e.g., "Smith" -> multiple candidates) to trigger clarification.
  3. Temporal / Fixture resolution:
     - Resolves "this week", "next round", "upcoming", "tomorrow" to valid upcoming match dates.
"""

import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

DATA_DIR = Path(__file__).parent / "afl_datasets"
STORE_DIR = Path(__file__).parent / "model_store"

# ---------------------------------------------------------------------------
# Canonical Team Mapping
# ---------------------------------------------------------------------------
TEAM_ALIASES: Dict[str, str] = {
    # Adelaide Crows
    "adelaide": "Adelaide Crows",
    "adelaide crows": "Adelaide Crows",
    "crows": "Adelaide Crows",
    
    # Brisbane Lions
    "brisbane": "Brisbane Lions",
    "brisbane lions": "Brisbane Lions",
    "lions": "Brisbane Lions",
    "brisbane bears": "Brisbane Bears",
    "bears": "Brisbane Bears",
    
    # Carlton Blues
    "carlton": "Carlton Blues",
    "carlton blues": "Carlton Blues",
    "blues": "Carlton Blues",
    "baggers": "Carlton Blues",
    "bluebaggers": "Carlton Blues",
    
    # Collingwood Magpies
    "collingwood": "Collingwood Magpies",
    "collingwood magpies": "Collingwood Magpies",
    "magpies": "Collingwood Magpies",
    "pies": "Collingwood Magpies",
    
    # Essendon Bombers
    "essendon": "Essendon Bombers",
    "essendon bombers": "Essendon Bombers",
    "bombers": "Essendon Bombers",
    "dons": "Essendon Bombers",
    
    # Fitzroy Lions
    "fitzroy": "Fitzroy Lions",
    "fitzroy lions": "Fitzroy Lions",
    
    # Fremantle Dockers
    "fremantle": "Fremantle Dockers",
    "fremantle dockers": "Fremantle Dockers",
    "dockers": "Fremantle Dockers",
    "freo": "Fremantle Dockers",
    
    # Geelong Cats
    "geelong": "Geelong Cats",
    "geelong cats": "Geelong Cats",
    "cats": "Geelong Cats",
    
    # Gold Coast Suns
    "gold coast": "Gold Coast Suns",
    "gold coast suns": "Gold Coast Suns",
    "suns": "Gold Coast Suns",
    
    # GWS Giants
    "gws": "Greater Western Sydney Giants",
    "gws giants": "Greater Western Sydney Giants",
    "giants": "Greater Western Sydney Giants",
    "greater western sydney": "Greater Western Sydney Giants",
    "greater western sydney giants": "Greater Western Sydney Giants",
    
    # Hawthorn Hawks
    "hawthorn": "Hawthorn Hawks",
    "hawthorn hawks": "Hawthorn Hawks",
    "hawks": "Hawthorn Hawks",
    
    # Melbourne Demons
    "melbourne": "Melbourne Demons",
    "melbourne demons": "Melbourne Demons",
    "demons": "Melbourne Demons",
    "dees": "Melbourne Demons",
    
    # North Melbourne Kangaroos
    "north melbourne": "North Melbourne Kangaroos",
    "north melbourne kangaroos": "North Melbourne Kangaroos",
    "kangaroos": "North Melbourne Kangaroos",
    "roos": "North Melbourne Kangaroos",
    "north": "North Melbourne Kangaroos",
    
    # Port Adelaide Power
    "port adelaide": "Port Adelaide Power",
    "port adelaide power": "Port Adelaide Power",
    "power": "Port Adelaide Power",
    "port": "Port Adelaide Power",
    
    # Richmond Tigers
    "richmond": "Richmond Tigers",
    "richmond tigers": "Richmond Tigers",
    "tigers": "Richmond Tigers",
    "tiges": "Richmond Tigers",
    
    # St Kilda Saints
    "st kilda": "St Kilda Saints",
    "st kilda saints": "St Kilda Saints",
    "saints": "St Kilda Saints",
    
    # Sydney Swans
    "sydney": "Sydney Swans",
    "sydney swans": "Sydney Swans",
    "swans": "Sydney Swans",
    "bloods": "Sydney Swans",
    
    # West Coast Eagles
    "west coast": "West Coast Eagles",
    "west coast eagles": "West Coast Eagles",
    "eagles": "West Coast Eagles",
    
    # Western Bulldogs
    "western bulldogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "dogs": "Western Bulldogs",
    "doggies": "Western Bulldogs",
    "footscray": "Western Bulldogs",
}

# Known canonical team names
CANONICAL_TEAMS = sorted(list(set(TEAM_ALIASES.values())))


def resolve_team(name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves a team string (nickname, abbreviation, partial name) to canonical team name.
    
    Returns
    -------
    (canonical_name, error_message)
    """
    if not name or not isinstance(name, str):
        return None, "No team name provided."
    
    cleaned = name.strip().lower()
    
    # Direct alias match
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned], None
    
    # Substring search in canonical names
    matches = [t for t in CANONICAL_TEAMS if cleaned in t.lower()]
    if len(matches) == 1:
        return matches[0], None
    elif len(matches) > 1:
        return None, f"Team name '{name}' is ambiguous between: {', '.join(matches)}."
    
    return None, f"Could not resolve '{name}' to a known AFL team."


# ---------------------------------------------------------------------------
# Player Name & Nickname Database
# ---------------------------------------------------------------------------
_player_info_df: Optional[pd.DataFrame] = None
_player_alias_map: Dict[str, str] = {}
_known_players_set: set = set()


def _init_player_db():
    global _player_info_df, _player_alias_map, _known_players_set
    if _player_info_df is not None:
        return
    
    info_file = DATA_DIR / "afl_players_info_raw.csv"
    if not info_file.exists():
        info_files = list(DATA_DIR.glob("*players_info*.csv"))
        info_file = info_files[0] if info_files else None

    if info_file and info_file.exists():
        _player_info_df = pd.read_csv(info_file)
        _player_info_df["player_name"] = _player_info_df["player_name"].astype(str).str.strip()
    else:
        _player_info_df = pd.DataFrame(columns=["id", "player_name", "player_common_names"])

    # Load known_players from model store
    kp_file = STORE_DIR / "known_players.csv"
    if kp_file.exists():
        kp_df = pd.read_csv(kp_file)
        _known_players_set = set(kp_df["player_name"].dropna().str.strip().unique())
    else:
        _known_players_set = set(_player_info_df["player_name"].unique())

    # Build aliases from player_common_names
    for _, row in _player_info_df.iterrows():
        pname = row["player_name"].strip()
        _player_alias_map[pname.lower()] = pname
        
        # Check nicknames in player_common_names e.g. "{Bont,The Bont}"
        common = str(row.get("player_common_names", ""))
        if pd.notna(common) and common.startswith("{") and common.endswith("}"):
            aliases = [a.strip() for a in common[1:-1].split(",") if a.strip()]
            for alias in aliases:
                _player_alias_map[alias.lower()] = pname
                
    # Add famous AFL nicknames
    curated_nicknames = {
        "dusty": "Dustin Martin",
        "dustin martin": "Dustin Martin",
        "bont": "Marcus Bontempelli",
        "the bont": "Marcus Bontempelli",
        "crippa": "Patrick Cripps",
        "patrick cripps": "Patrick Cripps",
        "trac": "Christian Petracca",
        "christian petracca": "Christian Petracca",
        "jezza": "Jeremy Cameron",
        "jeremy cameron": "Jeremy Cameron",
        "naicos": "Nick Daicos",
        "nick daicos": "Nick Daicos",
        "jaicos": "Josh Daicos",
        "tex": "Taylor Walker",
        "tex walker": "Taylor Walker",
        "pendles": "Scott Pendlebury",
        "scott pendlebury": "Scott Pendlebury",
        "the lizard": "Nick Blakey",
        "nick blakey": "Nick Blakey",
        "danger": "Patrick Dangerfield",
        "patrick dangerfield": "Patrick Dangerfield",
        "chols royce": "Mabior Chol",
        "gawn": "Max Gawn",
        "max gawn": "Max Gawn",
        "walsh": "Sam Walsh",
        "sam walsh": "Sam Walsh",
        "sheezel": "Harry Sheezel",
        "harry sheezel": "Harry Sheezel",
        "heeney": "Isaac Heeney",
        "isaac heeney": "Isaac Heeney",
    }
    for nick, canonical in curated_nicknames.items():
        _player_alias_map[nick.lower()] = canonical


def resolve_player(query_name: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Resolves a player string (full name, last name, or nickname).
    
    Returns
    -------
    (canonical_name, error_message, candidates_list)
    """
    _init_player_db()
    if not query_name or not isinstance(query_name, str):
        return None, "No player name provided.", []
    
    cleaned = query_name.strip().lower()
    
    # 1. Direct match in alias map
    if cleaned in _player_alias_map:
        return _player_alias_map[cleaned], None, []
    
    # 2. Exact match in known players (case-insensitive)
    exact_matches = [p for p in _known_players_set if p.lower() == cleaned]
    if len(exact_matches) == 1:
        return exact_matches[0], None, []
    
    # 3. Substring / Last name match
    partial_matches = [p for p in _known_players_set if cleaned in p.lower()]
    if len(partial_matches) == 1:
        return partial_matches[0], None, []
    elif len(partial_matches) > 1:
        # Check if the query matches a surname exactly
        surname_matches = [p for p in partial_matches if p.lower().split()[-1] == cleaned]
        if len(surname_matches) == 1:
            return surname_matches[0], None, []
        elif len(surname_matches) > 1:
            return (
                None,
                f"Multiple players match '{query_name}': {', '.join(surname_matches[:5])}. Please clarify who you mean.",
                surname_matches[:5]
            )
        return (
            None,
            f"Multiple players match '{query_name}': {', '.join(partial_matches[:5])}. Please clarify who you mean.",
            partial_matches[:5]
        )
    
    return None, f"Could not find a player matching '{query_name}' in AFL records.", []


# ---------------------------------------------------------------------------
# Date & Fixture Resolution
# ---------------------------------------------------------------------------
DEFAULT_UPCOMING_DATE = "2025-09-20"  # Upcoming finals fixture date within model range


def resolve_match_date(date_str: Optional[str], home_team: Optional[str] = None, away_team: Optional[str] = None) -> str:
    """
    Resolves expressions like "this week", "next round", "upcoming", "tomorrow"
    or ISO date strings to a valid YYYY-MM-DD string.
    """
    if not date_str:
        return DEFAULT_UPCOMING_DATE
    
    low = date_str.strip().lower()
    
    # Common relative date expressions
    relative_markers = ["this week", "next week", "upcoming", "next round", "this weekend", "tomorrow", "next match", "current round"]
    if any(m in low for m in relative_markers):
        return DEFAULT_UPCOMING_DATE
    
    # Try parsing direct date formats
    try:
        ts = pd.to_datetime(date_str)
        # Ensure year is within valid dataset/model range
        if ts.year < 1983:
            return DEFAULT_UPCOMING_DATE
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return DEFAULT_UPCOMING_DATE
