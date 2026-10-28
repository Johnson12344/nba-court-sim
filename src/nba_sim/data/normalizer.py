from __future__ import annotations

import re

import pandas as pd


def _snake(name: str) -> str:
    separated = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", separated.lower())).strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={column: _snake(str(column)) for column in frame.columns}).copy()


TEAM_COLUMN_MAP = {
    "team_id": "team_id", "team_abbreviation": "team_abbreviation", "team_name": "team_name",
    "gp": "games", "min": "minutes", "pts": "points", "fga": "fga", "fgm": "fgm",
    "fg3a": "fg3a", "fg3m": "fg3m", "fta": "fta", "ftm": "ftm", "oreb": "oreb",
    "dreb": "dreb", "ast": "assists", "tov": "turnovers", "pace": "pace",
}
PLAYER_COLUMN_MAP = {
    "player_id": "player_id", "player_name": "player_name", "team_abbreviation": "team_abbreviation",
    "gp": "games", "w": "wins", "min": "minutes", "pts": "points", "fga": "fga", "fgm": "fgm",
    "fg3a": "fg3a", "fg3m": "fg3m", "fta": "fta", "ftm": "ftm", "oreb": "oreb",
    "dreb": "dreb", "ast": "assists", "tov": "turnovers", "pf": "fouls", "stl": "steals",
    "blk": "blocks",
}


def normalize_nba_frame(frame: pd.DataFrame, kind: str, season: str) -> pd.DataFrame:
    mapping = TEAM_COLUMN_MAP if kind == "team" else PLAYER_COLUMN_MAP
    result = frame.rename(columns={column: mapping.get(_snake(column), _snake(column)) for column in frame.columns}).copy()
    result["season"] = season
    return result

