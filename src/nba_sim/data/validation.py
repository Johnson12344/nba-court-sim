from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TEAM_REQUIRED = {
    "team_id", "team_abbreviation", "team_name", "season", "games", "minutes",
    "points", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb",
    "assists", "turnovers",
}
PLAYER_REQUIRED = {
    "player_id", "player_name", "team_abbreviation", "season", "games", "minutes",
    "points", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb",
    "assists", "turnovers", "fouls",
}


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _validate_frame(frame: pd.DataFrame, required: set[str], entity: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{entity}: missing required columns: {', '.join(missing)}")
        return errors, warnings
    if frame.empty:
        errors.append(f"{entity}: file has no rows")
    id_col = "team_id" if entity == "teams" else "player_id"
    if frame[id_col].isna().any() or frame[id_col].astype(str).str.strip().eq("").any():
        errors.append(f"{entity}: invalid or missing {id_col}")
    key = [id_col, "season"]
    duplicates = int(frame.duplicated(key).sum())
    if duplicates:
        errors.append(f"{entity}: {duplicates} duplicated {key} rows")
    numeric = sorted((required - {id_col, "team_abbreviation", "team_name", "player_name", "season"}) & set(frame.columns))
    for column in numeric:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            errors.append(f"{entity}: {column} contains missing or non-numeric values")
        if (converted < 0).any():
            errors.append(f"{entity}: {column} contains negative values")
    for made, attempted in (("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")):
        if made in frame and attempted in frame and (pd.to_numeric(frame[made], errors="coerce") > pd.to_numeric(frame[attempted], errors="coerce")).any():
            errors.append(f"{entity}: {made} exceeds {attempted}")
    return errors, warnings


def validate_data(teams: pd.DataFrame, players: pd.DataFrame) -> ValidationReport:
    te, tw = _validate_frame(teams, TEAM_REQUIRED, "teams")
    pe, pw = _validate_frame(players, PLAYER_REQUIRED, "players")
    if not te and not pe:
        unknown = sorted(set(players.team_abbreviation) - set(teams.team_abbreviation))
        if unknown:
            pe.append(f"players: team abbreviations missing from team data: {', '.join(unknown)}")
        if (players.fg3a > players.fga).any():
            pe.append("players: three-point attempts exceed field-goal attempts")
    return ValidationReport(tuple(te + pe), tuple(tw + pw))


def require_valid(teams: pd.DataFrame, players: pd.DataFrame) -> None:
    report = validate_data(teams, players)
    if not report.valid:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(report.errors))

