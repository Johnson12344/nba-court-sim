from __future__ import annotations

import csv
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from nba_sim.market.events import age_minutes
from nba_sim.models import TeamProfile

FIELDS = ["event_id", "team", "player_id", "player_name", "status", "suggested_minutes",
          "expected_minutes", "starter", "reviewed_at", "source"]


def write_lineup_template(path: Path, teams: tuple[TeamProfile, TeamProfile], event_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for team in teams:
            for player in team.players:
                writer.writerow({"event_id": event_id, "team": team.abbreviation,
                                 "player_id": player.player_id, "player_name": player.name,
                                 "status": "unreviewed", "suggested_minutes": round(player.expected_minutes, 2)})


def _allocate(players, fixed: dict[str, float]) -> dict[str, float]:
    targets = dict(fixed)
    remaining = {p.player_id: p.expected_minutes for p in players if p.player_id not in fixed}
    budget = 240.0 - sum(fixed.values())
    if budget < -1e-6 or budget > 48 * len(remaining) + 1e-6:
        raise ValueError("Expected minutes must allow exactly 240 team minutes; adjust fixed targets or leave more blank.")
    while remaining:
        total = sum(remaining.values())
        weights = remaining if total > 0 else {key: 1.0 for key in remaining}
        total = sum(weights.values())
        shares = {key: budget * weight / total for key, weight in weights.items()}
        capped = [key for key, value in shares.items() if value > 48]
        if not capped:
            targets.update(shares)
            break
        for key in capped:
            targets[key] = 48.0
            budget -= 48.0
            del remaining[key]
    if abs(sum(targets.values()) - 240) > 1e-5:
        raise ValueError("Fixed expected minutes must total 240 when no automatic allocations remain.")
    return targets


def apply_lineups(path: Path, teams: tuple[TeamProfile, TeamProfile], event_id: str,
                  now: datetime | None = None) -> tuple[tuple[TeamProfile, TeamProfile], dict]:
    now = now or datetime.now(timezone.utc)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS) - {"suggested_minutes"} <= set(reader.fieldnames or []):
            raise ValueError("Lineup CSV has missing columns. Generate it with --write-lineups first.")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    profiles = {(t.abbreviation, p.player_id): p for t in teams for p in t.players}
    indexed = {}
    reasons = []
    for row in rows:
        key = (row["team"], row["player_id"])
        if key not in profiles or key in indexed:
            raise ValueError(f"Unknown or duplicate lineup player: {key}")
        if row["event_id"] != event_id:
            raise ValueError("Lineup event_id must match the selected upcoming fixture.")
        if row["player_name"] != profiles[key].name:
            raise ValueError(f"Lineup player name and ID do not match: {key}")
        row["status"] = row["status"].lower()
        if row["status"] not in {"active", "out", "questionable", "unreviewed"}:
            raise ValueError(f"Invalid lineup status for {profiles[key].name}")
        if row["starter"].lower() not in {"", "true", "false"}:
            raise ValueError("starter must be blank, true, or false")
        reviewed_age = age_minutes(row["reviewed_at"], now)
        if row["status"] in {"questionable", "unreviewed"}:
            reasons.append(f"Availability unresolved: {profiles[key].name}")
        if not row["source"] or reviewed_age is None or reviewed_age > 720:
            reasons.append(f"Availability needs a sourced review within 12 hours: {profiles[key].name}")
        indexed[key] = row
    if set(indexed) != set(profiles):
        raise ValueError("Lineup CSV must cover every modeled player on both teams, including players marked out.")
    updated = []
    for team in teams:
        players, fixed = [], {}
        for player in team.players:
            row = indexed[(team.abbreviation, player.player_id)]
            minutes = float(row["expected_minutes"]) if row["expected_minutes"] else None
            if minutes is not None and (not math.isfinite(minutes) or not 0 <= minutes <= 48):
                raise ValueError(f"Expected minutes must be between 0 and 48: {player.name}")
            if row["status"] == "out":
                if minutes not in {None, 0.0}:
                    raise ValueError(f"An out player cannot have positive minutes: {player.name}")
                continue
            if minutes == 0:
                continue
            starter = player.starter if not row["starter"] else row["starter"].lower() == "true"
            players.append(replace(player, starter=starter))
            if minutes is not None:
                fixed[player.player_id] = minutes
        if len(players) < 5:
            raise ValueError(f"{team.abbreviation} needs at least five players with positive minutes.")
        if sum(p.starter for p in players) > 5:
            raise ValueError(f"{team.abbreviation} has more than five starters; edit the starter column.")
        targets = _allocate(players, fixed)
        players = [replace(p, expected_minutes=targets[p.player_id]) for p in players if targets[p.player_id] > 0]
        updated.append(replace(team, players=tuple(players), data_evidence={
            **team.data_evidence, "fixed_minutes": fixed, "availability_file": str(path),
        }))
    return tuple(updated), {"reviewed": not reasons, "reasons": sorted(set(reasons)), "rows": rows,
                            "note": "User-reviewed availability; expected minutes are targets, not guaranteed playing time."}
