from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from nba_sim.market.odds_api import NBA_TEAM_NAMES


def utc_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def age_minutes(value: Any, now: datetime) -> float | None:
    parsed = utc_time(value)
    if parsed is None or parsed > now:
        return None
    return (now - parsed).total_seconds() / 60


def upcoming_events(cache: dict, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    return sorted(
        [event for event in cache.get("events", [])
         if event.get("id") and (utc_time(event.get("commence_time")) or now) > now],
        key=lambda event: utc_time(event["commence_time"]),
    )


def select_event(cache: dict, home: str, away: str, event_id: str | None = None,
                 now: datetime | None = None) -> dict | None:
    matches = [event for event in upcoming_events(cache, now)
               if event.get("home_team") == NBA_TEAM_NAMES.get(home, home)
               and event.get("away_team") == NBA_TEAM_NAMES.get(away, away)]
    if event_id:
        match = next((event for event in matches if str(event["id"]) == event_id), None)
        if match is None:
            raise ValueError("Event ID does not match an upcoming cached home/away fixture. Run refresh-odds and fixtures.")
        return match
    return matches[0] if matches else None
