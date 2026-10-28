from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.the-odds-api.com/v4"
NBA_SPORT_KEY = "basketball_nba"
CORE_MARKETS = ("h2h", "spreads", "totals")
PLAYER_MARKETS = ("player_points", "player_rebounds", "player_assists")
NBA_TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


class OddsApiError(RuntimeError):
    """A safe, API-key-free error from The Odds API."""


def _team_name(value: str) -> str:
    return NBA_TEAM_NAMES.get(value.strip().upper(), value.strip())


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _request(path: str, api_key: str, params: dict[str, str], timeout: int) -> tuple[Any, dict[str, int | None]]:
    request_params = {**params, "apiKey": api_key}
    try:
        response = requests.get(f"{BASE_URL}/{path.lstrip('/')}", params=request_params, timeout=timeout)
    except requests.RequestException as exc:
        raise OddsApiError("Could not connect to The Odds API. Check the internet connection and try again.") from exc
    if response.status_code != 200:
        try:
            body = response.json()
            detail = body.get("message") or body.get("error") or "request rejected"
        except (ValueError, AttributeError):
            detail = "request rejected"
        safe_detail = str(detail).replace(api_key, "[redacted]")
        raise OddsApiError(f"The Odds API returned HTTP {response.status_code}: {safe_detail}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise OddsApiError("The Odds API returned an invalid response.") from exc
    quota: dict[str, int | None] = {}
    for label, header in (
        ("remaining", "x-requests-remaining"),
        ("used", "x-requests-used"),
        ("last", "x-requests-last"),
    ):
        raw = response.headers.get(header)
        try:
            quota[label] = int(raw) if raw is not None else None
        except ValueError:
            quota[label] = None
    return payload, quota


def find_matchup(events: list[dict[str, Any]], home: str, away: str) -> dict[str, Any] | None:
    expected_home = _normalized(_team_name(home))
    expected_away = _normalized(_team_name(away))
    matches = [
        event for event in events
        if _normalized(str(event.get("home_team", ""))) == expected_home
        and _normalized(str(event.get("away_team", ""))) == expected_away
    ]
    if not matches:
        return None
    return min(matches, key=lambda event: str(event.get("commence_time", "")))


def refresh_nba_odds(
    raw_root: Path,
    processed_root: Path,
    api_key: str,
    *,
    regions: str = "us",
    bookmakers: str | None = None,
    home: str | None = None,
    away: str | None = None,
    include_player_props: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    if not api_key.strip():
        raise OddsApiError("THE_ODDS_API_KEY is empty.")
    if bool(home) != bool(away):
        raise ValueError("Provide both --home and --away, or neither.")
    if include_player_props and not (home and away):
        raise ValueError("--include-player-props requires --home and --away.")

    params = {
        "regions": regions,
        "markets": ",".join(CORE_MARKETS),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    events, core_quota = _request(f"sports/{NBA_SPORT_KEY}/odds", api_key, params, timeout)
    if not isinstance(events, list):
        raise OddsApiError("The Odds API returned an unexpected NBA odds format.")

    selected = find_matchup(events, home, away) if home and away else None
    props_quota: dict[str, int | None] | None = None
    if include_player_props:
        if selected is None:
            raise OddsApiError(f"No upcoming {away} at {home} event with sportsbook odds was found.")
        prop_params = {
            "regions": regions,
            "markets": ",".join(PLAYER_MARKETS),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if bookmakers:
            prop_params["bookmakers"] = bookmakers
        props, props_quota = _request(
            f"sports/{NBA_SPORT_KEY}/events/{selected['id']}/odds", api_key, prop_params, timeout,
        )
        selected["player_prop_bookmakers"] = props.get("bookmakers", []) if isinstance(props, dict) else []

    retrieved_at = datetime.now(timezone.utc).isoformat()
    cache = {
        "provider": "The Odds API",
        "sport_key": NBA_SPORT_KEY,
        "retrieved_at": retrieved_at,
        "regions": regions,
        "bookmakers_filter": bookmakers,
        "events": events,
        "quota": {"core": core_quota, "player_props": props_quota},
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (raw_root / f"the_odds_api_nba_{stamp}.json").write_text(
        json.dumps(cache, indent=2), encoding="utf-8",
    )
    cache_path = processed_root / "odds_nba.json"
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def load_odds_cache(processed_root: Path) -> dict[str, Any] | None:
    path = processed_root / "odds_nba.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _markets(event: dict[str, Any], include_props: bool = False):
    key = "player_prop_bookmakers" if include_props else "bookmakers"
    for book in event.get(key, []):
        for market in book.get("markets", []):
            yield str(book.get("key") or book.get("title") or "unknown"), market


def _median_or_none(values: list[float]) -> float | None:
    return round(float(median(values)), 3) if values else None


def _game_consensus(event: dict[str, Any]) -> dict[str, Any]:
    home = str(event.get("home_team", ""))
    away = str(event.get("away_team", ""))
    home_spreads: list[float] = []
    totals: list[float] = []
    home_probabilities: list[float] = []
    away_probabilities: list[float] = []
    books: set[str] = set()
    for book, market in _markets(event):
        key = market.get("key")
        outcomes = market.get("outcomes", [])
        if key == "spreads":
            outcome = next((item for item in outcomes if item.get("name") == home), None)
            if outcome is not None and outcome.get("point") is not None:
                home_spreads.append(float(outcome["point"]))
                books.add(book)
        elif key == "totals":
            outcome = next((item for item in outcomes if str(item.get("name", "")).casefold() == "over"), None)
            if outcome is not None and outcome.get("point") is not None:
                totals.append(float(outcome["point"]))
                books.add(book)
        elif key == "h2h":
            home_outcome = next((item for item in outcomes if item.get("name") == home), None)
            away_outcome = next((item for item in outcomes if item.get("name") == away), None)
            if home_outcome and away_outcome:
                home_price = float(home_outcome.get("price", 0))
                away_price = float(away_outcome.get("price", 0))
                if home_price > 1 and away_price > 1:
                    home_raw, away_raw = 1 / home_price, 1 / away_price
                    total_raw = home_raw + away_raw
                    home_probabilities.append(home_raw / total_raw)
                    away_probabilities.append(away_raw / total_raw)
                    books.add(book)
    return {
        "sportsbook_count": len(books),
        "home_spread": _median_or_none(home_spreads),
        "total": _median_or_none(totals),
        "home_no_vig_win_probability": _median_or_none(home_probabilities),
        "away_no_vig_win_probability": _median_or_none(away_probabilities),
    }


def _player_consensus(event: dict[str, Any], players: pd.DataFrame) -> list[dict[str, Any]]:
    stat_map = {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
    }
    lines: dict[tuple[str, str], list[float]] = {}
    books: dict[tuple[str, str], set[str]] = {}
    for book, market in _markets(event, include_props=True):
        stat = stat_map.get(str(market.get("key")))
        if stat is None:
            continue
        for outcome in market.get("outcomes", []):
            if str(outcome.get("name", "")).casefold() != "over" or outcome.get("point") is None:
                continue
            player = str(outcome.get("description", "")).strip()
            if not player:
                continue
            key = (_normalized(player), stat)
            lines.setdefault(key, []).append(float(outcome["point"]))
            books.setdefault(key, set()).add(book)
    comparisons: list[dict[str, Any]] = []
    if players.empty:
        return comparisons
    for row in players.itertuples():
        key = (_normalized(str(row.player_name)), str(row.stat))
        if key not in lines:
            continue
        line = float(median(lines[key]))
        model_mean = float(row.mean)
        comparisons.append({
            "player": row.player_name,
            "team": row.team,
            "stat": row.stat,
            "model_mean": round(model_mean, 3),
            "market_line": round(line, 3),
            "model_minus_line": round(model_mean - line, 3),
            "sportsbook_count": len(books[key]),
        })
    return sorted(comparisons, key=lambda item: abs(item["model_minus_line"]), reverse=True)


def build_market_comparison(
    cache: dict[str, Any],
    home_abbreviation: str,
    away_abbreviation: str,
    summary: dict[str, Any],
    players: pd.DataFrame,
) -> dict[str, Any] | None:
    event = find_matchup(cache.get("events", []), home_abbreviation, away_abbreviation)
    if event is None:
        return None
    consensus = _game_consensus(event)
    home_summary = summary["teams"][home_abbreviation]
    away_summary = summary["teams"][away_abbreviation]
    model_home_score = float(home_summary["score"]["mean"])
    model_away_score = float(away_summary["score"]["mean"])
    model_home_win = float(home_summary["win_rate"])
    market_margin = -consensus["home_spread"] if consensus["home_spread"] is not None else None
    cache_age_hours = None
    try:
        retrieved = datetime.fromisoformat(str(cache.get("retrieved_at", "")).replace("Z", "+00:00"))
        cache_age_hours = max(0.0, (datetime.now(timezone.utc) - retrieved).total_seconds() / 3600)
    except ValueError:
        pass
    result: dict[str, Any] = {
        "provider": cache.get("provider", "The Odds API"),
        "retrieved_at": cache.get("retrieved_at"),
        "event_id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        **consensus,
        "model_home_score": round(model_home_score, 3),
        "model_away_score": round(model_away_score, 3),
        "model_home_margin": round(model_home_score - model_away_score, 3),
        "model_total": round(model_home_score + model_away_score, 3),
        "model_home_win_probability": round(model_home_win, 4),
        "cache_age_hours": round(cache_age_hours, 2) if cache_age_hours is not None else None,
        "is_stale": cache_age_hours is None or cache_age_hours > 6,
        "model_minus_market_margin": (
            round((model_home_score - model_away_score) - market_margin, 3)
            if market_margin is not None else None
        ),
        "model_minus_market_total": (
            round((model_home_score + model_away_score) - consensus["total"], 3)
            if consensus["total"] is not None else None
        ),
        "player_props": _player_consensus(event, players),
        "disclaimer": "Market lines are comparison context only, not betting recommendations.",
    }
    return result
