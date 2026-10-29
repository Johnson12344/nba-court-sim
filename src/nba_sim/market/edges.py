from __future__ import annotations

import math
import unicodedata
from datetime import datetime, timezone

import numpy as np

from nba_sim.market.events import age_minutes, utc_time
from nba_sim.models import SimulatedGame, TeamProfile

PROP_STATS = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists"}


def normalized_name(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value).casefold() if c.isalnum() and not unicodedata.combining(c))


def return_statistics(values: np.ndarray, price: float) -> dict:
    """values > 0 win, == 0 push, < 0 lose; one unit staked, decimal prices."""
    win = float(np.mean(values > 0))
    push = float(np.mean(values == 0))
    loss = float(np.mean(values < 0))
    conditional = win / (win + loss) if win + loss else None
    ev = win * (price - 1) - loss
    # One-sided 95% Hoeffding bound; covers simulation sampling only, not model error.
    error = price * math.sqrt(math.log(20) / (2 * len(values)))
    return {"win_probability": win, "push_probability": push, "loss_probability": loss,
            "conditional_win_probability": conditional, "break_even_probability": 1 / price,
            "fair_decimal_odds": (1 - push) / win if win else None,
            "expected_profit_per_unit": ev, "mc_lower_profit_per_unit": max(-1.0, ev - error),
            "simulations": len(values)}


def _number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _opposite_price(market: dict, outcome: dict, event: dict) -> float | None:
    key = market["key"]
    matches = []
    for other in market.get("outcomes", []):
        if other is outcome:
            continue
        if key == "h2h":
            paired = other.get("name") in {event["home_team"], event["away_team"]} and other.get("name") != outcome.get("name")
        elif key == "spreads":
            paired = (other.get("name") in {event["home_team"], event["away_team"]}
                      and other.get("name") != outcome.get("name")
                      and _number(other.get("point")) == -float(outcome["point"]))
        else:
            paired = (other.get("name") == ("Under" if outcome.get("name") == "Over" else "Over")
                      and _number(other.get("point")) == float(outcome["point"])
                      and other.get("description") == outcome.get("description"))
        price = _number(other.get("price"))
        if paired and price is not None and price > 1:
            matches.append(price)
    return matches[0] if len(matches) == 1 else None


def build_edge_report(cache: dict, event: dict, games: list[SimulatedGame],
                      home: TeamProfile, away: TeamProfile, *, readiness_reasons: list[str],
                      max_odds_age_minutes: float = 60, min_ev: float = 0.02,
                      now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if not games:
        raise ValueError("Market probabilities need simulated games")
    if not math.isfinite(max_odds_age_minutes) or max_odds_age_minutes <= 0:
        raise ValueError("Maximum odds age must be positive")
    if not math.isfinite(min_ev) or min_ev < 0:
        raise ValueError("Minimum EV must be finite and nonnegative")
    reasons = list(readiness_reasons)
    kickoff = utc_time(event.get("commence_time"))
    if kickoff is None or kickoff <= now:
        reasons.append("Fixture has started or its start time is invalid")
    elif (kickoff - now).total_seconds() > 24 * 3600:
        reasons.append("Review availability within 24 hours of tip-off")
    cache_age = age_minutes(cache.get("retrieved_at"), now)
    if cache_age is None or cache_age > max_odds_age_minutes:
        reasons.append("Refresh odds: download timestamp is stale or invalid")
    if len(games) < 500:
        reasons.append("Use at least 500 simulations before reviewing apparent edges")
    names = {}
    for team in (home, away):
        for player in team.players:
            names.setdefault(normalized_name(player.name), []).append((team.abbreviation, player))
    margin = np.asarray([game.margin for game in games], dtype=float)
    total = np.asarray([game.total for game in games], dtype=float)
    samples = {}
    offers, skipped = [], []
    books = event.get("bookmakers", []) + event.get("player_prop_bookmakers", [])
    for book in books:
        for market in book.get("markets", []):
            key = market.get("key")
            if key not in {"h2h", "spreads", "totals", *PROP_STATS}:
                continue
            if key == "h2h" and (len(market.get("outcomes", [])) != 2 or
                                  {o.get("name") for o in market["outcomes"]} != {event["home_team"], event["away_team"]}):
                skipped.append("Unsupported moneyline market (requires two outcomes including overtime)")
                continue
            timestamp = market.get("last_update") or book.get("last_update")
            offer_age = age_minutes(timestamp, now)
            for outcome in market.get("outcomes", []):
                side = outcome.get("name")
                price = _number(outcome.get("price"))
                line = _number(outcome.get("point")) if key != "h2h" else None
                if price is None or price <= 1 or (key != "h2h" and line is None):
                    skipped.append("Invalid price or line")
                    continue
                if line is not None and not math.isclose(line * 2, round(line * 2)):
                    skipped.append("Only whole and half-point lines are supported")
                    continue
                player_id, player_name, team_code = None, None, None
                if key in {"h2h", "spreads"}:
                    if side not in {event["home_team"], event["away_team"]}:
                        continue
                    values = margin if side == event["home_team"] else -margin
                    values = values + line if key == "spreads" else values
                    team_code = home.abbreviation if side == event["home_team"] else away.abbreviation
                else:
                    if side not in {"Over", "Under"}:
                        continue
                    if key == "totals":
                        raw_values = total
                    else:
                        found = names.get(normalized_name(str(outcome.get("description", ""))), [])
                        if len(found) != 1:
                            skipped.append(f"Unmatched, excluded, or ambiguous player: {outcome.get('description', '')}")
                            continue
                        team_code, player = found[0]
                        player_id, player_name = player.player_id, player.name
                        sample_key = (team_code, player_id, key)
                        if sample_key not in samples:
                            boxes = [game.player_stats.get(team_code, {}).get(player_id) for game in games]
                            # A missing box is not silently counted as zero (e.g. score-only model).
                            samples[sample_key] = None if any(box is None for box in boxes) else np.asarray([
                                box.oreb + box.dreb if key == "player_rebounds" else getattr(box, PROP_STATS[key]) for box in boxes
                            ], dtype=float)
                        raw_values = samples[sample_key]
                        if raw_values is None:
                            skipped.append(f"No complete player simulations: {player_name}")
                            continue
                    values = raw_values - line if side == "Over" else line - raw_values
                stats = return_statistics(values, price)
                opposite = _opposite_price(market, outcome, event)
                no_vig = (1 / price) / (1 / price + 1 / opposite) if opposite else None
                blockers = list(reasons)
                if offer_age is None or offer_age > max_odds_age_minutes:
                    blockers.append("Bookmaker quote timestamp is stale or invalid")
                if not book.get("key"):
                    blockers.append("Missing bookmaker identity")
                conditional = stats["conditional_win_probability"]
                offers.append({"event_id": str(event["id"]), "bookmaker": book.get("key", "unknown"),
                               "market": key, "selection": side, "line": line, "decimal_odds": price,
                               "player_id": player_id, "player": player_name, "team": team_code,
                               "quote_updated_at": timestamp, "quote_age_minutes": offer_age,
                               **stats, "market_no_vig_probability": no_vig,
                               "edge_vs_no_vig": conditional - no_vig if conditional is not None and no_vig is not None else None,
                               "review_candidate": not blockers and stats["expected_profit_per_unit"] >= min_ev and stats["mc_lower_profit_per_unit"] > 0,
                               "blockers": sorted(set(blockers))})
    offers.sort(key=lambda row: (not row["review_candidate"], -row["expected_profit_per_unit"]))
    return {"event_id": str(event["id"]), "commence_time": event.get("commence_time"),
            "created_at": now.isoformat(), "min_ev": min_ev, "max_odds_age_minutes": max_odds_age_minutes,
            "readiness_reasons": sorted(set(reasons)), "offers": offers, "skipped": sorted(set(skipped)),
            "validation_status": "unvalidated",
            "interpretation": "Apparent model edge, not a demonstrated betting advantage. EV is expected profit per unit staked. The lower bound covers Monte Carlo sampling only, not model error or searching many offers. No staking advice.",
            "settlement_assumption": "Full-game markets include overtime; whole-number ties push. Check your bookmaker's rules, especially player participation and voids."}
