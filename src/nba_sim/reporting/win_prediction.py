"""Winner probabilities for this matchup, separate from historical accuracy."""
from __future__ import annotations

from datetime import datetime, timezone
import math

from nba_sim.advisor.memory import usable_offer
from nba_sim.market.events import age_minutes, utc_time


def build_win_prediction(summary: dict, metadata: dict, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    home, away = metadata["home_team"], metadata["away_team"]
    probabilities = {team: float(summary["teams"][team]["win_rate"]) for team in (home, away)}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Win probabilities must be finite values between zero and one")
    is_demo = metadata.get("is_demo", False)
    source = "Demo simulation only" if is_demo else "Simulation estimate (not yet calibrated)"
    bookmaker = None
    edge = summary.get("betting_analysis") or {}
    fit = (summary.get("advisor") or {}).get("models", {}).get("h2h", {})
    age = age_minutes(edge.get("created_at"), now)
    start = utc_time(edge.get("commence_time"))
    warnings = bool(edge.get("readiness_reasons") or metadata.get("availability", {}).get("reasons"))
    if (not is_demo and fit.get("status") == "passed" and not warnings
            and age is not None and age <= 60 and start and 0 < (start - now).total_seconds() <= 86400
            and math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4)):
        # Use only an outright-winner market, never a prop/spread selection's
        # probability. Stable book order avoids cherry-picking the highest %.
        offers = sorted((r for r in edge.get("offers", []) if r.get("market") == "h2h"
                         and r.get("team") == home and r.get("push_probability") == 0
                         and usable_offer(r, now, min(60, edge.get("max_odds_age_minutes", 60)))),
                        key=lambda r: r["bookmaker"])
        weight = fit.get("simulation_weight")
        if offers and isinstance(weight, (int, float)) and math.isfinite(weight) and 0 <= weight <= 1:
            quote = offers[0]
            p = weight * quote["conditional_win_probability"] + (1 - weight) * quote["market_no_vig_probability"]
            probabilities = {home: p, away: 1 - p}
            source = "Learned estimate (passed preliminary validation)"
            bookmaker = quote["bookmaker"]
    winner = None if math.isclose(probabilities[home], probabilities[away], abs_tol=1e-12) else max(probabilities, key=probabilities.get)
    return {"winner": winner, "probabilities": probabilities,
            "team_names": {team: summary["teams"][team]["name"] for team in (home, away)},
            "source": source, "bookmaker": bookmaker, "is_demo": is_demo, "input_warnings": warnings,
            "meaning": "Chance of winning this matchup, not of the exact score being correct. Estimates are not guarantees."}


def probability_colour(probability: float) -> str:
    return "green" if probability > .8 else "red" if probability < .5 else "yellow"
