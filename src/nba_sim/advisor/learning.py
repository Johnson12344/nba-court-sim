"""A constrained, outcome-trained blend of simulation and no-vig market probabilities.

The learned weight minimizes Brier loss. Validation uses later events whose
outcomes were unavailable during training. No LLM or external AI service is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np

from nba_sim.advisor.memory import MARKETS, fingerprint, memory_summary, training_rows, usable_offer
from nba_sim.market.events import age_minutes, utc_time

ADVISOR_VERSION = "1.0.0"
MIN_TRAIN = 100
MIN_VALIDATION = 40
MIN_WIN_PROBABILITY = 0.55


def fit_weight(rows: list[dict]) -> float:
    p, q, y = (np.array([r[key] for r in rows], dtype=float) for key in ("p", "q", "y"))
    difference = p - q
    denominator = float(difference @ difference)
    return float(np.clip((difference @ (y - q)) / denominator, 0, 1)) if denominator > 1e-12 else 0.0


def fit_market(rows: list[dict]) -> dict:
    # A defensive second deduplication also protects imported memory.
    events = {}
    for row in sorted(rows, key=lambda r: (r["created_at"], r["event_id"])):
        events.setdefault(row["event_id"], row)
    rows = sorted(events.values(), key=lambda r: (utc_time(r["created_at"]), r["event_id"]))
    report = {"status": "learning", "games": len(rows), "minimum_training_games": MIN_TRAIN,
              "minimum_validation_games": MIN_VALIDATION, "training_fingerprint": fingerprint(rows)}
    if len(rows) < MIN_TRAIN + MIN_VALIDATION:
        report["reason"] = f"Needs at least {MIN_TRAIN} training games plus {MIN_VALIDATION} later validation games."
        return report
    test = rows[-MIN_VALIDATION:]
    cutoff = utc_time(test[0]["created_at"])
    train = [r for r in rows[:-MIN_VALIDATION] if utc_time(r["observed_at"]) < cutoff
             and utc_time(r["created_at"]) < cutoff]
    report.update(training_games=len(train), validation_games=len(test), validation_starts_at=cutoff.isoformat())
    if len(train) < MIN_TRAIN:
        report["reason"] = "More completed results are needed before the validation period; overlapping games cannot train on future outcomes."
        return report
    weight = fit_weight(train)
    p, q, y = (np.array([r[key] for r in test], dtype=float) for key in ("p", "q", "y"))
    blended = weight * p + (1 - weight) * q
    scores = {name: float(np.mean((values - y) ** 2)) for name, values in
              (("learned", blended), ("simulation", p), ("market", q))}
    # This is a conservative engineering screen, not a proof of profitability.
    passed = scores["learned"] <= scores["simulation"] + 1e-12 and scores["learned"] <= scores["market"] - 0.002
    report.update(status="passed" if passed else "validation_failed", simulation_weight=weight,
                  validation_brier=scores,
                  reason="Passed the preliminary probability test; profitability is unproven." if passed else
                         "The learned probabilities have not improved on the validation benchmarks.")
    return report


def advise(edge: dict, memory: dict, version: str, model: str, *, now: datetime | None = None,
           excluded_event: str | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    excluded_event = excluded_event or edge.get("event_id")
    rows = [r for r in training_rows(memory, version, model, now) if r["event_id"] != excluded_event]
    models = {market: fit_market([r for r in rows if r["market"] == market]) for market in MARKETS}
    result = {"advisor_version": ADVISOR_VERSION, "created_at": now.isoformat(), "status": "no_bet",
              "selection": None, "paper_watch": None, "shortlist": [], "models": models,
              "memory": memory_summary(memory, version, model, now), "model_version": version,
              "reasons": [], "selection_rule": "Highest estimated win chance among eligible positive-EV quotes; then higher EV.",
              "limitations": "Probabilities and minutes remain estimates. No bet is safe or guaranteed. "
                             "Validation measures prediction error, not proven profit. No bets are placed. "
                             "Push probability remains the simulator estimate; player settlement assumes DNP voids."}
    start = utc_time(edge.get("commence_time"))
    created_age = age_minutes(edge.get("created_at"), now)
    if (not start or not 0 < (start - now).total_seconds() <= 86400 or created_age is None or created_age > 60):
        result["reasons"].append("No fresh pregame analysis within 24 hours of tip-off.")
        return result
    if edge.get("readiness_reasons"):
        result["reasons"] = list(edge["readiness_reasons"])
        return result
    candidates, watch = [], []
    for offer in edge.get("offers", []):
        if (not usable_offer(offer, now, min(60, edge.get("max_odds_age_minutes", 60)))
                or not offer.get("review_candidate") or offer.get("mc_lower_profit_per_unit", -1) <= 0):
            continue
        p, q = offer["conditional_win_probability"], offer["market_no_vig_probability"]
        push, price = offer["push_probability"], offer["decimal_odds"]
        minimum_ev = max(0.02, edge.get("min_ev", .02))
        raw_ev = (1 - push) * (p * price - 1)
        if raw_ev < minimum_ev:
            continue
        model_fit = models[offer["market"]]
        raw = {**offer, "estimated_win_probability": (1 - push) * p,
               "estimated_loss_probability": (1 - push) * (1 - p), "adjusted_ev": raw_ev,
               "probability_source": "Unvalidated simulation only", "learning_status": model_fit["status"]}
        if raw["estimated_win_probability"] >= MIN_WIN_PROBABILITY:
            watch.append(raw)
        if model_fit["status"] != "passed":
            continue
        weight = model_fit["simulation_weight"]
        adjusted = weight * p + (1 - weight) * q
        ev = (1 - push) * (adjusted * price - 1)
        win = (1 - push) * adjusted
        if win < MIN_WIN_PROBABILITY or ev < minimum_ev or not math.isfinite(ev):
            continue
        candidates.append({**offer, "estimated_win_probability": win,
                           "estimated_loss_probability": (1 - push) * (1 - adjusted), "adjusted_ev": ev,
                           "probability_source": "Outcome-trained simulation/market blend",
                           "simulation_weight": weight, "learning_status": "passed",
                           "training_games": model_fit["training_games"], "validation_games": model_fit["validation_games"]})
    order = lambda r: (-r["estimated_win_probability"], -r["adjusted_ev"], r["bookmaker"], r["market"])
    # Multiple books quoting the same selection should not occupy the shortlist.
    distinct = {}
    for row in sorted(candidates, key=order):
        identity = (row["market"], row.get("player_id"), row.get("team"), row["selection"], row.get("line"))
        distinct.setdefault(identity, row)
    result["shortlist"] = list(distinct.values())[:3]
    if result["shortlist"]:
        result.update(status="review_candidate", selection=result["shortlist"][0])
        result["reasons"] = ["Passed current data checks and preliminary chronological validation.",
                             "Ranks win probability only after checking positive modeled value; it is not a guaranteed winner."]
    else:
        result["reasons"] = ["No selection passed both the learning checks and the current price/risk checks."]
        if watch:
            result["paper_watch"] = sorted(watch, key=order)[0]
            result["reasons"].append("A paper-only simulation candidate is shown while the advisor collects and validates outcomes.")
    return result
