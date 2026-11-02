"""Transparent outcome review, including forecasts excluded from training."""
from collections import Counter

from nba_sim.evaluation.forward import settle_offer


def review_history(predictions: list[dict], now) -> dict:
    counts, markets, recent, abstentions = Counter(), {}, [], []
    for prediction in sorted(predictions, key=lambda p: p["created_at"], reverse=True):
        counts.update(set(prediction.get("exclusion_reasons", [])))
        actual = prediction.get("result")
        learned_markets = {r["market"] for r in prediction["offers"]}
        for offer in prediction.get("audit_offers", prediction["offers"]):
            market = offer["market"]
            item = markets.setdefault(market, {"saved": 0, "settled": 0, "wins": 0, "losses": 0,
                                               "pushes": 0, "voids": 0, "excluded": 0,
                                               "simulation_brier_sum": 0., "market_brier_sum": 0.})
            item["saved"] += 1
            excluded = bool(offer.get("blockers") or prediction.get("exclusion_reasons")) or market not in learned_markets
            item["excluded"] += int(excluded)
            status, profit = settle_offer(offer, actual) if actual else ("pending", None)
            if status in {"win", "loss"}:
                item["settled"] += 1
                item["wins" if status == "win" else "losses"] += 1
                y = int(status == "win")
                item["simulation_brier_sum"] += (offer["conditional_win_probability"] - y) ** 2
                item["market_brier_sum"] += (offer["market_no_vig_probability"] - y) ** 2
            item["pushes"] += int(status == "push")
            item["voids"] += int(status == "void_dnp")
        # Preserve exactly the choices made before tip-off, including losses.
        decision = prediction.get("decision", {})
        choices = ([decision["selection"]] if decision.get("selection") else
                   prediction.get("paper_portfolio") or
                   ([decision["paper_watch"]] if decision.get("paper_watch") else []))
        if decision.get("status") == "no_bet":
            abstentions.append({"fixture": f"{prediction['away_team']} at {prediction['home_team']}",
                                "created_at": prediction["created_at"], "reasons": decision.get("reasons", []),
                                "final_score": f"{actual['away_score']}-{actual['home_score']}" if actual else None})
        for choice in choices:
            status, profit = settle_offer(choice, actual) if actual else ("pending", None)
            recent.append({"fixture": f"{prediction['away_team']} at {prediction['home_team']}",
                           "created_at": choice.get("prediction_created_at", prediction["created_at"]), "market": choice["market"],
                           "selection": choice["selection"], "player": choice.get("player"),
                           "team": choice.get("team"), "line": choice.get("line"),
                           "bookmaker": choice["bookmaker"], "decimal_odds": choice["decimal_odds"],
                           "probability": choice.get("estimated_win_probability", choice.get("win_probability")),
                           "result": status, "profit_units": profit,
                           "kind": "review candidate" if decision.get("selection") else "paper only"})
    for item in markets.values():
        n = item["settled"]
        item["simulation_brier"] = item.pop("simulation_brier_sum") / n if n else None
        item["market_brier"] = item.pop("market_brier_sum") / n if n else None
    recent.sort(key=lambda r: r["created_at"], reverse=True)
    return {"market_history": markets, "recent_decisions": recent[:20], "recent_no_bets": abstentions[:10],
            "training_exclusions": [{"reason": k, "fixtures": v} for k, v in counts.most_common(5)],
            "fixtures_without_training": sum(not p["offers"] for p in predictions),
            "review_note": "Audit scores include excluded forecasts; they do not certify a model or count as training. Paper returns are hypothetical."}
