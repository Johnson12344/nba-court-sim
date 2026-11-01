from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from nba_sim.evaluation.forward import settle_offer
from nba_sim.market.events import age_minutes, utc_time

MARKETS = ("h2h", "spreads", "totals", "player_points", "player_rebounds", "player_assists")
MEMORY_FILE = "advisor_memory.json"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def load_memory(processed: Path) -> dict:
    path = processed / MEMORY_FILE
    if not path.exists():
        return {"schema_version": 1, "predictions": {}}
    memory = read_json(path)
    if memory.get("schema_version") != 1 or not isinstance(memory.get("predictions"), dict):
        raise ValueError("Advisor memory format is invalid; existing memory has not been overwritten.")
    for prediction in memory["predictions"].values():
        fields = ("event_id", "model_version", "model", "season", "home_team", "away_team", "created_at", "commence_time")
        if (not isinstance(prediction, dict) or not all(isinstance(prediction.get(k), str) for k in fields)
                or utc_time(prediction["created_at"]) is None or utc_time(prediction["commence_time"]) is None
                or not isinstance(prediction.get("offers"), list)
                or not all(isinstance(row, dict) for row in prediction["offers"])
                or not isinstance(prediction.get("decision", {}), dict)
                or not isinstance(prediction.get("result", {}), dict)):
            raise ValueError("An advisor memory entry is invalid; existing memory has not been overwritten.")
    return memory


def save_memory(processed: Path, memory: dict) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=processed, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(memory, handle, indent=2, allow_nan=False)
    try:
        os.replace(temporary, processed / MEMORY_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def usable_offer(row: dict, at: datetime, max_age: float = 60) -> bool:
    """Quality gate shared by training and inference; low EV is NOT excluded from training."""
    try:
        p, q = float(row["conditional_win_probability"]), float(row["market_no_vig_probability"])
        win, loss, push = (float(row[k]) for k in ("win_probability", "loss_probability", "push_probability"))
        price = float(row["decimal_odds"])
        age = age_minutes(row.get("quote_updated_at"), at)
        line_ok = row["market"] == "h2h" or math.isfinite(float(row["line"]))
        return (row["market"] in MARKETS and not row.get("blockers", ["missing quality checks"])
                and bool(row.get("bookmaker")) and int(row["simulations"]) >= 500 and line_ok
                and all(math.isfinite(x) for x in (p, q, win, loss, push, price))
                and 0 <= p <= 1 and 0 < q < 1 and min(win, loss, push) >= 0 and push < 1
                and math.isclose(win + loss + push, 1, abs_tol=1e-6)
                and math.isclose(p, win / (1 - push), abs_tol=1e-6)
                and price > 1 and age is not None and age <= max_age)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def representative_offers(offers: list[dict], metadata: dict, at: datetime, max_age: float) -> list[dict]:
    """One observation per event/market; repeated books, sides and players cannot pad N.

    Use home/Over orientation, most widely quoted player/line, then stable tie
    breaks. This is fixed before consulting outcomes, irrespective of EV.
    """
    choices = []
    for market in MARKETS:
        rows = [r for r in offers if r.get("market") == market and usable_offer(r, at, max_age)
                and (r.get("team") == metadata["home_team"] if market in {"h2h", "spreads"}
                     else r.get("selection") == "Over")]
        counts = Counter((r.get("player_id") or "", r.get("line")) for r in
                         { (r["bookmaker"], r.get("player_id"), r.get("line")): r for r in rows }.values())
        if rows:
            rows.sort(key=lambda r: (-counts[(r.get("player_id") or "", r.get("line"))],
                                     str(r.get("player_id") or ""), float(r.get("line") or 0), r["bookmaker"]))
            choices.append(dict(rows[0]))
    return choices


def sync_memory(output_root: Path, processed: Path, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    memory = load_memory(processed)
    warnings = []
    for path in sorted(output_root.glob("*/run_metadata.json")):
        edge_path = path.parent / "betting_analysis.json"
        if not edge_path.exists():
            continue
        try:
            meta, edge = read_json(path), read_json(edge_path)
            if not isinstance(edge.get("offers"), list) or not all(isinstance(r, dict) for r in edge["offers"]):
                raise ValueError("Invalid stored market quotes")
            if meta.get("is_demo") is not False or meta.get("model") != "possession" or meta.get("advisor_enabled") is False:
                continue
            market = meta.get("market_data") or {}
            event = str(market.get("event_id") or "")
            created, start, edge_time = (utc_time(x) for x in
                                         (meta.get("created_at"), market.get("commence_time"), edge.get("created_at")))
            if (not event or event != edge.get("event_id") or not all((created, start, edge_time))
                    or max(created, edge_time) >= start or max(created, edge_time) > now
                    or utc_time(edge.get("commence_time")) != start):
                continue
            key = json.dumps([event, str(meta["model_version"]), meta["model"]])
            previous = memory["predictions"].get(key)
            if previous and utc_time(previous["created_at"]) <= created:
                continue
            selection_path = path.parent / "advisor.json"
            decision = read_json(selection_path) if selection_path.exists() else {}
            choices = [] if edge.get("readiness_reasons") else representative_offers(
                edge.get("offers", []), meta, edge_time, min(60, float(edge.get("max_odds_age_minutes", 60))))
            memory["predictions"][key] = {
                "event_id": event, "model_version": str(meta["model_version"]), "model": meta["model"],
                "season": meta["season"], "home_team": meta["home_team"], "away_team": meta["away_team"],
                "created_at": created.isoformat(), "commence_time": start.isoformat(), "run": str(path.parent),
                "offers": choices, "decision": {"status": decision.get("status"),
                                                   "selection": decision.get("selection"),
                                                   "paper_watch": decision.get("paper_watch"),
                                                   "advisor_version": decision.get("advisor_version")},
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Skipped unreadable prediction {path.parent.name}: {type(exc).__name__}")
    result_path = processed / "results_nba.json"
    results = read_json(result_path) if result_path.exists() else {}
    if not isinstance(results.get("events", {}), dict):
        raise ValueError("Final result cache has an invalid event collection")
    for prediction in memory["predictions"].values():
        result = results.get("events", {}).get(prediction["event_id"])
        if result is not None:
            if not isinstance(result, dict):
                raise ValueError("Final result cache has an invalid event")
            # A corrected/mismatched result invalidates the old settlement.
            prediction.pop("result", None)
            if (result.get("completed") is True
                    and all(result.get(f"{s}_team") == prediction[f"{s}_team"] for s in ("home", "away"))
                    and utc_time(result.get("commence_time")) == utc_time(prediction["commence_time"])):
                prediction["result"] = {**result, "observed_at": result.get("observed_at", results.get("updated_at")),
                                        "players_observed_at": result.get("players_observed_at", results.get("updated_at"))}
    memory.update(updated_at=now.isoformat(), warnings=warnings)
    save_memory(processed, memory)
    return memory


def training_rows(memory: dict, version: str, model: str, now: datetime) -> list[dict]:
    rows = []
    for prediction in memory["predictions"].values():
        if (prediction["model_version"], prediction["model"]) != (version, model):
            continue
        actual = prediction.get("result")
        if not actual:
            continue
        for offer in prediction["offers"]:
            observed = utc_time(actual.get("players_observed_at" if offer["market"].startswith("player_") else "observed_at"))
            if observed is None or observed > now or observed <= utc_time(prediction["commence_time"]):
                continue
            status, _ = settle_offer(offer, actual)
            if status not in {"win", "loss"}:
                continue  # Calibrate conditional win probability: pushes/DNPs are not losses.
            rows.append({"event_id": prediction["event_id"], "market": offer["market"],
                         "created_at": prediction["created_at"], "observed_at": observed.isoformat(),
                         "p": offer["conditional_win_probability"], "q": offer["market_no_vig_probability"],
                         "y": int(status == "win")})
    return rows


def memory_summary(memory: dict, version: str, model: str, now: datetime) -> dict:
    predictions = [p for p in memory["predictions"].values()
                   if (p["model_version"], p["model"]) == (version, model)]
    settlements = []
    watch_settlements = []
    for prediction in predictions:
        decision = prediction.get("decision", {})
        if decision.get("status") == "review_candidate" and decision.get("selection") and prediction.get("result"):
            status, profit = settle_offer(decision["selection"], prediction["result"])
            if status in {"win", "loss", "push"}:
                settlements.append((status, profit))
        if decision.get("paper_watch") and prediction.get("result"):
            status, profit = settle_offer(decision["paper_watch"], prediction["result"])
            if status in {"win", "loss", "push"}:
                watch_settlements.append((status, profit))
    return {"saved_fixtures": len(predictions), "completed_fixtures": sum(bool(p.get("result")) for p in predictions),
            "pending_fixtures": sum(utc_time(p["commence_time"]) < now and not p.get("result") for p in predictions),
            "usable_games_by_market": dict(Counter(r["market"] for r in training_rows(memory, version, model, now))),
            "paper_selections_settled": len(settlements),
            "paper_profit_units": sum(profit for _, profit in settlements),
            "paper_roi": sum(profit for _, profit in settlements) / len(settlements) if settlements else None,
            "watch_selections_settled": len(watch_settlements),
            "watch_profit_units": sum(profit for _, profit in watch_settlements),
            "watch_roi": sum(profit for _, profit in watch_settlements) / len(watch_settlements) if watch_settlements else None}


def fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
