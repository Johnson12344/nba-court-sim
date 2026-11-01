from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_sim.market.events import utc_time
from nba_sim.market.odds_api import NBA_TEAM_NAMES, NBA_SPORT_KEY, _request


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid final score/player total") from exc
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError("Final scores and player totals must be nonnegative integers")
    return int(number)


def save_results(processed_root: Path, new: list[dict]) -> dict:
    processed_root.mkdir(parents=True, exist_ok=True)
    path = processed_root / "results_nba.json"
    payload = _read(path) if path.exists() else {"events": {}}
    observed_at = datetime.now(timezone.utc).isoformat()
    for event in new:
        previous = payload["events"].get(event["event_id"], {})
        score_changed = any(previous.get(key) != event.get(key) for key in ("home_score", "away_score", "completed"))
        event = dict(event)
        event["observed_at"] = observed_at if score_changed else previous.get("observed_at", observed_at)
        if "players" in event:
            event["players_observed_at"] = (previous.get("players_observed_at", observed_at)
                                             if event["players"] == previous.get("players") else observed_at)
        if "players" not in event and any(previous.get(key) != event.get(key) for key in ("home_score", "away_score")):
            previous.pop("players", None)
            previous.pop("players_observed_at", None)
        # A score-only update must not discard verified player results.
        payload["events"][event["event_id"]] = {**previous, **event}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return payload


def refresh_results(raw_root: Path, processed_root: Path, api_key: str) -> tuple[int, dict]:
    events, quota = _request(f"sports/{NBA_SPORT_KEY}/scores", api_key,
                             {"daysFrom": "3", "dateFormat": "iso"}, 30)
    if not isinstance(events, list):
        raise ValueError("Unexpected scores response")
    names = {name: abbr for abbr, name in NBA_TEAM_NAMES.items()}
    results = []
    for event in events:
        if event.get("completed") is not True:
            continue
        scores = {item["name"]: _score(item["score"]) for item in event.get("scores") or []}
        home, away = event.get("home_team"), event.get("away_team")
        if home not in names or away not in names or home not in scores or away not in scores:
            continue
        if scores[home] == scores[away]:
            continue
        results.append({"event_id": str(event["id"]), "home_team": names[home], "away_team": names[away],
                        "home_score": scores[home], "away_score": scores[away],
                        "commence_time": event["commence_time"], "source": "The Odds API completed scores",
                        "completed": True})
    raw_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (raw_root / f"scores_{stamp}.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    save_results(processed_root, results)
    return len(results), quota


def normalize_boxscore(payload: dict, metadata: dict) -> dict:
    game = payload.get("game", {})
    if game.get("gameStatus") != 3:
        raise ValueError("NBA game is not final; results have not been recorded")
    market = metadata.get("market_data") or {}
    scheduled = utc_time(market.get("commence_time"))
    actual = utc_time(game.get("gameTimeUTC"))
    if not market.get("event_id") or scheduled is None or actual is None or abs((actual - scheduled).total_seconds()) > 6 * 3600:
        raise ValueError("NBA game time does not match the saved fixture (maximum six-hour difference)")
    players = {}
    scores = {}
    for label in ("home", "away"):
        team = game.get(f"{label}Team", {})
        abbreviation = team.get("teamTricode")
        if abbreviation != metadata[f"{label}_team"]:
            raise ValueError("NBA game home/away teams do not match the saved prediction")
        scores[f"{label}_score"] = _score(team["score"])
        for player in team.get("players", []):
            player_id = str(player["personId"])
            key = f"{abbreviation}:{player_id}"
            if key in players:
                raise ValueError("Duplicate player in NBA box score")
            stats = player.get("statistics", {})
            if str(player.get("played")) not in {"0", "1"}:
                raise ValueError("Missing player participation status in NBA box score")
            if str(player["played"]) == "1" and not {"points", "reboundsTotal", "assists"} <= set(stats):
                raise ValueError("Incomplete statistics for a player who participated")
            players[key] = {"played": str(player["played"]) == "1", "points": _score(stats.get("points", 0)),
                            "rebounds": _score(stats.get("reboundsTotal", 0)), "assists": _score(stats.get("assists", 0))}
        if sum(p["points"] for key, p in players.items() if key.startswith(f"{abbreviation}:")) != scores[f"{label}_score"]:
            raise ValueError("Player points do not reconcile with the final team score")
    if scores["home_score"] == scores["away_score"]:
        raise ValueError("A final NBA result cannot be tied")
    return {"event_id": str(market["event_id"]), "nba_game_id": str(game["gameId"]),
            "home_team": metadata["home_team"], "away_team": metadata["away_team"],
            "commence_time": market["commence_time"], **scores, "players": players,
            "completed": True, "source": "NBA.com final box score"}


def record_boxscore(run: Path, nba_game_id: str, raw_root: Path, processed_root: Path) -> dict:
    if not re.fullmatch(r"\d{10}", nba_game_id):
        raise ValueError("NBA game ID must contain exactly ten digits (from the NBA.com game link)")
    from nba_api.live.nba.endpoints.boxscore import BoxScore
    metadata = _read(run / "run_metadata.json")
    try:
        payload = BoxScore(game_id=nba_game_id, timeout=30).get_dict()
    except Exception as exc:
        raise RuntimeError("Could not download the NBA final box score; check the game ID and connection") from exc
    result = normalize_boxscore(payload, metadata)
    if result["nba_game_id"] != nba_game_id:
        raise ValueError("NBA response game ID does not match the requested game")
    raw_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (raw_root / f"final_box_{nba_game_id}_{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_results(processed_root, [result])
    return result


def settle_offer(offer: dict, result: dict) -> tuple[str, float | None]:
    key = offer["market"]
    if key in {"h2h", "spreads"}:
        value = result["home_score"] - result["away_score"]
        if offer["team"] == result["away_team"]:
            value = -value
        elif offer["team"] != result["home_team"]:
            return "unmatched", None
        if key == "spreads":
            value += offer["line"]
    elif key == "totals":
        value = result["home_score"] + result["away_score"] - offer["line"]
        if offer["selection"] == "Under":
            value = -value
    else:
        stat = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists"}.get(key)
        player = result.get("players", {}).get(f"{offer['team']}:{offer['player_id']}")
        if stat is None or player is None:
            return "pending_player_result", None
        if not player["played"]:
            return "void_dnp", 0.0
        value = player[stat] - offer["line"]
        if offer["selection"] == "Under":
            value = -value
    if value == 0:
        return "push", 0.0
    return ("win", offer["decimal_odds"] - 1) if value > 0 else ("loss", -1.0)


def evaluate_forward(output_root: Path, results_path: Path) -> tuple[dict, Path]:
    if not results_path.exists():
        raise ValueError("No results cached. Run refresh-results or record-result first.")
    results = _read(results_path).get("events", {})
    predictions = {}
    skipped = []
    for path in sorted(output_root.glob("*/run_metadata.json")):
        metadata = _read(path)
        edge_path = path.parent / "betting_analysis.json"
        if metadata.get("is_demo") or not edge_path.exists():
            continue
        market = metadata.get("market_data") or {}
        event_id = str(market.get("event_id", ""))
        created, kickoff = utc_time(metadata.get("created_at")), utc_time(market.get("commence_time"))
        edge = _read(edge_path)
        edge_created = utc_time(edge.get("created_at"))
        if created is None or kickoff is None or edge_created is None or max(created, edge_created) >= kickoff:
            skipped.append(f"{path.parent.name}: not a saved pregame prediction")
            continue
        # Every model version gets its own evaluation. Repeated seeds/runs cannot pad sample size.
        identity = (event_id, str(metadata.get("model_version")), str(metadata.get("model")))
        if identity not in predictions or created < predictions[identity][0]:
            predictions[identity] = (created, path.parent, metadata, edge)
    game_rows, offer_rows, player_rows = [], [], []
    for (event_id, version, model), (_, run, metadata, edge) in predictions.items():
        actual = results.get(event_id)
        if not actual or actual.get("completed") is not True:
            continue
        if any(actual.get(f"{side}_team") != metadata[f"{side}_team"] for side in ("home", "away")):
            skipped.append(f"{run.name}: result teams do not match")
            continue
        if utc_time(actual.get("commence_time")) != utc_time(metadata["market_data"].get("commence_time")):
            skipped.append(f"{run.name}: result start time changed; review postponed fixture manually")
            continue
        summary = _read(run / "summary.json")
        home = summary["teams"][metadata["home_team"]]
        away = summary["teams"][metadata["away_team"]]
        won = int(actual["home_score"] > actual["away_score"])
        probability = home["win_rate"]
        game_rows.append({"event_id": event_id, "run": run.name, "model_version": version, "model": model,
                          "home_win_probability": probability, "home_won": won,
                          "brier_score": (probability - won) ** 2,
                          "team_score_mae": (abs(home["score"]["mean"] - actual["home_score"]) + abs(away["score"]["mean"] - actual["away_score"])) / 2,
                          "p10_p90_coverage": (int(home["score"]["p10"] <= actual["home_score"] <= home["score"]["p90"]) + int(away["score"]["p10"] <= actual["away_score"] <= away["score"]["p90"])) / 2})
        chosen = set()
        for offer in edge["offers"]:
            status, profit = settle_offer(offer, actual)
            # A fixed paper strategy: highest-EV eligible offer per market/player, from the first pregame run.
            group = (offer["market"], offer.get("player_id"))
            selected = bool(offer["review_candidate"] and group not in chosen)
            if selected:
                chosen.add(group)
            offer_rows.append({**offer, "run": run.name, "model_version": version, "model": model,
                               "paper_selected": selected, "result": status, "profit_per_unit": profit})
        if actual.get("players"):
            try:
                frame = pd.read_csv(run / "player_distributions.csv", dtype={"player_id": str})
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            for row in frame.to_dict("records"):
                observed = actual["players"].get(f"{row['team']}:{row['player_id']}")
                if observed and observed["played"] and row["stat"] in {"points", "rebounds", "assists"}:
                    value = observed[row["stat"]]
                    player_rows.append({"event_id": event_id, "model_version": version, "model": model,
                                        "player_id": row["player_id"], "stat": row["stat"], "predicted": row["mean"],
                                        "actual": value, "absolute_error": abs(row["mean"] - value)})
    groups = []
    for version, model in sorted({(r["model_version"], r["model"]) for r in game_rows}):
        games = [r for r in game_rows if (r["model_version"], r["model"]) == (version, model)]
        bets = [r for r in offer_rows if (r["model_version"], r["model"]) == (version, model) and r["paper_selected"]]
        settled = [r for r in bets if r["result"] in {"win", "loss", "push"}]
        profit = sum(r["profit_per_unit"] for r in settled)
        calibration = []
        for low in range(0, 100, 10):
            bucket = [r for r in games if low / 100 <= r["home_win_probability"] < (low + 10) / 100 or (low == 90 and r["home_win_probability"] == 1)]
            if bucket:
                calibration.append({"probability_range": f"{low}-{low + 10}%", "games": len(bucket),
                                    "mean_prediction": sum(r["home_win_probability"] for r in bucket) / len(bucket),
                                    "actual_win_rate": sum(r["home_won"] for r in bucket) / len(bucket)})
        groups.append({"model_version": version, "model": model, "unique_games": len(games),
                       "brier_score": sum(r["brier_score"] for r in games) / len(games),
                       "team_score_mae": sum(r["team_score_mae"] for r in games) / len(games),
                       "p10_p90_coverage": sum(r["p10_p90_coverage"] for r in games) / len(games),
                       "paper_settled_selections": len(settled), "paper_profit_units": profit,
                       "paper_roi": profit / len(settled) if settled else None,
                       "paper_pending": sum(r["profit_per_unit"] is None for r in bets),
                       "paper_void_dnp": sum(r["result"] == "void_dnp" for r in bets),
                       "player_mae": {stat: sum(r["absolute_error"] for r in player_rows if (r["model_version"], r["model"], r["stat"]) == (version, model, stat)) / max(1, sum((r["model_version"], r["model"], r["stat"]) == (version, model, stat) for r in player_rows))
                                      for stat in {r["stat"] for r in player_rows if (r["model_version"], r["model"]) == (version, model)}},
                       "calibration": calibration})
    report = {"groups": groups, "skipped": skipped,
              "selection_rule": "First saved pregame run per event/model/version; highest-EV review candidate per market/player; one hypothetical unit each. All offers are exported, including losses.",
              "limitations": ["Paper returns assume quoted prices were available; no actual bets are placed or tracked.",
                              "Selections within a game are correlated. Positive paper ROI does not establish a reliable advantage.",
                              "Player DNPs are void; other bookmaker-specific void rules are not modeled.",
                              "Local files are an audit trail, not tamper-proof records."]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = output_root / f"evaluation_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    (target / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(game_rows).to_csv(target / "games.csv", index=False)
    pd.DataFrame(offer_rows).to_csv(target / "offers.csv", index=False)
    pd.DataFrame(player_rows).to_csv(target / "players.csv", index=False)
    return report, target
