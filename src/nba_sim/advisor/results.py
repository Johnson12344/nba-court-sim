"""Resolve saved fixtures to cached NBA game IDs and fetch verified final boxes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pandas as pd

from nba_sim.advisor.memory import save_memory
from nba_sim.evaluation.forward import normalize_boxscore, save_results
from nba_sim.market.events import age_minutes, utc_time


def find_game_id(prediction: dict, processed: Path) -> str | None:
    season = prediction["season"]
    if not re.fullmatch(r"\d{4}-\d{2}", season):
        return None
    path = processed / f"game_log_{season}.csv"
    if not path.exists():
        return None
    log = pd.read_csv(path, dtype={"game_id": str})
    if not {"game_id", "game_date", "team_abbreviation", "matchup"} <= set(log):
        return None
    date = utc_time(prediction["commence_time"]).astimezone(ZoneInfo("America/New_York")).date()
    dates = pd.to_datetime(log.game_date, errors="coerce").dt.date
    matchup = f"{prediction['home_team']} vs. {prediction['away_team']}"
    ids = set(log.loc[(dates == date) & (log.team_abbreviation == prediction["home_team"])
                     & (log.matchup == matchup), "game_id"].astype(str))
    if len(ids) == 1:
        game_id = ids.pop()
        return game_id if re.fullmatch(r"\d{10}", game_id) else None
    return None


def refresh_memory_results(memory: dict, raw: Path, processed: Path, *, force: bool = False,
                           now: datetime | None = None, progress=None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    age = age_minutes(memory.get("results_checked_at"), now)
    if not force and age is not None and age < 360:
        return memory.get("result_warnings", [])
    pending = {}
    for prediction in memory["predictions"].values():
        start = utc_time(prediction["commence_time"])
        actual = prediction.get("result", {})
        if start and start < now and not actual.get("players"):
            pending.setdefault(prediction["event_id"], prediction)
    warnings = []
    last_attempts = memory.setdefault("result_attempts", {})
    # Bound online work; oldest pending fixtures first. An unresolvable fixture
    # does not consume a download slot or starve later resolvable games.
    attempts = 0
    for prediction in sorted(pending.values(), key=lambda p: (last_attempts.get(p["event_id"], ""), p["commence_time"])):
        try:
            game_id = find_game_id(prediction, processed)
            if not game_id:
                warnings.append(f"{prediction['event_id']}: awaiting a matching completed game in the NBA game log.")
                continue
            if attempts >= 5:
                warnings.append("Remaining results will be checked on a later analyze run.")
                break
            attempts += 1
            last_attempts[prediction["event_id"]] = now.isoformat()
            if progress:
                progress(f"Memory: checking final NBA result {game_id}")
            from nba_api.live.nba.endpoints.boxscore import BoxScore
            payload = BoxScore(game_id=game_id, timeout=15).get_dict()
            metadata = {**prediction, "market_data": {"event_id": prediction["event_id"],
                                                       "commence_time": prediction["commence_time"]}}
            result = normalize_boxscore(payload, metadata)
            if result["nba_game_id"] != game_id:
                raise ValueError("Final box score has a different NBA game ID")
            raw.mkdir(parents=True, exist_ok=True)
            (raw / f"advisor_final_{game_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            save_results(processed, [result])
        except Exception as exc:
            warnings.append(f"{prediction['event_id']}: final result could not be verified ({type(exc).__name__}); will retry.")
    if pending:
        memory["result_warnings"] = warnings
        if attempts:
            memory["results_checked_at"] = now.isoformat()
        else:
            # No network work occurred. A later analyze may refresh the game
            # log immediately, so do not throttle newly discoverable results.
            memory.pop("results_checked_at", None)
        save_memory(processed, memory)
    return warnings
