from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_sim.evaluation.metrics import interval_coverage, mean_absolute_error

REQUIRED = {"game_id", "game_date", "home_team", "away_team", "home_score", "away_score"}


def walk_forward_backtest(path: Path, season: str, game_limit: int, output_root: Path) -> tuple[dict[str, Any], Path]:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical game file missing: {path}. Provide observed games with columns: {', '.join(sorted(REQUIRED | {'season'}))}"
        )
    games = pd.read_csv(path)
    missing = REQUIRED - set(games.columns)
    if missing:
        raise ValueError("Historical games missing columns: " + ", ".join(sorted(missing)))
    if "season" in games:
        games = games.loc[games.season.astype(str) == season].copy()
    games["game_date"] = pd.to_datetime(games.game_date, errors="raise", utc=True)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, game in games.iterrows():
        prior = games.loc[games.game_date < game.game_date]
        # Strictly earlier dates prevent same-day and future leakage.
        home_history = prior.loc[(prior.home_team == game.home_team) | (prior.away_team == game.home_team)]
        away_history = prior.loc[(prior.home_team == game.away_team) | (prior.away_team == game.away_team)]
        if len(home_history) < 5 or len(away_history) < 5:
            continue
        league_scores = pd.concat([prior.home_score, prior.away_score]).astype(float)
        league_mean = float(league_scores.mean())
        league_sd = max(8.0, float(league_scores.std(ddof=0)))

        def team_for_against(history: pd.DataFrame, team: str) -> tuple[float, float]:
            scored = np.where(history.home_team == team, history.home_score, history.away_score).astype(float)
            allowed = np.where(history.home_team == team, history.away_score, history.home_score).astype(float)
            return float(np.mean(scored)), float(np.mean(allowed))

        home_for, home_against = team_for_against(home_history, str(game.home_team))
        away_for, away_against = team_for_against(away_history, str(game.away_team))
        possession_home = (home_for + away_against) / 2 + 1.5
        possession_away = (away_for + home_against) / 2
        rows.append({
            "game_id": game.game_id, "game_date": game.game_date.isoformat(),
            "training_max_date": prior.game_date.max().isoformat(), "training_games": len(prior),
            "home_team": game.home_team, "away_team": game.away_team,
            "actual_home": float(game.home_score), "actual_away": float(game.away_score),
            "baseline_home": league_mean + 1.5, "baseline_away": league_mean,
            "possession_home": possession_home, "possession_away": possession_away,
            "p10_home": possession_home - 1.2816 * league_sd, "p90_home": possession_home + 1.2816 * league_sd,
            "p10_away": possession_away - 1.2816 * league_sd, "p90_away": possession_away + 1.2816 * league_sd,
        })
        if len(rows) >= game_limit:
            break
    if not rows:
        raise ValueError("No eligible holdout games: each team needs at least five strictly earlier games")
    results = pd.DataFrame(rows)
    actual = results[["actual_home", "actual_away"]].to_numpy().ravel().tolist()
    baseline = results[["baseline_home", "baseline_away"]].to_numpy().ravel().tolist()
    possession = results[["possession_home", "possession_away"]].to_numpy().ravel().tolist()
    low = results[["p10_home", "p10_away"]].to_numpy().ravel().tolist()
    high = results[["p90_home", "p90_away"]].to_numpy().ravel().tolist()
    report = {
        "season": season, "evaluated_games": len(results),
        "method": "walk-forward team scoring evaluation; training rows are strictly earlier than each holdout date",
        "baseline_team_score_mae": round(mean_absolute_error(actual, baseline), 3),
        "possession_feature_team_score_mae": round(mean_absolute_error(actual, possession), 3),
        "possession_p10_p90_coverage": round(interval_coverage(actual, low, high), 4),
        "limitations": [
            "This initial evaluator compares score-level estimators; player-stat evaluation requires player box-score history.",
            "The possession-feature estimate uses pregame for/against histories, not the holdout game's final features.",
        ],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = output_root / f"backtest_{season}_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    results.to_csv(target / "backtest_games.csv", index=False)
    (target / "backtest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, target

