from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from nba_sim.models import SimulatedGame, TeamProfile


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 3),
        "median": round(float(np.median(array)), 3),
        "std_dev": round(float(np.std(array, ddof=0)), 3),
        "p10": round(float(np.percentile(array, 10)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
        "min": round(float(np.min(array)), 3),
        "max": round(float(np.max(array)), 3),
    }


def create_summary(games: list[SimulatedGame], home: TeamProfile, away: TeamProfile) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    score_rows = []
    for game in games:
        row: dict[str, Any] = {
            "simulation": game.game_index, "seed": game.seed,
            "home_team": home.abbreviation, "away_team": away.abbreviation,
            "home_score": game.home_score, "away_score": game.away_score,
            "total_score": game.total, "home_margin": game.margin, "overtimes": game.overtimes,
        }
        for quarter in game.quarter_scores:
            label = f"q{quarter['period']}" if quarter["period"] <= 4 else f"ot{quarter['period'] - 4}"
            row[f"home_{label}"] = quarter["home"]
            row[f"away_{label}"] = quarter["away"]
        score_rows.append(row)
    scores = pd.DataFrame(score_rows)
    team_summary: dict[str, Any] = {}
    for label, team, column in (("home", home, "home_score"), ("away", away, "away_score")):
        quarter_distributions: dict[str, Any] = {}
        for period in range(1, 5):
            values = [game.quarter_scores[period - 1][label] for game in games]
            quarter_distributions[f"q{period}"] = _distribution(values)
        team_summary[team.abbreviation] = {
            "name": team.name, "score": _distribution(scores[column].tolist()),
            "quarter_scores": quarter_distributions,
            "win_rate": round(float(np.mean(scores.home_score > scores.away_score) if label == "home" else np.mean(scores.away_score > scores.home_score)), 4),
        }
    player_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for game in games:
        for team_abbr, boxes in game.player_stats.items():
            for player_id, box in boxes.items():
                for stat in ("minutes", "points", "oreb", "dreb", "assists", "fg3m", "turnovers"):
                    value = getattr(box, stat)
                    if stat == "dreb":
                        continue
                    if stat == "oreb":
                        value = box.oreb + box.dreb
                        stat = "rebounds"
                    player_values[(team_abbr, player_id, stat)].append(float(value))
    profile_map = {player.player_id: player for team in (home, away) for player in team.players}
    player_rows: list[dict[str, Any]] = []
    for (team_abbr, player_id, stat), values in sorted(player_values.items()):
        profile = profile_map[player_id]
        distribution = _distribution(values)
        player_rows.append({
            "team": team_abbr, "player_id": player_id, "player_name": profile.name, "stat": stat,
            **distribution, "expected_minutes": round(profile.expected_minutes, 2),
            "source_games": profile.games, "source_attempts": round(profile.sample_sizes.get("fga", 0), 1),
            "reliability": profile.reliability,
            "starter": profile.starter,
            "shooter_selection_source": profile.shooter_selection_source,
            "shot_location_source": profile.shot_location_source,
        })
    players = pd.DataFrame(player_rows)
    return {"teams": team_summary, "game_total": _distribution(scores.total_score.tolist()),
            "home_margin": _distribution(scores.home_margin.tolist())}, scores, players

