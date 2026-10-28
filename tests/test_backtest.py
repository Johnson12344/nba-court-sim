from __future__ import annotations

import pandas as pd

from nba_sim.evaluation.backtest import walk_forward_backtest


def test_walk_forward_cutoff_prevents_future_data(tmp_path):
    rows = []
    for index in range(16):
        rows.append({
            "game_id": index, "game_date": f"2025-01-{index + 1:02d}", "season": "2025-26",
            "home_team": "AAA" if index % 2 == 0 else "BBB",
            "away_team": "BBB" if index % 2 == 0 else "AAA",
            "home_score": 100 + index, "away_score": 98 + index,
        })
    path = tmp_path / "historical_games.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    report, target = walk_forward_backtest(path, "2025-26", 4, tmp_path / "outputs")
    result = pd.read_csv(target / "backtest_games.csv")
    assert report["evaluated_games"] == 4
    assert (pd.to_datetime(result.training_max_date, utc=True) < pd.to_datetime(result.game_date, utc=True)).all()
