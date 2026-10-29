from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nba_sim.models import SimulatedGame
from nba_sim.reporting.narrative import representative_narrative


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def export_run(
    output_root: Path,
    summary: dict[str, Any],
    scores: pd.DataFrame,
    players: pd.DataFrame,
    representative: SimulatedGame,
    metadata: dict[str, Any],
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_{metadata['away_team']}_at_{metadata['home_team']}_{metadata['seed']}"
    target = output_root / run_id
    target.mkdir(parents=True, exist_ok=False)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    scores.to_csv(target / "game_scores.csv", index=False)
    players.to_csv(target / "player_distributions.csv", index=False)
    (target / "representative_game.json").write_text(json.dumps(representative_narrative(representative), indent=2, default=_json_default), encoding="utf-8")
    (target / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    if summary.get("betting_analysis"):
        analysis = summary["betting_analysis"]
        (target / "betting_analysis.json").write_text(json.dumps(analysis, indent=2, default=_json_default), encoding="utf-8")
        pd.DataFrame(analysis["offers"]).to_csv(target / "market_probabilities.csv", index=False)
    return target

