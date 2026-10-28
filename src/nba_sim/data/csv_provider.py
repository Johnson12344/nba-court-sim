from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_sim.data.normalizer import normalize_columns
from nba_sim.data.provider import DataBundle, DataProvider, EnrichmentBundle
from nba_sim.data.validation import require_valid
from nba_sim.models import SourceMetadata


class CsvProvider(DataProvider):
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, season: str, refresh: bool = False) -> DataBundle:
        team_path = self.root / "team_stats.csv"
        player_path = self.root / "player_stats.csv"
        missing = [str(path) for path in (team_path, player_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("Manual CSV data missing: " + ", ".join(missing))
        teams = pd.read_csv(team_path)
        players = pd.read_csv(player_path)
        teams = teams.loc[teams["season"].astype(str) == season].copy()
        players = players.loc[players["season"].astype(str) == season].copy()
        require_valid(teams, players)
        retrieved = datetime.fromtimestamp(max(team_path.stat().st_mtime, player_path.stat().st_mtime), tz=timezone.utc).isoformat()
        meta = SourceMetadata("manual_csv", season, retrieved, False, {"teams": len(teams), "players": len(players)}, ("CSV provenance is user supplied and cannot be independently verified by this program.",))
        return DataBundle(teams, players, meta)

    def load_enrichment(self, season: str, teams: list[str]) -> EnrichmentBundle:
        frames: dict[str, pd.DataFrame] = {}
        for key, filename in {
            "shots": "shot_events.csv",
            "play_by_play": "play_by_play.csv",
            "rotations": "rotations.csv",
        }.items():
            path = self.root / filename
            frame = normalize_columns(pd.read_csv(path)) if path.exists() else pd.DataFrame()
            if not frame.empty and "season" in frame:
                frame = frame.loc[frame.season.astype(str) == season].copy()
            frames[key] = frame
        return EnrichmentBundle(
            frames["shots"], frames["play_by_play"], frames["rotations"],
            {
                "available": any(not frame.empty for frame in frames.values()),
                "source": "manual CSV enrichment",
                "shot_rows": len(frames["shots"]),
                "play_by_play_rows": len(frames["play_by_play"]),
                "rotation_rows": len(frames["rotations"]),
            },
        )

