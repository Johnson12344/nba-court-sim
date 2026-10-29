from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from nba_sim.data.normalizer import normalize_columns, normalize_nba_frame
from nba_sim.data.provider import DataBundle, DataProvider, EnrichmentBundle
from nba_sim.data.validation import require_valid
from nba_sim.models import SourceMetadata


class NbaStatsProvider(DataProvider):
    """NBA.com aggregate provider. It never falls back to synthetic observations."""

    def __init__(self, raw_root: Path, processed_root: Path, settings: dict) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.settings = settings
        raw_root.mkdir(parents=True, exist_ok=True)
        processed_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(season: str) -> str:
        return season.replace("/", "-")

    def _paths(self, season: str) -> tuple[Path, Path, Path]:
        key = self._key(season)
        return (
            self.processed_root / f"teams_{key}.csv",
            self.processed_root / f"players_{key}.csv",
            self.processed_root / f"metadata_{key}.json",
        )

    def _retry(self, operation: Callable[[], pd.DataFrame], label: str) -> pd.DataFrame:
        config = self.settings["data"]
        last: Exception | None = None
        for attempt in range(int(config["max_retries"])):
            try:
                return operation()
            except Exception as exc:  # endpoint errors differ by nba_api version
                last = exc
                if attempt + 1 < int(config["max_retries"]):
                    time.sleep(float(config["backoff_seconds"]) * (2**attempt))
        raise RuntimeError(f"NBA.com request failed for {label} after retries: {last}") from last

    def _download(self, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
            from nba_api.stats.static import teams as nba_teams
        except ImportError as exc:
            raise RuntimeError("nba_api is not installed. Install requirements or use --data-source csv/--mode demo.") from exc
        timeout = int(self.settings["data"]["request_timeout_seconds"])
        team_raw = self._retry(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                season=season, season_type_all_star="Regular Season", measure_type_detailed_defense="Base", timeout=timeout
            ).get_data_frames()[0],
            "team statistics",
        )
        time.sleep(float(self.settings["data"]["request_pause_seconds"]))
        advanced_raw = self._retry(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                season=season, season_type_all_star="Regular Season", measure_type_detailed_defense="Advanced", timeout=timeout
            ).get_data_frames()[0],
            "team advanced statistics",
        )
        time.sleep(float(self.settings["data"]["request_pause_seconds"]))
        player_raw = self._retry(
            lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season", measure_type_detailed_defense="Base", timeout=timeout
            ).get_data_frames()[0],
            "player statistics",
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        team_raw.to_json(self.raw_root / f"league_team_{self._key(season)}_{stamp}.json", orient="records", indent=2)
        advanced_raw.to_json(self.raw_root / f"league_team_advanced_{self._key(season)}_{stamp}.json", orient="records", indent=2)
        player_raw.to_json(self.raw_root / f"league_player_{self._key(season)}_{stamp}.json", orient="records", indent=2)
        teams = normalize_nba_frame(team_raw, "team", season)
        advanced = normalize_nba_frame(advanced_raw, "team", season)
        abbreviations = {str(item["id"]): item["abbreviation"] for item in nba_teams.get_teams()}
        teams["team_abbreviation"] = teams["team_id"].astype(str).map(abbreviations)
        if "pace" in advanced:
            teams = teams.merge(advanced[["team_id", "pace"]], on="team_id", how="left", validate="one_to_one")
        return teams, normalize_nba_frame(player_raw, "player", season)

    def load(self, season: str, refresh: bool = False) -> DataBundle:
        team_path, player_path, metadata_path = self._paths(season)
        if not refresh and not (team_path.exists() and player_path.exists() and metadata_path.exists()):
            raise FileNotFoundError(
                f"No cached NBA.com data for {season}. Run: python -m nba_sim refresh-data --season {season}"
            )
        if refresh:
            teams, players = self._download(season)
            require_valid(teams, players)
            teams.to_csv(team_path, index=False)
            players.to_csv(player_path, index=False)
            metadata = {
                "source": "NBA.com via nba_api",
                "season": season,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "coverage": {"teams": len(teams), "players": len(players), "play_by_play": False, "shot_zones": False},
                "assumptions": [
                    "Aggregate Base endpoints do not identify possession-level defenders or screens.",
                    "Shot-zone and play-by-play enrichment are unavailable in this aggregate refresh.",
                ],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        teams = pd.read_csv(team_path)
        players = pd.read_csv(player_path)
        require_valid(teams, players)
        raw_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta = SourceMetadata(
            raw_meta["source"], raw_meta["season"], raw_meta["retrieved_at"], False,
            raw_meta.get("coverage", {}), tuple(raw_meta.get("assumptions", [])),
        )
        return DataBundle(teams, players, meta)

    def _save_raw(self, label: str, season: str, frame: pd.DataFrame) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.raw_root / f"{label}_{self._key(season)}_{stamp}.json"
        frame.to_json(target, orient="records", indent=2)
        return target

    def refresh_enrichment(
        self,
        season: str,
        team_abbreviations: list[str],
        games_per_team: int = 20,
        rotation_games: int = 3,
        force: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> EnrichmentBundle:
        """Cache shot charts, PlayByPlayV3 events, and recent verified rotations."""
        if not 1 <= games_per_team <= 82:
            raise ValueError("games_per_team must be between 1 and 82")
        if not 1 <= rotation_games <= 10:
            raise ValueError("rotation_games must be between 1 and 10")
        bundle = self.load(season, refresh=False)
        requested = sorted({value.upper() for value in team_abbreviations})
        known = set(bundle.teams.team_abbreviation.astype(str).str.upper())
        missing = sorted(set(requested) - known)
        if missing:
            raise ValueError(f"Unknown team abbreviation(s) for {season}: {', '.join(missing)}")
        try:
            from nba_api.stats.endpoints import boxscoretraditionalv3, leaguegamelog, playbyplayv3, shotchartdetail
        except ImportError as exc:
            raise RuntimeError("nba_api is required for enrichment refresh") from exc
        timeout = int(self.settings["data"]["request_timeout_seconds"])
        pause = float(self.settings["data"]["request_pause_seconds"])
        key = self._key(season)
        game_log_path = self.processed_root / f"game_log_{key}.csv"
        # These season-wide observations change; only completed per-game files are reusable.
        if progress:
            progress("Downloading league game log")
        raw_games = self._retry(
            lambda: leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation="T", timeout=timeout,
            ).get_data_frames()[0],
            "league game log",
        )
        self._save_raw("league_game_log", season, raw_games)
        normalize_columns(raw_games).to_csv(game_log_path, index=False)
        time.sleep(pause)
        game_log = pd.read_csv(game_log_path, dtype={"game_id": str})
        game_log["team_abbreviation"] = game_log.team_abbreviation.astype(str).str.upper()
        selected_rows: list[pd.DataFrame] = []
        for abbreviation in requested:
            rows = game_log.loc[game_log.team_abbreviation == abbreviation].copy()
            rows["game_date"] = pd.to_datetime(rows.game_date, errors="coerce")
            selected_rows.append(rows.sort_values("game_date", ascending=False).head(games_per_team))
        selected_games = pd.concat(selected_rows, ignore_index=True)
        game_ids = sorted(set(selected_games.game_id.astype(str)))

        team_ids = {
            str(row.team_abbreviation).upper(): int(row.team_id)
            for row in bundle.teams.itertuples()
        }
        for abbreviation in requested:
            shot_path = self.processed_root / f"shots_{key}_{abbreviation}.csv"
            if progress:
                progress(f"Downloading {abbreviation} player shot chart")
            raw_shots = self._retry(
                lambda abbr=abbreviation: shotchartdetail.ShotChartDetail(
                    team_id=team_ids[abbr], player_id=0, context_measure_simple="FGA",
                    season_nullable=season, season_type_all_star="Regular Season", timeout=timeout,
                ).get_data_frames()[0],
                f"{abbreviation} shot chart",
            )
            self._save_raw(f"shot_chart_{abbreviation}", season, raw_shots)
            shots = normalize_columns(raw_shots)
            shots["team_abbreviation"] = abbreviation
            shots.to_csv(shot_path, index=False)
            time.sleep(pause)

        for game_id in game_ids:
            pbp_path = self.processed_root / f"pbp_{key}_{game_id}.csv"
            if force or not pbp_path.exists():
                if progress:
                    progress(f"Downloading play-by-play {game_id}")
                raw_pbp = self._retry(
                    lambda gid=game_id: playbyplayv3.PlayByPlayV3(game_id=gid, timeout=timeout).get_data_frames()[0],
                    f"play-by-play {game_id}",
                )
                self._save_raw(f"play_by_play_{game_id}", season, raw_pbp)
                normalize_columns(raw_pbp).to_csv(pbp_path, index=False)
                time.sleep(pause)

        rotation_ids: set[str] = set()
        for abbreviation in requested:
            rows = selected_games.loc[selected_games.team_abbreviation == abbreviation].copy()
            rows["game_date"] = pd.to_datetime(rows.game_date, errors="coerce")
            rotation_ids.update(rows.sort_values("game_date", ascending=False).head(rotation_games).game_id.astype(str))
        for game_id in sorted(rotation_ids):
            box_path = self.processed_root / f"box_{key}_{game_id}.csv"
            if force or not box_path.exists():
                if progress:
                    progress(f"Downloading rotation box score {game_id}")
                raw_box = self._retry(
                    lambda gid=game_id: boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, timeout=timeout).get_data_frames()[0],
                    f"box score {game_id}",
                )
                self._save_raw(f"box_score_{game_id}", season, raw_box)
                normalize_columns(raw_box).to_csv(box_path, index=False)
                time.sleep(pause)
        manifest_path = self.processed_root / f"enrichment_{key}.json"
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        refreshed_at = datetime.now(timezone.utc).isoformat()
        team_refreshes = previous.get("team_refreshes", {})
        team_refreshes.update({team: refreshed_at for team in requested})
        manifest = {
            "season": season, "teams": requested, "games_per_team": games_per_team,
            "rotation_games": rotation_games, "game_ids": game_ids,
            "rotation_game_ids": sorted(rotation_ids), "refreshed_at": refreshed_at,
            "team_refreshes": team_refreshes,
            "sources": ["NBA.com ShotChartDetail", "NBA.com PlayByPlayV3", "NBA.com BoxScoreTraditionalV3"],
            "limitations": ["PlayByPlayV3 does not provide an observed on-ball defender for every shot."],
        }
        (self.processed_root / f"enrichment_{key}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.load_enrichment(season, requested)

    def load_enrichment(self, season: str, teams: list[str]) -> EnrichmentBundle:
        key = self._key(season)
        requested = {value.upper() for value in teams}
        shot_frames = [
            pd.read_csv(path, dtype={"game_id": str, "player_id": str, "team_id": str})
            for abbreviation in requested
            for path in [self.processed_root / f"shots_{key}_{abbreviation}.csv"]
            if path.exists()
        ]
        pbp_paths = list(self.processed_root.glob(f"pbp_{key}_*.csv"))
        pbp_frames = [pd.read_csv(path, dtype={"game_id": str, "person_id": str, "team_id": str}) for path in pbp_paths]
        box_paths = list(self.processed_root.glob(f"box_{key}_*.csv"))
        box_frames = [pd.read_csv(path, dtype={"game_id": str, "person_id": str, "team_id": str}) for path in box_paths]
        shots = pd.concat(shot_frames, ignore_index=True) if shot_frames else pd.DataFrame()
        pbp = pd.concat(pbp_frames, ignore_index=True) if pbp_frames else pd.DataFrame()
        rotations = pd.concat(box_frames, ignore_index=True) if box_frames else pd.DataFrame()
        if not pbp.empty and "team_tricode" in pbp:
            # Neutral events from unrelated cached games must not imply matchup coverage.
            pbp = pbp.loc[pbp.team_tricode.astype(str).str.upper().isin(requested)].copy()
        if not rotations.empty and "team_tricode" in rotations:
            rotations = rotations.loc[rotations.team_tricode.astype(str).str.upper().isin(requested)].copy()
        manifest_path = self.processed_root / f"enrichment_{key}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        team_dates = manifest.get("team_refreshes", {})
        refreshed_at = min(team_dates[t] for t in requested) if requested and all(t in team_dates for t in requested) else (
            manifest.get("refreshed_at") if requested <= set(manifest.get("teams", [])) else None
        )
        metadata = {
            **manifest,
            "refreshed_at": refreshed_at,
            "available": not shots.empty or not pbp.empty or not rotations.empty,
            "shot_rows": len(shots), "play_by_play_rows": len(pbp), "rotation_rows": len(rotations),
        }
        return EnrichmentBundle(shots, pbp, rotations, metadata)

