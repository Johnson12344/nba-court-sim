from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd

from nba_sim.data.nba_stats_provider import NbaStatsProvider


def test_enrichment_refresh_updates_season_data_and_reuses_game_files(tmp_path, monkeypatch, demo_bundle, settings):
    from nba_api.stats.endpoints import boxscoretraditionalv3, leaguegamelog, playbyplayv3, shotchartdetail

    local = deepcopy(settings)
    local["data"]["request_pause_seconds"] = 0
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    provider = NbaStatsProvider(raw, processed, local)
    demo_bundle.teams["team_id"] = [1, 2]
    monkeypatch.setattr(provider, "load", lambda *a, **k: demo_bundle)
    counts = {key: 0 for key in ("log", "shot", "pbp", "box")}

    def response(key, frame):
        counts[key] += 1
        return SimpleNamespace(get_data_frames=lambda: [frame])

    logs = pd.DataFrame([{"GAME_ID": "0000000001", "TEAM_ABBREVIATION": "OKC", "GAME_DATE": "2026-10-01"},
                         {"GAME_ID": "0000000001", "TEAM_ABBREVIATION": "SAS", "GAME_DATE": "2026-10-01"}])
    monkeypatch.setattr(leaguegamelog, "LeagueGameLog", lambda **kw: response("log", logs))
    monkeypatch.setattr(shotchartdetail, "ShotChartDetail", lambda **kw: response("shot", pd.DataFrame([{"PLAYER_ID": 1, "SHOT_MADE_FLAG": 1}])))
    monkeypatch.setattr(playbyplayv3, "PlayByPlayV3", lambda **kw: response("pbp", pd.DataFrame([
        {"gameId": "0000000001", "teamTricode": "OKC"}, {"gameId": "0000000001", "teamTricode": None}])))
    monkeypatch.setattr(boxscoretraditionalv3, "BoxScoreTraditionalV3", lambda **kw: response("box", pd.DataFrame([
        {"gameId": "0000000001", "teamTricode": "OKC", "personId": 1, "minutes": "20:00", "position": "G"}])))
    provider.refresh_enrichment("2025-26", ["OKC", "SAS"], 1, 1)
    provider.refresh_enrichment("2025-26", ["OKC", "SAS"], 1, 1)
    assert counts == {"log": 2, "shot": 4, "pbp": 1, "box": 1}
    matched = provider.load_enrichment("2025-26", ["OKC", "SAS"])
    assert matched.metadata["refreshed_at"]
    unrelated = provider.load_enrichment("2025-26", ["NOP", "SAC"])
    assert not unrelated.metadata["available"]
    assert unrelated.metadata["refreshed_at"] is None
