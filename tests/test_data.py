from __future__ import annotations

import json

import pandas as pd
import pytest

from nba_sim.data.cache import JsonCache
from nba_sim.data.normalizer import normalize_nba_frame
from nba_sim.data.validation import validate_data


def test_nba_column_normalization_adds_season():
    frame = pd.DataFrame([{"TEAM_ID": 1, "TEAM_NAME": "Example", "GP": 2, "PTS": 200, "FG3A": 50}])
    result = normalize_nba_frame(frame, "team", "2025-26")
    assert {"team_id", "team_name", "games", "points", "fg3a", "season"} <= set(result.columns)
    assert result.loc[0, "season"] == "2025-26"


def test_demo_is_explicit_and_valid(demo_bundle):
    report = validate_data(demo_bundle.teams, demo_bundle.players)
    assert report.valid, report.errors
    assert demo_bundle.metadata.is_demo
    assert "DEMO DATA" in " ".join(demo_bundle.metadata.assumptions)
    assert demo_bundle.players.player_id.str.startswith("DEMO-").all()


def test_validation_rejects_impossible_shooting(demo_bundle):
    players = demo_bundle.players.copy()
    players.loc[0, "fgm"] = players.loc[0, "fga"] + 1
    report = validate_data(demo_bundle.teams, players)
    assert not report.valid
    assert any("fgm exceeds fga" in error for error in report.errors)


def test_validation_rejects_missing_column(demo_bundle):
    report = validate_data(demo_bundle.teams.drop(columns="points"), demo_bundle.players)
    assert not report.valid
    assert any("missing required columns" in error for error in report.errors)


def test_cache_round_trip_and_no_silent_overwrite(tmp_path):
    cache = JsonCache(tmp_path)
    first = cache.save("sample", {"value": 1}, {"source": "test"})
    second = cache.save("sample", {"value": 2})
    assert first != second
    loaded = cache.load("sample")
    assert loaded["payload"] == {"value": 1}
    assert json.loads(second.read_text())["payload"] == {"value": 2}

