from __future__ import annotations

import pytest

from nba_sim.features.historical_weights import current_season_weight, shrink_rate
from nba_sim.features.profile_builder import build_team_profile
from nba_sim.simulation.lineups import sample_minute_targets


def test_shrinkage_respects_sample_size():
    tiny = shrink_rate(5, 8, 0.36, 80)
    large = shrink_rate(500, 800, 0.36, 80)
    assert abs(tiny - 0.36) < abs(5 / 8 - 0.36)
    assert large > tiny
    assert current_season_weight(0, 20) == 0
    assert current_season_weight(20, 20) == 0.5


def test_profile_probabilities_and_minutes(demo_profiles):
    for team in demo_profiles:
        assert sum(player.expected_minutes for player in team.players) == pytest.approx(240)
        assert sum(team.zone_shares.values()) == pytest.approx(1)
        assert all(sum(player.zone_shares.values()) == pytest.approx(1) for player in team.players)


def test_sampled_minute_targets_are_valid(demo_profiles):
    import numpy as np

    for team in demo_profiles:
        targets = sample_minute_targets(team, np.random.default_rng(7), 0.08)
        assert sum(targets.values()) == pytest.approx(240)
        assert max(targets.values()) <= 48


def test_invalid_probability_rejected(demo_profiles):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(demo_profiles[0], turnover_rate=1.5)


def test_play_by_play_shots_and_rotations_enrich_profiles(settings, demo_bundle):
    import pandas as pd

    players = demo_bundle.players.loc[demo_bundle.players.team_abbreviation == "OKC"].head(7)
    ids = players.player_id.astype(str).tolist()
    shots = pd.DataFrame([
        {
            "team_abbreviation": "OKC", "player_id": ids[0],
            "shot_zone_basic": "Restricted Area" if index < 18 else "Left Corner 3",
            "shot_zone_area": "Center(C)" if index < 18 else "Left Side(L)",
            "shot_distance": 2 if index < 18 else 23, "shot_made_flag": int(index % 2 == 0),
            "loc_x": 0 if index < 18 else -220, "loc_y": 10 if index < 18 else 20,
        }
        for index in range(24)
    ])
    pbp = pd.DataFrame([
        {
            "team_tricode": "OKC", "is_field_goal": 1, "person_id": ids[0],
            "game_id": f"G{index % 2}", "shot_result": "Made",
            "description": f"Player 1 shot (2 {index + 1} AST)",
        }
        for index in range(12)
    ])
    rotations = pd.DataFrame([
        {
            "team_tricode": "OKC", "game_id": "G2", "person_id": player_id,
            "minutes": f"{34 - index}:00", "position": "G" if index < 5 else "",
        }
        for index, player_id in enumerate(ids)
    ])
    profile = build_team_profile(
        demo_bundle.teams, demo_bundle.players, "OKC", settings,
        shots=shots, play_by_play=pbp, rotations=rotations,
    )
    shooter = next(player for player in profile.players if player.player_id == ids[0])
    assert profile.data_evidence["play_by_play_games"] == 2
    assert profile.data_evidence["shot_chart_attempts"] == 24
    assert shooter.starter
    assert shooter.sample_sizes["play_by_play_shots"] == 12
    assert shooter.zone_shares["rim"] > shooter.zone_shares["midrange"]
    assert shooter.shot_locations["rim"]
    assert shooter.assister_weights[ids[1]] == 12

