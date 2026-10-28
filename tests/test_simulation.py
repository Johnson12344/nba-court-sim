from __future__ import annotations

from dataclasses import asdict

import pytest

from nba_sim.simulation.game import simulate_game
from nba_sim.simulation.monte_carlo import run_simulations
from nba_sim.simulation.possession import PossessionResult


def test_fixed_seed_repeatability(demo_profiles, settings):
    home, away = demo_profiles
    one = simulate_game(home, away, 1234, settings, capture_events=True)
    two = simulate_game(home, away, 1234, settings, capture_events=True)
    assert asdict(one) == asdict(two)


def test_game_accounting_and_state_transitions(demo_profiles, settings):
    home, away = demo_profiles
    game = simulate_game(home, away, 99, settings, capture_events=True)
    assert game.home_score == sum(box.points for box in game.player_stats[home.abbreviation].values())
    assert game.away_score == sum(box.points for box in game.player_stats[away.abbreviation].values())
    assert sum(q["home"] for q in game.quarter_scores) == game.home_score
    assert sum(q["away"] for q in game.quarter_scores) == game.away_score
    assert game.home_score != game.away_score
    assert any(event["type"] == "turnover" for event in game.events)
    assert any(event["type"] in {"made_shot", "missed_shot"} for event in game.events)
    assert sum(box.minutes for box in game.player_stats[home.abbreviation].values()) == pytest.approx((48 + 5 * game.overtimes) * 5)


def test_representative_game_replays_exact_seed(demo_profiles, settings):
    home, away = demo_profiles
    games, representative = run_simulations(home, away, 12, 42, settings)
    original = games[representative.game_index]
    assert (original.seed, original.home_score, original.away_score) == (representative.seed, representative.home_score, representative.away_score)
    assert representative.events


def test_simulation_count_is_capped(demo_profiles, settings):
    with pytest.raises(ValueError, match="capped"):
        run_simulations(*demo_profiles, 10_001, 42, settings)


def test_overtime_transition(monkeypatch, demo_profiles, settings):
    import nba_sim.simulation.game as game_module

    home, away = demo_profiles
    pace_adjusted_mean = 24 * ((home.pace + away.pace) / 2) / 100
    local = {**settings, "simulation": {**settings["simulation"], "possession_seconds_mean": pace_adjusted_mean, "possession_seconds_sd": 0}}
    counter = {"events": 0}

    def controlled_event(rng, offense, defense, offense_lineup, defense_lineup, offense_boxes, defense_boxes, home_offense, settings):
        counter["events"] += 1
        points = 2 if counter["events"] == 121 else 0
        if points:
            offense_boxes[offense_lineup[0].player_id].points += points
        return PossessionResult(points, False, {"type": "controlled", "team": offense.abbreviation, "points": points})

    monkeypatch.setattr(game_module, "simulate_event", controlled_event)
    game = simulate_game(home, away, seed=3, settings=local)
    assert game.overtimes == 1
    assert len(game.quarter_scores) == 5
    assert game.home_score != game.away_score

