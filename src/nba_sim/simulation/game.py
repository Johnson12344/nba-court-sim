from __future__ import annotations

from typing import Any

import numpy as np

from nba_sim.models import PlayerBoxScore, SimulatedGame, TeamProfile
from nba_sim.simulation.lineups import credit_minutes, sample_minute_targets, select_lineup
from nba_sim.simulation.possession import simulate_event


def _boxes(team: TeamProfile) -> dict[str, PlayerBoxScore]:
    return {player.player_id: PlayerBoxScore(player.player_id, player.name) for player in team.players}


def _score_from_boxes(boxes: dict[str, PlayerBoxScore]) -> int:
    return sum(box.points for box in boxes.values())


def simulate_game(
    home: TeamProfile,
    away: TeamProfile,
    seed: int,
    settings: dict[str, Any],
    game_index: int = 0,
    capture_events: bool = False,
) -> SimulatedGame:
    rng = np.random.default_rng(seed)
    home_boxes, away_boxes = _boxes(home), _boxes(away)
    uncertainty = float(settings["simulation"]["minute_uncertainty_sd"])
    home_targets = sample_minute_targets(home, rng, uncertainty)
    away_targets = sample_minute_targets(away, rng, uncertainty)
    scores = {home.abbreviation: 0, away.abbreviation: 0}
    quarter_scores: list[dict[str, int]] = []
    events: list[dict[str, Any]] = []
    avg_pace = (home.pace + away.pace) / 2
    base_seconds = float(settings["simulation"]["possession_seconds_mean"]) * (100.0 / avg_pace)
    sd_seconds = float(settings["simulation"]["possession_seconds_sd"])
    possessing_home = bool(rng.integers(0, 2))
    elapsed_seconds = 0.0
    period = 1
    home_lineup = None
    away_lineup = None
    next_lineup_update = 0.0
    max_periods = 4 + int(settings["simulation"].get("max_overtimes", 8))
    while True:
        if period > max_periods:
            raise RuntimeError("Game remained tied beyond configured overtime safety limit")
        duration = 720.0 if period <= 4 else 300.0
        remaining = duration
        start_home, start_away = scores[home.abbreviation], scores[away.abbreviation]
        scheduled_minutes = 48.0 + max(0, period - 4) * 5.0
        # Overtime targets extend in proportion to regulation role.
        if period > 4:
            extension = 25.0
            for player in home.players:
                home_targets[player.player_id] += extension * player.expected_minutes / 240.0
            for player in away.players:
                away_targets[player.player_id] += extension * player.expected_minutes / 240.0
        while remaining > 1e-8:
            elapsed_minutes = elapsed_seconds / 60.0
            if home_lineup is None or elapsed_seconds >= next_lineup_update:
                home_lineup = select_lineup(home, home_boxes, home_targets, elapsed_minutes, scheduled_minutes)
                away_lineup = select_lineup(away, away_boxes, away_targets, elapsed_minutes, scheduled_minutes)
                next_lineup_update = elapsed_seconds + 90.0
            seconds = float(np.clip(rng.normal(base_seconds, sd_seconds), 4.0, 24.0))
            seconds = min(seconds, remaining)
            credit_minutes(home_boxes, home_lineup, seconds)
            credit_minutes(away_boxes, away_lineup, seconds)
            offense, defense = (home, away) if possessing_home else (away, home)
            offense_lineup, defense_lineup = (home_lineup, away_lineup) if possessing_home else (away_lineup, home_lineup)
            offense_boxes, defense_boxes = (home_boxes, away_boxes) if possessing_home else (away_boxes, home_boxes)
            result = simulate_event(
                rng, offense, defense, offense_lineup, defense_lineup, offense_boxes, defense_boxes,
                possessing_home, settings,
            )
            scores[offense.abbreviation] += result.points
            remaining -= seconds
            elapsed_seconds += seconds
            if capture_events:
                event = dict(result.event)
                event.update({
                    "period": period, "clock_seconds": round(remaining, 1),
                    "home_score": scores[home.abbreviation], "away_score": scores[away.abbreviation],
                })
                events.append(event)
            if not result.retains_possession:
                possessing_home = not possessing_home
        quarter_scores.append({
            "period": period,
            "home": scores[home.abbreviation] - start_home,
            "away": scores[away.abbreviation] - start_away,
        })
        if period >= 4 and scores[home.abbreviation] != scores[away.abbreviation]:
            break
        period += 1
        possessing_home = bool(rng.integers(0, 2))
        home_lineup = None
        away_lineup = None
    # Accounting assertions are part of runtime integrity, not only tests.
    if scores[home.abbreviation] != _score_from_boxes(home_boxes):
        raise AssertionError("Home team points do not reconcile with player points")
    if scores[away.abbreviation] != _score_from_boxes(away_boxes):
        raise AssertionError("Away team points do not reconcile with player points")
    return SimulatedGame(
        game_index, int(seed), home.abbreviation, away.abbreviation,
        scores[home.abbreviation], scores[away.abbreviation], quarter_scores,
        {home.abbreviation: home_boxes, away.abbreviation: away_boxes}, events, max(0, period - 4),
    )


def simulate_baseline_game(
    home: TeamProfile,
    away: TeamProfile,
    seed: int,
    settings: dict[str, Any],
    game_index: int = 0,
) -> SimulatedGame:
    """Transparent score-level comparator; it intentionally has no player outputs."""
    rng = np.random.default_rng(seed)
    possessions = max(75, int(round(rng.normal((home.pace + away.pace) / 2, 5))))

    def expected_ppp(offense: TeamProfile, defense: TeamProfile, is_home: bool) -> float:
        shot_value = sum(
            offense.zone_shares[z] * offense.zone_make_pct[z] * (3 if "three" in z else 2)
            for z in offense.zone_shares
        )
        ppp = (1 - offense.turnover_rate) * (shot_value + offense.shooting_foul_rate * 1.2)
        ppp *= 1 + offense.offensive_rebound_rate * 0.15
        if is_home:
            ppp *= float(settings["simulation"]["home_efficiency_multiplier"])
        return float(np.clip(ppp, 0.85, 1.35))

    home_score = max(0, int(round(rng.normal(possessions * expected_ppp(home, away, True), 9))))
    away_score = max(0, int(round(rng.normal(possessions * expected_ppp(away, home, False), 9))))
    while home_score == away_score:
        home_score += int(rng.integers(3, 13))
        away_score += int(rng.integers(2, 12))
    home_q = rng.multinomial(home_score, [0.25] * 4)
    away_q = rng.multinomial(away_score, [0.25] * 4)
    quarters = [{"period": i + 1, "home": int(home_q[i]), "away": int(away_q[i])} for i in range(4)]
    return SimulatedGame(game_index, int(seed), home.abbreviation, away.abbreviation, home_score, away_score, quarters, {home.abbreviation: {}, away.abbreviation: {}}, [], 0)

