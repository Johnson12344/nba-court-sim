from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nba_sim.models import PlayerBoxScore, PlayerProfile, TeamProfile


def weighted_choice(rng: np.random.Generator, items: Sequence[PlayerProfile], weights: Sequence[float]) -> PlayerProfile:
    if len(items) == 0:
        raise ValueError("Cannot choose from an empty lineup")
    cleaned = [float(value) if np.isfinite(value) and value > 0 else 0.0 for value in weights]
    total = sum(cleaned)
    if total <= 0:
        return items[int(rng.integers(0, len(items)))]
    threshold = float(rng.random()) * total
    cumulative = 0.0
    for item, weight in zip(items, cleaned):
        cumulative += weight
        if threshold <= cumulative:
            return item
    return items[-1]


def choose_shooter(rng: np.random.Generator, lineup: Sequence[PlayerProfile]) -> PlayerProfile:
    return weighted_choice(rng, lineup, [player.usage_weight for player in lineup])


def choose_zone(rng: np.random.Generator, shooter: PlayerProfile, offense: TeamProfile) -> str:
    zones = list(offense.zone_shares)
    weights = [0.7 * shooter.zone_shares[zone] + 0.3 * offense.zone_shares[zone] for zone in zones]
    threshold = float(rng.random()) * sum(weights)
    cumulative = 0.0
    for zone, weight in zip(zones, weights):
        cumulative += weight
        if threshold <= cumulative:
            return zone
    return zones[-1]


def choose_location(rng: np.random.Generator, shooter: PlayerProfile, zone: str) -> dict[str, Any]:
    locations = shooter.shot_locations.get(zone, ())
    if not locations:
        return {"location_source": "zone only"}
    x, y, distance = locations[int(rng.integers(0, len(locations)))]
    return {
        "location_x": x, "location_y": y, "shot_distance": distance,
        "location_source": "sampled from shooter's observed ShotChartDetail locations",
    }


@dataclass(frozen=True)
class PossessionResult:
    points: int
    retains_possession: bool
    event: dict[str, Any]


def simulate_event(
    rng: np.random.Generator,
    offense: TeamProfile,
    defense: TeamProfile,
    offense_lineup: tuple[PlayerProfile, ...],
    defense_lineup: tuple[PlayerProfile, ...],
    offense_boxes: dict[str, PlayerBoxScore],
    defense_boxes: dict[str, PlayerBoxScore],
    home_offense: bool,
    settings: dict[str, Any],
) -> PossessionResult:
    turnover_p = float(np.clip(offense.turnover_rate, 0.01, 0.35))
    foul_p = float(np.clip(offense.shooting_foul_rate, 0.01, 0.30))
    draw = rng.random()
    if draw < turnover_p:
        player = weighted_choice(rng, offense_lineup, [p.usage_weight * max(0.02, p.turnover_rate) for p in offense_lineup])
        player_box = offense_boxes[player.player_id]
        player_box.turnovers += 1
        steal = rng.random() < 0.58
        defender = weighted_choice(rng, defense_lineup, [1.0 for _ in defense_lineup])
        if steal:
            defense_boxes[defender.player_id].steals += 1
        return PossessionResult(0, False, {
            "type": "turnover", "team": offense.abbreviation, "player": player.name,
            "steal_by": defender.name if steal else None, "points": 0,
        })

    shooter = choose_shooter(rng, offense_lineup)
    zone = choose_zone(rng, shooter, offense)
    location = choose_location(rng, shooter, zone)
    is_three = "three" in zone
    shooter_box = offense_boxes[shooter.player_id]
    if draw < turnover_p + foul_p:
        attempts = 3 if is_three and rng.random() < 0.22 else 2
        ft_results = rng.random(attempts) < shooter.ft_pct
        makes = int(np.sum(ft_results))
        shooter_box.fta += attempts
        shooter_box.ftm += makes
        shooter_box.points += makes
        fouler = weighted_choice(rng, defense_lineup, [1.0 for _ in defense_lineup])
        defense_boxes[fouler.player_id].fouls += 1
        retained = False
        rebounder: str | None = None
        if not bool(ft_results[-1]):
            retained = rng.random() < offense.offensive_rebound_rate
            if retained:
                rebound = weighted_choice(rng, offense_lineup, [p.oreb_rate for p in offense_lineup])
                offense_boxes[rebound.player_id].oreb += 1
            else:
                rebound = weighted_choice(rng, defense_lineup, [p.dreb_rate for p in defense_lineup])
                defense_boxes[rebound.player_id].dreb += 1
            rebounder = rebound.name
        event = {
            "type": "shooting_foul", "team": offense.abbreviation, "shooter": shooter.name,
            "fouler": fouler.name, "free_throws": attempts, "made": makes,
            "rebounder": rebounder, "offensive_rebound": retained, "points": makes,
            "zone": zone, **location,
        }
        return PossessionResult(makes, retained, event)

    shooter_box.fga += 1
    if is_three:
        shooter_box.fg3a += 1
    player_pct = shooter.zone_make_pct[zone]
    team_pct = offense.zone_make_pct[zone]
    defense_factor = defense.defensive_zone_factor.get(zone, 1.0)
    home_factor = float(settings["simulation"]["home_efficiency_multiplier"]) if home_offense else 1.0
    make_probability = float(np.clip(np.sqrt(player_pct * team_pct) * defense_factor * home_factor, 0.15, 0.80))
    made = rng.random() < make_probability
    point_value = 3 if is_three else 2
    if made:
        shooter_box.fgm += 1
        shooter_box.points += point_value
        if is_three:
            shooter_box.fg3m += 1
        assister: str | None = None
        if rng.random() < offense.assist_rate:
            eligible = [player for player in offense_lineup if player.player_id != shooter.player_id]
            if eligible:
                helper = weighted_choice(
                    rng, eligible,
                    [shooter.assister_weights.get(p.player_id, 0.0) + 2.0 * p.assist_rate for p in eligible],
                )
                offense_boxes[helper.player_id].assists += 1
                assister = helper.name
        event = {
            "type": "made_shot", "team": offense.abbreviation, "shooter": shooter.name,
            "zone": zone, "value": point_value, "assist_by": assister,
            "make_probability": round(make_probability, 4), "points": point_value,
            **location,
        }
        return PossessionResult(point_value, False, event)

    retained = rng.random() < offense.offensive_rebound_rate
    if retained:
        rebounder = weighted_choice(rng, offense_lineup, [p.oreb_rate for p in offense_lineup])
        offense_boxes[rebounder.player_id].oreb += 1
    else:
        rebounder = weighted_choice(rng, defense_lineup, [p.dreb_rate for p in defense_lineup])
        defense_boxes[rebounder.player_id].dreb += 1
    event = {
        "type": "missed_shot", "team": offense.abbreviation, "shooter": shooter.name,
        "zone": zone, "value": point_value, "rebounder": rebounder.name,
        "offensive_rebound": retained, "make_probability": round(make_probability, 4), "points": 0,
        **location,
    }
    return PossessionResult(0, retained, event)

