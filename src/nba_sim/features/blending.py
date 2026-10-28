from __future__ import annotations

from dataclasses import replace

from nba_sim.models import PlayerProfile, TeamProfile, ZONES


def _mix(old: float, new: float, new_weight: float) -> float:
    return (1.0 - new_weight) * old + new_weight * new


def blend_team_profiles(
    historical: TeamProfile,
    recent: TeamProfile,
    recent_weight: float,
    season_label: str,
    recent_minutes_weight: float = 0.25,
) -> TeamProfile:
    """Blend rates while using the newer observation's verified roster assignment."""
    if not 0 <= recent_weight <= 1:
        raise ValueError("recent_weight must be between zero and one")
    historical_players = {player.player_id: player for player in historical.players}
    players: list[PlayerProfile] = []
    has_verified_recent_rotation = bool(recent.data_evidence.get("rotation_games", 0))
    minute_weight = max(recent_weight, 0.75 if has_verified_recent_rotation else recent_minutes_weight)
    for new in recent.players:
        old = historical_players.get(new.player_id)
        if old is None:
            players.append(replace(new, reliability=f"new-roster observation; {new.reliability}"))
            continue
        players.append(PlayerProfile(
            player_id=new.player_id, name=new.name, team=new.team,
            games=new.games + old.games, minutes=new.minutes + old.minutes,
            expected_minutes=_mix(old.expected_minutes, new.expected_minutes, minute_weight),
            usage_weight=_mix(old.usage_weight, new.usage_weight, recent_weight),
            turnover_rate=_mix(old.turnover_rate, new.turnover_rate, recent_weight),
            assist_rate=_mix(old.assist_rate, new.assist_rate, recent_weight),
            oreb_rate=_mix(old.oreb_rate, new.oreb_rate, recent_weight),
            dreb_rate=_mix(old.dreb_rate, new.dreb_rate, recent_weight),
            ft_rate=_mix(old.ft_rate, new.ft_rate, recent_weight),
            ft_pct=_mix(old.ft_pct, new.ft_pct, recent_weight),
            zone_shares={zone: _mix(old.zone_shares[zone], new.zone_shares[zone], recent_weight) for zone in ZONES},
            zone_make_pct={zone: _mix(old.zone_make_pct[zone], new.zone_make_pct[zone], recent_weight) for zone in ZONES},
            sample_sizes={key: old.sample_sizes.get(key, 0) + new.sample_sizes.get(key, 0) for key in set(old.sample_sizes) | set(new.sample_sizes)},
            reliability=f"sample-size blend ({recent_weight:.1%} recent); {new.reliability}",
            starter=new.starter,
            shooter_selection_source=new.shooter_selection_source,
            shot_location_source=new.shot_location_source,
            assister_weights=new.assister_weights or old.assister_weights,
            shot_locations=new.shot_locations or old.shot_locations,
        ))
    # Renormalize expected regulation minutes after roster blending.
    remaining = set(range(len(players)))
    allocated = [0.0] * len(players)
    target = 240.0
    while remaining and target > 1e-9:
        weight_total = sum(players[index].expected_minutes for index in remaining)
        proposed = {index: target * players[index].expected_minutes / weight_total for index in remaining}
        capped = {index for index, value in proposed.items() if value >= 48.0}
        if not capped:
            for index, value in proposed.items():
                allocated[index] = value
            break
        for index in capped:
            allocated[index] = 48.0
            target -= 48.0
            remaining.remove(index)
    players = [replace(player, expected_minutes=allocated[index]) for index, player in enumerate(players)]
    return TeamProfile(
        team_id=recent.team_id, abbreviation=recent.abbreviation, name=recent.name, season=season_label,
        games=historical.games + recent.games,
        pace=_mix(historical.pace, recent.pace, recent_weight),
        turnover_rate=_mix(historical.turnover_rate, recent.turnover_rate, recent_weight),
        shooting_foul_rate=_mix(historical.shooting_foul_rate, recent.shooting_foul_rate, recent_weight),
        offensive_rebound_rate=_mix(historical.offensive_rebound_rate, recent.offensive_rebound_rate, recent_weight),
        assist_rate=_mix(historical.assist_rate, recent.assist_rate, recent_weight),
        zone_shares={zone: _mix(historical.zone_shares[zone], recent.zone_shares[zone], recent_weight) for zone in ZONES},
        zone_make_pct={zone: _mix(historical.zone_make_pct[zone], recent.zone_make_pct[zone], recent_weight) for zone in ZONES},
        defensive_zone_factor={zone: _mix(historical.defensive_zone_factor[zone], recent.defensive_zone_factor[zone], recent_weight) for zone in ZONES},
        players=tuple(players), data_evidence={**historical.data_evidence, **recent.data_evidence},
    )

