from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from nba_sim.features.historical_weights import shrink_rate
from nba_sim.models import PlayerProfile, TeamProfile, ZONES


def _safe(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if denominator <= 0 else numerator / denominator


def _normalize(values: dict[str, float], fallback: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(values.get(key, 0.0))) for key in fallback}
    total = sum(cleaned.values())
    return dict(fallback) if total <= 0 else {key: value / total for key, value in cleaned.items()}


def _id(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _minutes_value(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    match = re.match(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", text)
    if match:
        return float(match.group(1) or 0) + float(match.group(2) or 0) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _zone_from_shot(row: pd.Series) -> str:
    basic = str(row.get("shot_zone_basic", "")).lower()
    area = str(row.get("shot_zone_area", "")).lower()
    distance = float(row.get("shot_distance", 0) or 0)
    if "restricted" in basic or distance <= 4:
        return "rim"
    if "paint" in basic:
        return "paint"
    if "mid-range" in basic or "midrange" in basic:
        return "midrange"
    if "corner" in basic or "corner" in area:
        return "corner_three"
    if "3" in basic or distance >= 22:
        return "above_break_three"
    return "paint" if distance < 10 else "midrange"


def _rotation_observations(rotation: pd.DataFrame | None, abbreviation: str) -> tuple[dict[str, float], set[str], int]:
    if rotation is None or rotation.empty or "team_tricode" not in rotation:
        return {}, set(), 0
    team_rows = rotation.loc[rotation.team_tricode.astype(str).str.upper() == abbreviation.upper()].copy()
    if team_rows.empty:
        return {}, set(), 0
    games = sorted(team_rows.game_id.astype(str).unique(), reverse=True)[:3]
    team_rows = team_rows.loc[team_rows.game_id.astype(str).isin(games)].copy()
    team_rows["parsed_minutes"] = team_rows.minutes.map(_minutes_value)
    minute_map = {
        _id(player_id): float(value)
        for player_id, value in team_rows.groupby("person_id").parsed_minutes.mean().items()
        if value > 0
    }
    latest = team_rows.loc[team_rows.game_id.astype(str) == games[0]]
    starters = {
        _id(row.person_id) for row in latest.itertuples()
        if str(getattr(row, "position", "")).strip() not in {"", "nan", "None"}
    }
    return minute_map, starters, len(games)


def _expected_minutes(
    group: pd.DataFrame,
    team_games: float,
    recent_minutes: dict[str, float] | None = None,
    recent_weight: float = 0.0,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for row in group.to_dict("records"):
        mpg = float(row.get("expected_minutes", np.nan))
        if not np.isfinite(mpg):
            # Total minutes / team games incorporates observed availability;
            # minutes / player games would over-allocate teams with absences.
            mpg = _safe(float(row["minutes"]), team_games)
        if mpg > 0:
            raw[_id(row["player_id"])] = min(48.0, mpg)
    if recent_minutes:
        for player_id in list(raw):
            observed = recent_minutes.get(player_id, 0.0)
            raw[player_id] = (1.0 - recent_weight) * raw[player_id] + recent_weight * observed
    if len(raw) < 5:
        raise ValueError("At least five players with positive expected minutes are required")
    # Only the likely rotation is retained, then scaled to exactly 240 regulation minutes.
    retained = dict(sorted(raw.items(), key=lambda item: item[1], reverse=True)[:15])
    keys = list(retained)
    weights = np.array([retained[key] for key in keys], dtype=float)
    allocated = np.zeros_like(weights)
    open_mask = np.ones(len(weights), dtype=bool)
    target = 240.0
    while open_mask.any() and target > 1e-9:
        indices = np.flatnonzero(open_mask)
        shares = target * weights[indices] / weights[indices].sum()
        capped = shares >= 48.0
        if not capped.any():
            allocated[indices] = shares
            target = 0.0
            break
        for index in indices[capped]:
            allocated[index] = 48.0
            open_mask[index] = False
            target -= 48.0
    return {key: float(value) for key, value in zip(keys, allocated)}


def build_team_profile(
    teams: pd.DataFrame,
    players: pd.DataFrame,
    abbreviation: str,
    settings: dict[str, Any],
    shots: pd.DataFrame | None = None,
    play_by_play: pd.DataFrame | None = None,
    rotations: pd.DataFrame | None = None,
) -> TeamProfile:
    selected = teams.loc[teams.team_abbreviation.str.upper() == abbreviation.upper()]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one team row for {abbreviation}; found {len(selected)}")
    team = selected.iloc[0]
    group = players.loc[players.team_abbreviation.str.upper() == abbreviation.upper()].copy()
    recent_minutes, starters, rotation_game_count = _rotation_observations(rotations, abbreviation)
    minutes = _expected_minutes(
        group, float(team.games), recent_minutes,
        float(settings["blending"].get("rotation_minutes_weight", 0.65)) if recent_minutes else 0.0,
    )
    group["normalized_player_id"] = group.player_id.map(_id)
    group = group.loc[group.normalized_player_id.isin(minutes)].copy()
    priors = settings["league_priors"]
    zone_priors = priors["zones"]
    prior_shares = {zone: float(zone_priors[zone]["attempt_share"]) for zone in ZONES}
    shooting_prior = float(settings["blending"]["shooting_prior_attempts"])
    zone_share_prior = float(settings["blending"]["zone_prior_attempts"])

    team_shots = pd.DataFrame()
    if shots is not None and not shots.empty:
        if "team_abbreviation" in shots:
            team_shots = shots.loc[shots.team_abbreviation.astype(str).str.upper() == abbreviation.upper()].copy()
        elif "team_id" in shots:
            team_shots = shots.loc[shots.team_id.map(_id) == _id(team.team_id)].copy()
        if not team_shots.empty:
            team_shots["model_zone"] = team_shots.apply(_zone_from_shot, axis=1)
            team_shots["normalized_player_id"] = team_shots.player_id.map(_id)

    team_pbp = pd.DataFrame()
    if play_by_play is not None and not play_by_play.empty and "team_tricode" in play_by_play:
        team_pbp = play_by_play.loc[play_by_play.team_tricode.astype(str).str.upper() == abbreviation.upper()].copy()
        if "is_field_goal" in team_pbp:
            team_pbp = team_pbp.loc[pd.to_numeric(team_pbp.is_field_goal, errors="coerce").fillna(0) == 1].copy()
        if not team_pbp.empty:
            team_pbp["normalized_player_id"] = team_pbp.person_id.map(_id)
    pbp_games = int(team_pbp.game_id.nunique()) if not team_pbp.empty and "game_id" in team_pbp else 0
    pbp_shot_counts = team_pbp.normalized_player_id.value_counts().to_dict() if not team_pbp.empty else {}

    name_ids: dict[str, list[str]] = {}
    for row in group.itertuples():
        last_name = str(row.player_name).split()[-1].lower()
        name_ids.setdefault(last_name, []).append(_id(row.player_id))
    assister_counts: dict[str, dict[str, float]] = {}
    if not team_pbp.empty and "description" in team_pbp:
        for event in team_pbp.itertuples():
            if str(getattr(event, "shot_result", "")).lower() != "made":
                continue
            match = re.search(r"\(([^()]+?)\s+\d+\s+AST\)", str(event.description), re.IGNORECASE)
            if not match:
                continue
            candidates = name_ids.get(match.group(1).strip().lower(), [])
            if len(candidates) == 1:
                shooter_id = _id(event.person_id)
                passer_id = candidates[0]
                assister_counts.setdefault(shooter_id, {})[passer_id] = assister_counts.setdefault(shooter_id, {}).get(passer_id, 0.0) + 1.0
    total_possessions = max(1.0, float(team.fga) + 0.44 * float(team.fta) + float(team.turnovers) - float(team.oreb))
    team_three_share = min(0.65, max(0.05, _safe(float(team.fg3a), float(team.fga), 0.38)))
    team_zone_shares = {
        "rim": (1 - team_three_share) * 0.51,
        "paint": (1 - team_three_share) * 0.29,
        "midrange": (1 - team_three_share) * 0.20,
        "corner_three": team_three_share * 0.22,
        "above_break_three": team_three_share * 0.78,
    }
    team_two_attempts = max(0.0, float(team.fga) - float(team.fg3a))
    team_two_makes = max(0.0, float(team.fgm) - float(team.fg3m))
    team_two_pct = _safe(team_two_makes, team_two_attempts, 0.54)
    team_three_pct = _safe(float(team.fg3m), float(team.fg3a), 0.36)
    team_zone_pct = {
        zone: shrink_rate(
            (team_three_pct if "three" in zone else team_two_pct) * float(team.fg3a if "three" in zone else team_two_attempts) * team_zone_shares[zone],
            float(team.fg3a if "three" in zone else team_two_attempts) * team_zone_shares[zone],
            float(zone_priors[zone]["make_pct"]),
            shooting_prior,
        )
        for zone in ZONES
    }
    if not team_shots.empty:
        team_counts = team_shots.model_zone.value_counts().to_dict()
        team_total_shots = float(len(team_shots))
        team_zone_shares = {
            zone: (float(team_counts.get(zone, 0)) + zone_share_prior * prior_shares[zone]) / (team_total_shots + zone_share_prior)
            for zone in ZONES
        }
        for zone in ZONES:
            zone_rows = team_shots.loc[team_shots.model_zone == zone]
            attempts = float(len(zone_rows))
            makes = float(pd.to_numeric(zone_rows.shot_made_flag, errors="coerce").fillna(0).sum())
            team_zone_pct[zone] = shrink_rate(makes, attempts, float(zone_priors[zone]["make_pct"]), shooting_prior)
    profiles: list[PlayerProfile] = []
    for row in group.to_dict("records"):
        player_id = _id(row["player_id"])
        fga = float(row["fga"])
        fg3a = min(fga, float(row["fg3a"]))
        fg3m = min(fg3a, float(row["fg3m"]))
        two_a = max(0.0, fga - fg3a)
        two_m = max(0.0, float(row["fgm"]) - fg3m)
        three_share = _safe(fg3a, fga, team_three_share)
        observed_shares = {
            "rim": (1 - three_share) * 0.51,
            "paint": (1 - three_share) * 0.29,
            "midrange": (1 - three_share) * 0.20,
            "corner_three": three_share * 0.22,
            "above_break_three": three_share * 0.78,
        }
        for zone in ZONES:
            optional = row.get(f"{zone}_fga")
            if optional is not None and pd.notna(optional):
                observed_shares[zone] = float(optional) / max(1.0, fga)
        zone_shares = _normalize(observed_shares, prior_shares)
        player_shots = team_shots.loc[team_shots.normalized_player_id == player_id] if not team_shots.empty else pd.DataFrame()
        if not player_shots.empty:
            shot_counts = player_shots.model_zone.value_counts().to_dict()
            shot_total = float(len(player_shots))
            zone_shares = {
                zone: (float(shot_counts.get(zone, 0)) + zone_share_prior * zone_shares[zone]) / (shot_total + zone_share_prior)
                for zone in ZONES
            }
        zone_make_pct: dict[str, float] = {}
        sample_sizes: dict[str, float] = {"games": float(row["games"]), "fga": fga}
        for zone in ZONES:
            actual_zone_rows = player_shots.loc[player_shots.model_zone == zone] if not player_shots.empty else pd.DataFrame()
            optional_a = row.get(f"{zone}_fga")
            optional_m = row.get(f"{zone}_fgm")
            if not actual_zone_rows.empty:
                attempts = float(len(actual_zone_rows))
                makes = float(pd.to_numeric(actual_zone_rows.shot_made_flag, errors="coerce").fillna(0).sum())
            elif optional_a is not None and optional_m is not None and pd.notna(optional_a) and pd.notna(optional_m):
                attempts, makes = float(optional_a), float(optional_m)
            else:
                attempts = (fg3a if "three" in zone else two_a) * zone_shares[zone]
                base_pct = _safe(fg3m, fg3a, team_three_pct) if "three" in zone else _safe(two_m, two_a, team_two_pct)
                makes = attempts * base_pct
            zone_make_pct[zone] = shrink_rate(makes, attempts, float(zone_priors[zone]["make_pct"]), shooting_prior)
            sample_sizes[f"{zone}_attempts"] = attempts
        possessions = max(1.0, fga + 0.44 * float(row["fta"]) + float(row["turnovers"]))
        usage_weight = possessions / max(1.0, float(row["minutes"]))
        pbp_attempts = float(pbp_shot_counts.get(player_id, 0))
        if pbp_games:
            aggregate_shot_rate = fga / max(1.0, float(row["minutes"]))
            coverage_minutes = pbp_games * max(1.0, minutes[player_id])
            prior_minutes = float(settings["blending"].get("play_by_play_prior_minutes", 100))
            pbp_shot_rate = (pbp_attempts + aggregate_shot_rate * prior_minutes) / (coverage_minutes + prior_minutes)
            usage_weight = pbp_shot_rate * (1.0 + 0.44 * min(1.0, _safe(float(row["fta"]), fga, 0.2))) + _safe(float(row["turnovers"]), float(row["minutes"]), 0.0)
        sample_sizes["play_by_play_games"] = float(pbp_games)
        sample_sizes["play_by_play_shots"] = pbp_attempts
        sample_sizes["shot_chart_attempts"] = float(len(player_shots))
        shot_locations: dict[str, tuple[tuple[float, float, float], ...]] = {}
        if not player_shots.empty and {"loc_x", "loc_y", "shot_distance"} <= set(player_shots.columns):
            for zone in ZONES:
                location_rows = player_shots.loc[player_shots.model_zone == zone, ["loc_x", "loc_y", "shot_distance"]]
                if location_rows.empty:
                    continue
                stride = max(1, len(location_rows) // 300)
                shot_locations[zone] = tuple(
                    (float(item.loc_x), float(item.loc_y), float(item.shot_distance))
                    for item in location_rows.iloc[::stride].itertuples()
                )
        profiles.append(PlayerProfile(
            player_id=player_id,
            name=str(row["player_name"]),
            team=abbreviation.upper(),
            games=int(row["games"]),
            minutes= float(row["minutes"]),
            expected_minutes=minutes[player_id],
            usage_weight=max(0.01, usage_weight),
            turnover_rate=min(0.35, _safe(float(row["turnovers"]), possessions, 0.12)),
            assist_rate=max(0.001, _safe(float(row["assists"]), float(row["minutes"]), 0.0)),
            oreb_rate=max(0.001, _safe(float(row["oreb"]), float(row["minutes"]), 0.0)),
            dreb_rate=max(0.001, _safe(float(row["dreb"]), float(row["minutes"]), 0.0)),
            ft_rate=min(1.0, _safe(float(row["fta"]), fga, 0.2)),
            ft_pct=shrink_rate(float(row["ftm"]), float(row["fta"]), float(priors["ft_pct"]), 40.0),
            zone_shares=zone_shares,
            zone_make_pct=zone_make_pct,
            sample_sizes=sample_sizes,
            reliability="observed aggregate with shrinkage" + ("; PBP and shot-chart enriched" if pbp_games or not player_shots.empty else ""),
            starter=player_id in starters,
            shooter_selection_source=f"PlayByPlayV3 ({pbp_games} games) with aggregate shrinkage" if pbp_games else "aggregate rates",
            shot_location_source=f"ShotChartDetail ({len(player_shots)} attempts)" if not player_shots.empty else "inferred zones",
            assister_weights=assister_counts.get(player_id, {}),
            shot_locations=shot_locations,
        ))
    pace = float(team.get("pace", np.nan))
    if not np.isfinite(pace):
        pace = total_possessions / max(1.0, float(team.games))
    return TeamProfile(
        team_id=str(team.team_id), abbreviation=abbreviation.upper(), name=str(team.team_name), season=str(team.season),
        games=int(team.games), pace=float(np.clip(pace, 85, 115)),
        turnover_rate=float(np.clip(float(team.turnovers) / total_possessions, 0.06, 0.24)),
        shooting_foul_rate=float(np.clip(float(team.fta) / total_possessions / 2.0, 0.04, 0.20)),
        offensive_rebound_rate=float(np.clip(float(team.oreb) / max(1.0, float(team.oreb) + float(team.dreb)), 0.15, 0.38)),
        assist_rate=float(np.clip(_safe(float(team.assists), float(team.fgm), float(priors["assist_rate"])), 0.35, 0.82)),
        zone_shares=team_zone_shares, zone_make_pct=team_zone_pct,
        defensive_zone_factor={zone: 1.0 for zone in ZONES}, players=tuple(profiles),
        data_evidence={
            "play_by_play_games": pbp_games,
            "play_by_play_shots": int(len(team_pbp)),
            "shot_chart_attempts": int(len(team_shots)),
            "rotation_games": rotation_game_count,
            "verified_starters": len(starters),
            "defender_identity": "unavailable",
        },
    )
