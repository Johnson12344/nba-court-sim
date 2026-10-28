from __future__ import annotations

import numpy as np

from nba_sim.models import PlayerBoxScore, PlayerProfile, TeamProfile


def sample_minute_targets(team: TeamProfile, rng: np.random.Generator, uncertainty: float) -> dict[str, float]:
    expected = np.array([player.expected_minutes for player in team.players], dtype=float)
    noisy = expected * rng.lognormal(mean=-0.5 * uncertainty**2, sigma=uncertainty, size=len(expected))
    noisy = np.maximum(noisy, 0.25)
    # Capped proportional allocation (water filling) guarantees 240 minutes.
    allocated = np.zeros_like(noisy)
    remaining = np.ones(len(noisy), dtype=bool)
    target = 240.0
    while remaining.any() and target > 1e-9:
        weights = noisy[remaining]
        shares = target * weights / weights.sum()
        indices = np.flatnonzero(remaining)
        capped = shares >= 48.0
        if not capped.any():
            allocated[indices] = shares
            target = 0.0
            break
        for index in indices[capped]:
            allocated[index] = 48.0
            remaining[index] = False
            target -= 48.0
    return {player.player_id: float(value) for player, value in zip(team.players, allocated)}


def select_lineup(
    team: TeamProfile,
    boxes: dict[str, PlayerBoxScore],
    targets: dict[str, float],
    elapsed_game_minutes: float,
    total_scheduled_minutes: float,
) -> tuple[PlayerProfile, ...]:
    remaining_game = max(0.01, total_scheduled_minutes - elapsed_game_minutes)
    ranked: list[tuple[float, PlayerProfile]] = []
    for player in team.players:
        remaining_player = max(0.0, targets[player.player_id] - boxes[player.player_id].minutes)
        urgency = remaining_player / remaining_game
        # Verified starters receive an opening/second-half bump; urgency drives correction.
        opening_window = elapsed_game_minutes % 24.0 < 6.0
        starter_bonus = 0.08 if player.starter and opening_window else 0.0
        score = urgency + 0.001 * player.expected_minutes + starter_bonus
        ranked.append((score, player))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(player for _, player in ranked[:5])


def credit_minutes(boxes: dict[str, PlayerBoxScore], lineup: tuple[PlayerProfile, ...], seconds: float) -> None:
    for player in lineup:
        boxes[player.player_id].minutes += seconds / 60.0

