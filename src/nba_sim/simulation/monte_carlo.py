from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from nba_sim.models import SimulatedGame, TeamProfile
from nba_sim.simulation.game import simulate_baseline_game, simulate_game

_WORKER_CONTEXT: tuple[TeamProfile, TeamProfile, dict[str, Any], str] | None = None


def _initialize_worker(home: TeamProfile, away: TeamProfile, settings: dict[str, Any], model: str) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (home, away, settings, model)


def _worker_game(item: tuple[int, int]) -> SimulatedGame:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Simulation worker was not initialized")
    home, away, settings, model = _WORKER_CONTEXT
    index, seed = item
    function = simulate_game if model == "possession" else simulate_baseline_game
    return function(home, away, seed, settings, index)


def run_simulations(
    home: TeamProfile,
    away: TeamProfile,
    count: int,
    seed: int,
    settings: dict[str, Any],
    model: str = "possession",
    progress: Callable[[int], None] | None = None,
    workers: int = 1,
) -> tuple[list[SimulatedGame], SimulatedGame]:
    if count < 1:
        raise ValueError("Simulation count must be positive")
    max_count = int(settings.get("simulation", {}).get("max_count", 10000))
    if count > max_count:
        raise ValueError(f"Simulation count is capped at {max_count:,}")
    if model not in {"possession", "baseline"}:
        raise ValueError("model must be 'possession' or 'baseline'")
    master = np.random.default_rng(seed)
    seeds = master.integers(0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32)
    games: list[SimulatedGame] = []
    jobs = [(index, int(child_seed)) for index, child_seed in enumerate(seeds)]
    if workers > 1 and count >= workers * 2:
        chunksize = max(1, count // (workers * 16))
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker,
            initargs=(home, away, settings, model),
        ) as executor:
            for game in executor.map(_worker_game, jobs, chunksize=chunksize):
                games.append(game)
                if progress:
                    progress(len(games))
    else:
        function = simulate_game if model == "possession" else simulate_baseline_game
        for index, child_seed in jobs:
            games.append(function(home, away, child_seed, settings, index))
            if progress:
                progress(index + 1)
    totals = np.array([game.total for game in games], dtype=float)
    margins = np.array([game.margin for game in games], dtype=float)
    median_total = float(np.median(totals))
    median_margin = float(np.median(margins))
    total_scale = max(1.0, float(np.std(totals)))
    margin_scale = max(1.0, float(np.std(margins)))
    representative_index = int(np.argmin(abs(totals - median_total) / total_scale + abs(margins - median_margin) / margin_scale))
    representative_base = games[representative_index]
    representative = simulate_game(home, away, representative_base.seed, settings, representative_index, capture_events=True) if model == "possession" else representative_base
    return games, representative
