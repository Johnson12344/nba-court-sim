from __future__ import annotations


def current_season_weight(games_played: float, prior_games: float = 20.0) -> float:
    """Empirical-Bayes sample-size weight for current-season evidence."""
    if games_played < 0 or prior_games <= 0:
        raise ValueError("games_played must be non-negative and prior_games positive")
    return games_played / (games_played + prior_games)


def recency_weights(number_of_seasons: int, decay: float) -> list[float]:
    if number_of_seasons < 1 or not 0 < decay <= 1:
        raise ValueError("Invalid season count or decay")
    raw = [decay**index for index in range(number_of_seasons)]
    total = sum(raw)
    return [value / total for value in raw]


def shrink_rate(successes: float, attempts: float, prior_rate: float, prior_attempts: float) -> float:
    if attempts < 0 or prior_attempts < 0 or not 0 <= prior_rate <= 1:
        raise ValueError("Invalid shrinkage inputs")
    if successes < 0 or successes > attempts:
        raise ValueError("Successes must be within attempts")
    denominator = attempts + prior_attempts
    return prior_rate if denominator == 0 else (successes + prior_rate * prior_attempts) / denominator

