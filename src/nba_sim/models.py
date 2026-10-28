from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ZONES = ("rim", "paint", "midrange", "corner_three", "above_break_three")


@dataclass(frozen=True)
class SourceMetadata:
    source: str
    season: str
    retrieved_at: str
    is_demo: bool = False
    coverage: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    @classmethod
    def now(cls, source: str, season: str, is_demo: bool = False, **kwargs: Any) -> "SourceMetadata":
        return cls(source, season, datetime.now(timezone.utc).isoformat(), is_demo, **kwargs)


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    name: str
    team: str
    games: int
    minutes: float
    expected_minutes: float
    usage_weight: float
    turnover_rate: float
    assist_rate: float
    oreb_rate: float
    dreb_rate: float
    ft_rate: float
    ft_pct: float
    zone_shares: dict[str, float]
    zone_make_pct: dict[str, float]
    sample_sizes: dict[str, float] = field(default_factory=dict)
    reliability: str = "estimated"
    starter: bool = False
    shooter_selection_source: str = "aggregate rates"
    shot_location_source: str = "inferred zones"
    assister_weights: dict[str, float] = field(default_factory=dict)
    shot_locations: dict[str, tuple[tuple[float, float, float], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expected_minutes < 0 or self.expected_minutes > 48:
            raise ValueError(f"Invalid expected minutes for {self.name}")
        if self.usage_weight < 0:
            raise ValueError(f"Invalid usage weight for {self.name}")
        for label, value in {"turnover_rate": self.turnover_rate, "ft_pct": self.ft_pct, **self.zone_make_pct}.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between zero and one")


@dataclass(frozen=True)
class TeamProfile:
    team_id: str
    abbreviation: str
    name: str
    season: str
    games: int
    pace: float
    turnover_rate: float
    shooting_foul_rate: float
    offensive_rebound_rate: float
    assist_rate: float
    zone_shares: dict[str, float]
    zone_make_pct: dict[str, float]
    defensive_zone_factor: dict[str, float]
    players: tuple[PlayerProfile, ...]
    data_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in {
            "turnover_rate": self.turnover_rate,
            "shooting_foul_rate": self.shooting_foul_rate,
            "offensive_rebound_rate": self.offensive_rebound_rate,
            "assist_rate": self.assist_rate,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between zero and one")
        if abs(sum(self.zone_shares.values()) - 1.0) > 1e-6:
            raise ValueError("Team zone shares must sum to one")
        if len(self.players) < 5:
            raise ValueError(f"{self.abbreviation} needs at least five available players")


@dataclass
class PlayerBoxScore:
    player_id: str
    name: str
    minutes: float = 0.0
    points: int = 0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    assists: int = 0
    turnovers: int = 0
    fouls: int = 0
    steals: int = 0


@dataclass
class SimulatedGame:
    game_index: int
    seed: int
    home: str
    away: str
    home_score: int
    away_score: int
    quarter_scores: list[dict[str, int]]
    player_stats: dict[str, dict[str, PlayerBoxScore]]
    events: list[dict[str, Any]]
    overtimes: int

    @property
    def total(self) -> int:
        return self.home_score + self.away_score

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score
