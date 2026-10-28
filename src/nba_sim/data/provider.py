from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from nba_sim.models import SourceMetadata


@dataclass(frozen=True)
class DataBundle:
    teams: pd.DataFrame
    players: pd.DataFrame
    metadata: SourceMetadata


@dataclass(frozen=True)
class EnrichmentBundle:
    shots: pd.DataFrame
    play_by_play: pd.DataFrame
    rotations: pd.DataFrame
    metadata: dict

    @classmethod
    def empty(cls) -> "EnrichmentBundle":
        return cls(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"available": False})


class DataProvider(ABC):
    @abstractmethod
    def load(self, season: str, refresh: bool = False) -> DataBundle:
        """Return normalized aggregate team and player observations."""

    def load_enrichment(self, season: str, teams: list[str]) -> EnrichmentBundle:
        return EnrichmentBundle.empty()

