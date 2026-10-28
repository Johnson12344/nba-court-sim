from __future__ import annotations

import pandas as pd

from nba_sim.data.provider import DataBundle, DataProvider
from nba_sim.models import SourceMetadata


class DemoProvider(DataProvider):
    """Deterministic fictional inputs for offline software verification only."""

    def load(self, season: str, refresh: bool = False) -> DataBundle:
        teams: list[dict] = []
        players: list[dict] = []
        for team_index, (abbr, name, pace, efficiency) in enumerate((
            ("OKC", "Oklahoma City Thunder (DEMO)", 100.5, 1.03),
            ("SAS", "San Antonio Spurs (DEMO)", 102.0, 0.98),
        )):
            games = 82
            fga = 89.0 * games
            fg3a = (34.0 + team_index * 2) * games
            fgm = fga * (0.48 * efficiency)
            fg3m = fg3a * (0.36 * efficiency)
            fta = 22.0 * games
            teams.append({
                "team_id": f"DEMO-T{team_index+1}", "team_abbreviation": abbr, "team_name": name,
                "season": season, "games": games, "minutes": games * 240, "points": games * (114 + team_index),
                "fga": fga, "fgm": fgm, "fg3a": fg3a, "fg3m": fg3m, "fta": fta, "ftm": fta * 0.78,
                "oreb": games * 10.5, "dreb": games * 33.5, "assists": games * 26,
                "turnovers": games * (12.5 + team_index), "pace": pace,
            })
            minute_pattern = [35, 34, 32, 30, 28, 23, 21, 19, 12, 6]
            usage_pattern = [0.27, 0.22, 0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.07, 0.06]
            for player_index, (mpg, usage) in enumerate(zip(minute_pattern, usage_pattern), 1):
                p_fga = games * mpg / 32 * usage * 65
                p_fg3a = p_fga * (0.28 + 0.025 * (player_index % 4))
                p_fta = p_fga * (0.20 + 0.02 * (player_index % 3))
                p_fgm = p_fga * (0.44 + 0.01 * ((player_index + team_index) % 4))
                p_fg3m = p_fg3a * (0.34 + 0.01 * ((player_index + 1) % 4))
                players.append({
                    "player_id": f"DEMO-{abbr}-{player_index}", "player_name": f"{abbr} Demo Player {player_index}",
                    "team_abbreviation": abbr, "season": season, "games": games, "minutes": games * mpg,
                    "points": 2 * (p_fgm - p_fg3m) + 3 * p_fg3m + p_fta * 0.78,
                    "fga": p_fga, "fgm": p_fgm, "fg3a": p_fg3a, "fg3m": p_fg3m,
                    "fta": p_fta, "ftm": p_fta * (0.72 + 0.015 * (player_index % 6)),
                    "oreb": games * mpg / 48 * (0.8 + 0.15 * (player_index % 3)),
                    "dreb": games * mpg / 48 * (3.0 + 0.4 * (player_index % 4)),
                    "assists": games * mpg / 48 * (2.0 + (5 if player_index == 1 else 0.4 * (player_index % 3))),
                    "turnovers": games * mpg / 48 * (1.0 + usage * 5), "fouls": games * mpg / 48 * 2.5,
                    "expected_minutes": mpg,
                })
        assumptions = (
            "DEMO DATA: all team and player observations are synthetic and must not be interpreted as NBA facts.",
            "Zone shares and defensive effects use configured league priors because demo aggregates contain no shot locations.",
        )
        return DataBundle(pd.DataFrame(teams), pd.DataFrame(players), SourceMetadata.now("synthetic_demo", season, True, coverage={"teams": 2, "players": 20}, assumptions=assumptions))
