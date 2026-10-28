from __future__ import annotations

import numpy as np

from nba_sim.models import PlayerBoxScore
from nba_sim.simulation.possession import simulate_event


class StubRng:
    def __init__(self):
        self.scalar = iter([0.13, 0.10, 0.10, 0.10, 0.10, 0.10])

    def random(self, size=None):
        if size is not None:
            return np.array([0.1, 0.9])[:size]
        return next(self.scalar)

    def choice(self, count, p=None):
        return 0


def test_free_throw_scoring_and_offensive_rebound(demo_profiles, settings):
    offense, defense = demo_profiles
    offense = type(offense)(**{**offense.__dict__, "turnover_rate": 0.10, "shooting_foul_rate": 0.10, "offensive_rebound_rate": 0.50})
    offense_lineup, defense_lineup = offense.players[:5], defense.players[:5]
    offense_boxes = {p.player_id: PlayerBoxScore(p.player_id, p.name) for p in offense.players}
    defense_boxes = {p.player_id: PlayerBoxScore(p.player_id, p.name) for p in defense.players}
    result = simulate_event(StubRng(), offense, defense, offense_lineup, defense_lineup, offense_boxes, defense_boxes, True, settings)
    shooter = offense_boxes[offense_lineup[0].player_id]
    assert result.event["type"] == "shooting_foul"
    assert (shooter.fta, shooter.ftm, shooter.points) == (2, 1, 1)
    assert result.retains_possession
    assert sum(box.oreb for box in offense_boxes.values()) == 1

