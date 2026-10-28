from __future__ import annotations

from typing import Any

from nba_sim.models import SimulatedGame


def representative_narrative(game: SimulatedGame) -> dict[str, Any]:
    leaders: dict[str, list[dict[str, Any]]] = {}
    for team, boxes in game.player_stats.items():
        leaders[team] = [
            {"player": box.name, "points": box.points, "rebounds": box.oreb + box.dreb, "assists": box.assists}
            for box in sorted(boxes.values(), key=lambda item: (item.points, item.assists), reverse=True)[:3]
        ]
    lead_changes = 0
    previous_leader = 0
    for event in game.events:
        margin = int(event["home_score"]) - int(event["away_score"])
        leader = 1 if margin > 0 else -1 if margin < 0 else 0
        if leader and previous_leader and leader != previous_leader:
            lead_changes += 1
        if leader:
            previous_leader = leader
    return {
        "selection_rule": "minimum standardized distance from the median total score and median home margin",
        "disclaimer": "This is one generated simulation, not an actual game or guaranteed outcome.",
        "final_score": {game.home: game.home_score, game.away: game.away_score},
        "quarter_scores": game.quarter_scores,
        "overtimes": game.overtimes,
        "lead_changes_in_generated_events": lead_changes,
        "leaders": leaders,
        "events": game.events,
    }

