from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nba_sim.evaluation.forward import normalize_boxscore, save_results, settle_offer
from nba_sim.market.catalog import EXTENDED_MARKETS
from nba_sim.market.edges import build_edge_report
from nba_sim.market.odds_api import refresh_nba_odds
from nba_sim.models import PlayerBoxScore

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


def actual():
    return {"home_team": "OKC", "away_team": "SAS", "home_score": 120, "away_score": 100,
            "periods": {"home": {"1": 30}, "away": {"1": 25}},
            "players": {"OKC:7": {"played": True, "points": 20, "rebounds": 8, "assists": 6, "threes": 3}}}


@pytest.mark.parametrize("market,line,team", [
    ("totals_q1", 54.5, None), ("spreads_q1", -4.5, "OKC"), ("team_totals", 119.5, "OKC"),
    ("player_threes", 2.5, "OKC"), ("player_points_rebounds", 27.5, "OKC"),
    ("player_points_assists", 25.5, "OKC"), ("player_rebounds_assists", 13.5, "OKC"),
    ("player_points_rebounds_assists", 33.5, "OKC"),
])
def test_new_markets_use_matching_simulation_and_settlement(market, line, team, demo_profiles):
    player = demo_profiles[0].players[0]
    description = (player.name if market.startswith("player_") else
                   "Oklahoma City Thunder" if market == "team_totals" else None)
    side = "Oklahoma City Thunder" if market == "spreads_q1" else "Over"
    opposite = "San Antonio Spurs" if market == "spreads_q1" else "Under"
    outcomes = [{"name": side, "point": line, "price": 1.9},
                {"name": opposite, "point": -line if market == "spreads_q1" else line, "price": 1.9}]
    if description:
        for row in outcomes:
            row["description"] = description
    event = {"id": "one", "home_team": "Oklahoma City Thunder", "away_team": "San Antonio Spurs",
             "commence_time": (NOW + timedelta(hours=2)).isoformat(), "bookmakers": [],
             "player_prop_bookmakers": [{"key": "book", "last_update": NOW.isoformat(),
                                          "markets": [{"key": market, "outcomes": outcomes}]}]}
    box = PlayerBoxScore(player.player_id, player.name, points=20, oreb=3, dreb=5, assists=6, fg3m=3)
    game = SimpleNamespace(margin=20, total=220, quarter_scores=[{"period": 1, "home": 30, "away": 25}],
                           player_stats={"OKC": {player.player_id: box}})
    report = build_edge_report({"retrieved_at": NOW.isoformat()}, event, [game] * 500,
                               *demo_profiles, readiness_reasons=[], now=NOW)
    selected = next(row for row in report["offers"] if row["selection"] == side)
    assert selected["win_probability"] == 1
    assert selected["market_no_vig_probability"] == .5
    result = actual()
    result["players"][f"OKC:{player.player_id}"] = result["players"].pop("OKC:7")
    assert settle_offer(selected, result) == ("win", pytest.approx(.9))
    other = next(row for row in report["offers"] if row["selection"] == opposite)
    assert settle_offer(other, result) == ("loss", -1.)
    if market.endswith("_q1"):
        game.quarter_scores = []
        missing = build_edge_report({"retrieved_at": NOW.isoformat()}, event, [game] * 500,
                                    *demo_profiles, readiness_reasons=[], now=NOW)
        assert missing["offers"] == []
        assert missing["skipped"]


def test_missing_period_or_player_component_stays_pending_and_dnp_voids():
    result = actual()
    offer = {"market": "totals_q1", "selection": "Over", "line": 55, "decimal_odds": 2}
    assert settle_offer(offer, result) == ("push", 0.)
    result.pop("periods")
    assert settle_offer(offer, result) == ("pending_period_result", None)
    offer.update(market="player_points_rebounds_assists", team="OKC", player_id="7", line=34)
    assert settle_offer(offer, result) == ("push", 0.)
    result["players"]["OKC:7"].pop("assists")
    assert settle_offer(offer, result) == ("pending_player_result", None)
    result["players"]["OKC:7"]["played"] = False
    assert settle_offer(offer, result) == ("void_dnp", 0.)


def test_periods_and_threes_are_verified_and_invalidated_on_correction(tmp_path):
    def team(code, points):
        return {"teamTricode": code, "score": points,
                "periods": [{"period": n, "score": points // 4} for n in range(1, 5)],
                "players": [{"personId": 7, "played": "1", "statistics": {
                    "points": points, "reboundsTotal": 8, "assists": 6, "threePointersMade": 3}}]}
    meta = {"home_team": "OKC", "away_team": "SAS",
            "market_data": {"event_id": "one", "commence_time": NOW.isoformat()}}
    payload = {"game": {"gameStatus": 3, "gameId": "0022900001", "gameTimeUTC": NOW.isoformat(),
                        "homeTeam": team("OKC", 120), "awayTeam": team("SAS", 100)}}
    result = normalize_boxscore(payload, meta)
    assert result["periods"]["home"]["1"] == 30
    assert result["players"]["OKC:7"]["threes"] == 3
    saved = save_results(tmp_path, [result])["events"]["one"]
    assert saved["periods_observed_at"]
    unchanged = save_results(tmp_path, [result])["events"]["one"]
    assert unchanged["periods_observed_at"] == saved["periods_observed_at"]
    corrected = save_results(tmp_path, [{"event_id": "one", "home_score": 121, "away_score": 100, "completed": True}])
    assert "periods" not in corrected["events"]["one"]
    payload["game"]["homeTeam"]["periods"][0]["score"] = 31
    with pytest.raises(ValueError, match="Period scores"):
        normalize_boxscore(payload, meta)


@pytest.mark.parametrize("props", [True, False])
def test_extended_odds_request_uses_event_endpoint_and_respects_prop_opt_out(tmp_path, monkeypatch, props):
    calls = []
    now = datetime.now(timezone.utc)
    event = {"id": "one", "home_team": "Oklahoma City Thunder", "away_team": "San Antonio Spurs",
             "commence_time": (now + timedelta(hours=2)).isoformat(), "bookmakers": []}
    def request(path, key, params, timeout):
        calls.append((path, params))
        return (deepcopy(event) if "/events/" in path else [deepcopy(event)]), {"remaining": 100}
    monkeypatch.setattr("nba_sim.market.odds_api._request", request)
    result = refresh_nba_odds(tmp_path / "raw", tmp_path / "processed", "test-key", home="OKC", away="SAS",
                              include_player_props=props, extended_markets=True)
    assert len(calls) == 2 and "/events/one/odds" in calls[1][0]
    requested = set(calls[1][1]["markets"].split(","))
    assert {"totals_q1", "spreads_q1", "team_totals"} <= requested
    assert ("player_points_rebounds_assists" in requested) == props
    assert ("player_assists" in requested) == props
    assert set(result["events"][0]["requested_additional_markets"]) == requested
