from __future__ import annotations

import json

import pandas as pd

from nba_sim.market.odds_api import build_market_comparison, find_matchup, refresh_nba_odds


def _event():
    return {
        "id": "event-1",
        "commence_time": "2026-10-27T00:00:00Z",
        "home_team": "Oklahoma City Thunder",
        "away_team": "San Antonio Spurs",
        "bookmakers": [
            {
                "key": "book-a",
                "markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Oklahoma City Thunder", "point": -2.5, "price": 1.91},
                        {"name": "San Antonio Spurs", "point": 2.5, "price": 1.91},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 229.5, "price": 1.91},
                        {"name": "Under", "point": 229.5, "price": 1.91},
                    ]},
                    {"key": "h2h", "outcomes": [
                        {"name": "Oklahoma City Thunder", "price": 1.70},
                        {"name": "San Antonio Spurs", "price": 2.20},
                    ]},
                ],
            },
            {
                "key": "book-b",
                "markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "Oklahoma City Thunder", "point": -3.5, "price": 1.91},
                        {"name": "San Antonio Spurs", "point": 3.5, "price": 1.91},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 230.5, "price": 1.91},
                        {"name": "Under", "point": 230.5, "price": 1.91},
                    ]},
                ],
            },
        ],
        "player_prop_bookmakers": [{
            "key": "book-a",
            "markets": [{
                "key": "player_points",
                "outcomes": [
                    {"name": "Over", "description": "Shai Gilgeous-Alexander", "point": 29.5, "price": 1.91},
                    {"name": "Under", "description": "Shai Gilgeous-Alexander", "point": 29.5, "price": 1.91},
                ],
            }],
        }],
    }


def test_matchup_aliases_and_market_comparison():
    event = _event()
    cache = {
        "provider": "The Odds API",
        "retrieved_at": "2026-10-26T12:00:00+00:00",
        "events": [event],
    }
    assert find_matchup(cache["events"], "OKC", "SAS") is event
    summary = {
        "teams": {
            "OKC": {"score": {"mean": 117.4}, "win_rate": 0.525},
            "SAS": {"score": {"mean": 116.2}, "win_rate": 0.475},
        }
    }
    players = pd.DataFrame([{
        "player_name": "Shai Gilgeous-Alexander", "team": "OKC", "stat": "points", "mean": 30.2,
    }])
    result = build_market_comparison(cache, "OKC", "SAS", summary, players)
    assert result is not None
    assert result["home_spread"] == -3.0
    assert result["total"] == 230.0
    assert result["model_minus_market_margin"] == -1.8
    assert result["player_props"][0]["model_minus_line"] == 0.7


def test_refresh_never_writes_api_key(monkeypatch, tmp_path):
    event = _event()

    class Response:
        status_code = 200
        headers = {"x-requests-remaining": "497", "x-requests-used": "3", "x-requests-last": "3"}

        def json(self):
            return [event]

    monkeypatch.setattr("nba_sim.market.odds_api.requests.get", lambda *args, **kwargs: Response())
    cache = refresh_nba_odds(tmp_path / "raw", tmp_path / "processed", "super-secret")
    saved = (tmp_path / "processed" / "odds_nba.json").read_text(encoding="utf-8")
    assert "super-secret" not in saved
    assert json.loads(saved)["events"][0]["id"] == "event-1"
    assert cache["quota"]["core"]["remaining"] == 497
