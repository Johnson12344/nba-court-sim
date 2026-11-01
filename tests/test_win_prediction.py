from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import StringIO

import pytest
from rich.console import Console

from nba_sim.reporting.console_report import render_win_prediction
from nba_sim.reporting.win_prediction import build_win_prediction, probability_colour

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


def inputs():
    return ({"teams": {"IND": {"name": "Indiana Pacers", "win_rate": .68},
                       "BKN": {"name": "Brooklyn Nets", "win_rate": .32}}},
            {"home_team": "IND", "away_team": "BKN", "is_demo": False})


def learned_inputs():
    summary, metadata = inputs()
    summary["advisor"] = {"models": {"h2h": {"status": "passed", "simulation_weight": .5}},
                          "selection": {"market": "player_points", "estimated_win_probability": .99}}
    summary["betting_analysis"] = {
        "created_at": NOW.isoformat(), "commence_time": (NOW + timedelta(hours=2)).isoformat(),
        "readiness_reasons": [], "offers": [{"market": "h2h", "team": "IND", "bookmaker": "book-a",
            "conditional_win_probability": .68, "win_probability": .68, "loss_probability": .32,
            "push_probability": 0, "market_no_vig_probability": .4, "decimal_odds": 2.4,
            "simulations": 5000, "quote_updated_at": NOW.isoformat(), "blockers": []}]}
    return summary, metadata


def test_simulation_probability_and_rendering_show_both_teams():
    summary, metadata = inputs()
    prediction = build_win_prediction(summary, metadata, now=NOW)
    assert prediction["winner"] == "IND"
    assert prediction["probabilities"] == {"IND": .68, "BKN": .32}
    assert prediction["source"] == "Simulation estimate (not yet calibrated)"
    output = StringIO()
    render_win_prediction(prediction, Console(file=output, force_terminal=True, color_system="standard",
                                              no_color=False, legacy_windows=False, width=120))
    text = output.getvalue()
    assert "Indiana Pacers" in text and "Brooklyn Nets" in text
    assert "68%" in text and "32%" in text
    assert "1;33m68%" in text and "1;31m32%" in text
    assert "not of the exact score" in text


@pytest.mark.parametrize("p,colour", [(.4999, "red"), (.5, "yellow"), (.8, "yellow"), (.8001, "green")])
def test_requested_colour_boundaries(p, colour):
    assert probability_colour(p) == colour


def test_learned_probability_uses_moneyline_not_selected_player_prop():
    summary, metadata = learned_inputs()
    prediction = build_win_prediction(summary, metadata, now=NOW)
    assert prediction["probabilities"]["IND"] == pytest.approx(.54)
    assert prediction["probabilities"]["BKN"] == pytest.approx(.46)
    assert "Learned" in prediction["source"]
    # Learning may change the favourite; the footer must follow the chosen probabilities.
    summary["advisor"]["models"]["h2h"]["simulation_weight"] = 0
    assert build_win_prediction(summary, metadata, now=NOW)["winner"] == "BKN"


@pytest.mark.parametrize("change", ["not_validated", "stale", "injury", "started", "no_quote", "demo"])
def test_unusable_learning_falls_back_to_labelled_simulation(change):
    summary, metadata = learned_inputs()
    if change == "not_validated":
        summary["advisor"]["models"]["h2h"]["status"] = "learning"
    elif change == "stale":
        summary["betting_analysis"]["offers"][0]["quote_updated_at"] = (NOW - timedelta(hours=2)).isoformat()
    elif change == "injury":
        metadata["availability"] = {"reasons": ["Questionable player"]}
    elif change == "started":
        summary["betting_analysis"]["commence_time"] = NOW.isoformat()
    elif change == "no_quote":
        summary["betting_analysis"]["offers"] = []
    else:
        metadata["is_demo"] = True
    prediction = build_win_prediction(summary, metadata, now=NOW)
    assert prediction["probabilities"]["IND"] == .68
    assert "simulation" in prediction["source"].lower()
    assert prediction["is_demo"] == (change == "demo")


def test_equal_probabilities_have_no_invented_favourite():
    summary, metadata = inputs()
    for team in summary["teams"].values():
        team["win_rate"] = .5
    assert build_win_prediction(summary, metadata, now=NOW)["winner"] is None


def test_book_selection_is_stable_not_highest_probability():
    summary, metadata = learned_inputs()
    other = deepcopy(summary["betting_analysis"]["offers"][0])
    other.update(bookmaker="book-z", market_no_vig_probability=.99)
    summary["betting_analysis"]["offers"].insert(0, other)
    result = build_win_prediction(summary, metadata, now=NOW)
    assert result["bookmaker"] == "book-a"
    assert result["probabilities"]["IND"] == pytest.approx(.54)
