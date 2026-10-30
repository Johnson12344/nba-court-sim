from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import nba_sim.cli as cli
from nba_sim.models import SimulatedGame
from nba_sim.reporting.summary import create_summary


def setup_workflow(monkeypatch, settings, *, fresh=False):
    now = datetime.now(timezone.utc)
    timestamp = (now - timedelta(minutes=5 if fresh else 500)).isoformat()
    calls = []

    class Provider:
        def load(self, season, refresh=False):
            if refresh:
                calls.append(("data", season))
            return SimpleNamespace(metadata=SimpleNamespace(retrieved_at=timestamp))

        def load_enrichment(self, *args):
            return SimpleNamespace(metadata={"available": True, "refreshed_at": timestamp,
                                             "games_per_team": 10, "rotation_games": 3})

        def refresh_enrichment(self, *args):
            calls.append(("enrichment", args[0], args[1]))

    event = {"id": "event-one", "home_team": "Oklahoma City Thunder", "away_team": "San Antonio Spurs",
             "commence_time": (now + timedelta(hours=2)).isoformat(), "player_prop_bookmakers": []}
    cache = {"retrieved_at": timestamp, "events": [event], "regions": "us", "bookmakers_filter": None}
    monkeypatch.setattr(cli, "_settings", lambda args: settings)
    monkeypatch.setattr(cli, "_provider", lambda *args: Provider())
    monkeypatch.setattr(cli, "load_odds_cache", lambda *args: cache)
    monkeypatch.setattr(cli, "refresh_odds_command", lambda args: calls.append(("odds", args.include_player_props)))
    monkeypatch.setattr(cli, "simulate_command", lambda args: calls.append(("simulate", args)) or 0)
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-placeholder")
    args = cli.build_parser().parse_args(["analyze", "--home", "OKC", "--away", "SAS", "--season", "2026-27"])
    return args, calls


def test_one_command_refreshes_then_runs_with_props(monkeypatch, settings):
    args, calls = setup_workflow(monkeypatch, settings)
    assert cli.analyze_command(args) == 0
    assert [call[0] for call in calls] == ["data", "enrichment", "odds", "simulate"]
    assert calls[0][1] == "2026-27"  # Existing completed baseline is reused.
    assert calls[2][1] is True
    assert calls[-1][1].home == "OKC"


def test_recent_inputs_are_reused_and_refresh_can_override(monkeypatch, settings):
    args, calls = setup_workflow(monkeypatch, settings, fresh=True)
    cli.analyze_command(args)
    assert [call[0] for call in calls] == ["simulate"]
    calls.clear()
    args.refresh = True
    args.no_player_props = True
    cli.analyze_command(args)
    assert [call[0] for call in calls] == ["data", "enrichment", "odds", "simulate"]
    assert calls[2][1] is False


@pytest.mark.parametrize("option", ["cached", "demo"])
def test_cached_and_demo_modes_make_no_downloads(monkeypatch, settings, option):
    args, calls = setup_workflow(monkeypatch, settings)
    if option == "cached":
        args.cached = True
    else:
        args.mode = "demo"
    cli.analyze_command(args)
    assert [call[0] for call in calls] == ["simulate"]


def test_missing_key_still_allows_statistics_report(monkeypatch, settings):
    args, calls = setup_workflow(monkeypatch, settings)
    monkeypatch.delenv("THE_ODDS_API_KEY")
    cli.analyze_command(args)
    assert [call[0] for call in calls] == ["data", "enrichment", "simulate"]


def test_unavailable_fixture_still_shows_projections_without_betting_comparison(monkeypatch, settings):
    args, calls = setup_workflow(monkeypatch, settings)
    def missing(args):
        raise cli.UpcomingEventUnavailable("No upcoming event")
    monkeypatch.setattr(cli, "refresh_odds_command", missing)
    cli.analyze_command(args)
    assert calls[-1][0] == "simulate"
    assert calls[-1][1].no_odds
    assert "No upcoming" in calls[-1][1].input_notes[0]


def test_invalid_run_does_not_spend_credits(monkeypatch, settings):
    args, calls = setup_workflow(monkeypatch, settings)
    args.simulations = 0
    with pytest.raises(ValueError, match="Simulation count"):
        cli.analyze_command(args)
    assert calls == []


def test_refresh_failure_does_not_silently_claim_fresh_inputs(monkeypatch, settings):
    args, calls = setup_workflow(monkeypatch, settings)
    def failure(*args):
        raise RuntimeError("NBA.com unavailable")
    monkeypatch.setattr(cli, "_provider", failure)
    with pytest.raises(RuntimeError, match="unavailable"):
        cli.analyze_command(args)
    assert calls == []


def test_total_distribution_uses_joint_scores(demo_profiles):
    games = []
    for index, (home_score, away_score) in enumerate([(10, 30), (30, 10)]):
        quarters = [{"period": p, "home": home_score if p == 1 else 0, "away": away_score if p == 1 else 0} for p in range(1, 5)]
        games.append(SimulatedGame(index, index, "OKC", "SAS", home_score, away_score, quarters, {}, [], 0))
    summary, _, _ = create_summary(games, *demo_profiles)
    assert summary["game_total"]["mean"] == 40
    assert summary["game_total"]["p10"] == 40
    assert summary["game_total"]["p90"] == 40
    assert summary["home_margin"]["mean"] == 0
