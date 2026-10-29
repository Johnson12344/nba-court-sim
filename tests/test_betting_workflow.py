from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nba_sim.features.availability import FIELDS, apply_lineups, write_lineup_template
from nba_sim.market.edges import build_edge_report, return_statistics
from nba_sim.market.events import select_event, upcoming_events
from nba_sim.models import PlayerBoxScore
from nba_sim.simulation.game import simulate_game
from nba_sim.simulation.lineups import sample_minute_targets
from nba_sim.evaluation.forward import evaluate_forward, normalize_boxscore, save_results, settle_offer

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


def event_fixture():
    return {"id": "future-1", "commence_time": (NOW + timedelta(hours=2)).isoformat(),
            "home_team": "Oklahoma City Thunder", "away_team": "San Antonio Spurs",
            "bookmakers": [{"key": "book-a", "last_update": NOW.isoformat(), "markets": [
                {"key": "h2h", "outcomes": [{"name": "Oklahoma City Thunder", "price": 2.0}, {"name": "San Antonio Spurs", "price": 2.0}]},
                {"key": "spreads", "outcomes": [{"name": "Oklahoma City Thunder", "point": -2, "price": 2.0}, {"name": "San Antonio Spurs", "point": 2, "price": 2.0}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 202, "price": 2.0}, {"name": "Under", "point": 202, "price": 2.0}]},
            ]}]}


def samples(profiles, count=1000):
    home, away = profiles
    result = []
    for i in range(count):
        points = (26, 25, 24, 26)[i % 4]
        player = home.players[0]
        result.append(SimpleNamespace(
            margin=(3, 2, 1, -3)[i % 4], total=(203, 202, 201, 203)[i % 4],
            player_stats={home.abbreviation: {player.player_id: PlayerBoxScore(player.player_id, player.name, points=points)}}))
    return result


def report(profiles, event=None, **kwargs):
    event = event or event_fixture()
    return build_edge_report({"retrieved_at": NOW.isoformat()}, event, samples(profiles), *profiles,
                             readiness_reasons=kwargs.pop("readiness_reasons", []), now=NOW, **kwargs)


def test_upcoming_selection_excludes_started_and_reversed_games():
    future = event_fixture()
    past = {**future, "id": "past", "commence_time": (NOW - timedelta(minutes=1)).isoformat()}
    invalid = {**future, "id": "bad", "commence_time": "2030-01-02"}
    cache = {"events": [past, invalid, future]}
    assert upcoming_events(cache, NOW) == [future]
    assert select_event(cache, "OKC", "SAS", now=NOW) is future
    assert select_event(cache, "SAS", "OKC", now=NOW) is None
    with pytest.raises(ValueError, match="Event ID"):
        select_event(cache, "OKC", "SAS", "past", NOW)


def test_push_aware_expected_value_and_fair_price():
    data = return_statistics(np.array([1, 0, -1, 1]), 1.9)
    assert data["win_probability"] == .5
    assert data["push_probability"] == .25
    assert data["loss_probability"] == .25
    assert data["expected_profit_per_unit"] == pytest.approx(.2)
    assert data["conditional_win_probability"] == pytest.approx(2 / 3)
    assert data["fair_decimal_odds"] == 1.5
    # Even all wins have a nonzero simulation uncertainty margin.
    assert return_statistics(np.ones(1000), 2)["mc_lower_profit_per_unit"] < 1


def test_spread_direction_push_and_whole_total(demo_profiles):
    rows = report(demo_profiles)["offers"]
    home = next(r for r in rows if r["market"] == "spreads" and r["team"] == "OKC")
    away = next(r for r in rows if r["market"] == "spreads" and r["team"] == "SAS")
    assert (home["win_probability"], home["push_probability"], home["loss_probability"]) == (.25, .25, .5)
    assert away["win_probability"] == .5
    assert home["market_no_vig_probability"] == .5
    total = next(r for r in rows if r["market"] == "totals" and r["selection"] == "Over")
    assert total["expected_profit_per_unit"] == .25
    assert total["review_candidate"]


@pytest.mark.parametrize("change", ["stale", "missing", "future", "started", "unreviewed"])
def test_incomplete_or_stale_inputs_never_become_candidates(demo_profiles, change):
    event = event_fixture()
    blockers = []
    if change == "stale":
        event["bookmakers"][0]["last_update"] = (NOW - timedelta(hours=2)).isoformat()
    elif change == "missing":
        del event["bookmakers"][0]["last_update"]
    elif change == "future":
        event["bookmakers"][0]["last_update"] = (NOW + timedelta(minutes=1)).isoformat()
    elif change == "started":
        event["commence_time"] = NOW.isoformat()
    else:
        blockers = ["Availability unresolved"]
    rows = report(demo_profiles, event, readiness_reasons=blockers)["offers"]
    assert rows
    assert not any(r["review_candidate"] for r in rows)
    assert all(r["blockers"] for r in rows)


def test_player_props_use_samples_and_keep_book_specific_prices(demo_profiles):
    event = event_fixture()
    player = demo_profiles[0].players[0]
    market = {"key": "player_points", "outcomes": [
        {"name": "Over", "description": player.name, "point": 25, "price": 2.0},
        {"name": "Under", "description": player.name, "point": 25, "price": 1.8}]}
    event["player_prop_bookmakers"] = [
        {"key": "book-a", "last_update": NOW.isoformat(), "markets": [market]},
        {"key": "book-b", "last_update": NOW.isoformat(), "markets": [deepcopy(market)]}]
    event["player_prop_bookmakers"][1]["markets"][0]["outcomes"][0]["price"] = 2.2
    rows = [r for r in report(demo_profiles, event)["offers"] if r["market"] == "player_points" and r["selection"] == "Over"]
    assert len(rows) == 2
    assert all(r["win_probability"] == .5 and r["push_probability"] == .25 for r in rows)
    assert rows[0]["decimal_odds"] == 2.2
    assert rows[0]["expected_profit_per_unit"] == pytest.approx(.35)
    assert rows[0]["market_no_vig_probability"] == pytest.approx((1 / 2.2) / (1 / 2.2 + 1 / 1.8))
    # Missing player simulation samples are never treated as losing/zero outcomes.
    event["player_prop_bookmakers"][0]["markets"][0]["outcomes"][0]["description"] = "Unknown Player"
    assert any("Unknown Player" in s for s in report(demo_profiles, event)["skipped"])


def test_no_vig_does_not_pair_different_lines(demo_profiles):
    event = event_fixture()
    event["bookmakers"][0]["markets"][2]["outcomes"][1]["point"] = 203
    rows = [r for r in report(demo_profiles, event)["offers"] if r["market"] == "totals"]
    assert all(r["market_no_vig_probability"] is None for r in rows)


def test_ambiguous_player_names_and_three_way_markets_are_skipped(demo_profiles):
    from dataclasses import replace
    home, away = demo_profiles
    away = replace(away, players=(replace(away.players[0], name=home.players[0].name), *away.players[1:]))
    event = event_fixture()
    event["bookmakers"][0]["markets"][0]["outcomes"].append({"name": "Draw", "price": 15})
    event["player_prop_bookmakers"] = [{"key": "book", "last_update": NOW.isoformat(), "markets": [{
        "key": "player_points", "outcomes": [{"name": "Over", "point": 20.5, "price": 2, "description": home.players[0].name}]}]}]
    rows = report((home, away), event)
    assert not any(r["market"] in {"h2h", "player_points"} for r in rows["offers"])
    assert any("ambiguous" in r for r in rows["skipped"])


def write_review(tmp_path, profiles):
    path = tmp_path / "lineups.csv"
    write_lineup_template(path, profiles, "future-1")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(status="active", reviewed_at=NOW.isoformat(), source="Test verified source")
    return path, rows


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_out_players_removed_fixed_targets_survive_simulation(tmp_path, demo_profiles, settings):
    path, rows = write_review(tmp_path, demo_profiles)
    excluded = rows[0]["player_id"]
    fixed_id = rows[1]["player_id"]
    rows[0]["status"] = "out"
    rows[1]["expected_minutes"] = "20"
    save_rows(path, rows)
    teams, review = apply_lineups(path, demo_profiles, "future-1", NOW)
    assert review["reviewed"]
    assert excluded not in {p.player_id for p in teams[0].players}
    assert sum(p.expected_minutes for p in teams[0].players) == pytest.approx(240)
    for seed in range(5):
        targets = sample_minute_targets(teams[0], np.random.default_rng(seed), .2)
        assert targets[fixed_id] == 20
        assert sum(targets.values()) == pytest.approx(240)
    game = simulate_game(*teams, 42, settings)
    assert excluded not in game.player_stats[teams[0].abbreviation]


@pytest.mark.parametrize("change", ["wrong_event", "missing_player", "duplicate", "bad_minutes", "overallocated"])
def test_invalid_lineups_fail_closed(tmp_path, demo_profiles, change):
    path, rows = write_review(tmp_path, demo_profiles)
    if change == "wrong_event":
        rows[0]["event_id"] = "another"
    elif change == "missing_player":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "bad_minutes":
        rows[0]["expected_minutes"] = "nan"
    else:
        for row in rows:
            row["expected_minutes"] = "48"
    save_rows(path, rows)
    with pytest.raises(ValueError):
        apply_lineups(path, demo_profiles, "future-1", NOW)


def test_questionable_and_stale_reviews_are_visible(tmp_path, demo_profiles):
    path, rows = write_review(tmp_path, demo_profiles)
    rows[0]["status"] = "questionable"
    rows[1]["reviewed_at"] = (NOW - timedelta(days=1)).isoformat()
    save_rows(path, rows)
    _, review = apply_lineups(path, demo_profiles, "future-1", NOW)
    assert not review["reviewed"]
    assert len(review["reasons"]) == 2


def test_settlement_push_loss_and_dnp():
    result = {"home_team": "OKC", "away_team": "SAS", "home_score": 102, "away_score": 100,
              "players": {"OKC:1": {"played": False, "points": 0}}}
    offer = {"market": "spreads", "team": "OKC", "line": -2, "decimal_odds": 1.9}
    assert settle_offer(offer, result) == ("push", 0)
    assert settle_offer({**offer, "line": -3}, result) == ("loss", -1)
    prop = {"market": "player_points", "team": "OKC", "player_id": "1", "line": 10.5, "selection": "Under", "decimal_odds": 1.9}
    assert settle_offer(prop, result) == ("void_dnp", 0)
    assert settle_offer({**prop, "player_id": "2"}, result) == ("pending_player_result", None)


def test_boxscore_must_be_final_and_match_fixture():
    metadata = {"home_team": "OKC", "away_team": "SAS", "market_data": event_fixture()}
    metadata["market_data"]["event_id"] = "future-1"
    with pytest.raises(ValueError, match="not final"):
        normalize_boxscore({"game": {"gameStatus": 2}}, metadata)
    payload = {"game": {"gameStatus": 3, "gameId": "0020000001", "gameTimeUTC": event_fixture()["commence_time"],
                         "homeTeam": {"teamTricode": "BOS"}, "awayTeam": {"teamTricode": "SAS"}}}
    with pytest.raises(ValueError, match="teams"):
        normalize_boxscore(payload, metadata)


def test_final_boxscore_records_players_and_rejects_wrong_date():
    metadata = {"home_team": "OKC", "away_team": "SAS", "market_data": {"event_id": "future-1", "commence_time": event_fixture()["commence_time"]}}
    def team(code, score, person):
        return {"teamTricode": code, "score": score, "players": [{"personId": person, "played": "1",
                "statistics": {"points": score, "reboundsTotal": 5, "assists": 2}}]}
    payload = {"game": {"gameId": "0020000001", "gameStatus": 3, "gameTimeUTC": event_fixture()["commence_time"],
                         "homeTeam": team("OKC", 100, 1), "awayTeam": team("SAS", 99, 2)}}
    result = normalize_boxscore(payload, metadata)
    assert result["players"]["OKC:1"]["points"] == 100
    incomplete = deepcopy(payload)
    del incomplete["game"]["homeTeam"]["players"][0]["statistics"]["assists"]
    with pytest.raises(ValueError, match="Incomplete"):
        normalize_boxscore(incomplete, metadata)
    payload["game"]["gameTimeUTC"] = (NOW + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="time"):
        normalize_boxscore(payload, metadata)


def test_forward_evaluation_deduplicates_and_rejects_postgame(tmp_path, demo_profiles):
    event = event_fixture()
    edge = report(demo_profiles)
    meta = {"home_team": "OKC", "away_team": "SAS", "is_demo": False, "model_version": "test", "model": "possession",
            "market_data": {"event_id": event["id"], "commence_time": event["commence_time"]}}
    summary = {"teams": {"OKC": {"win_rate": .75, "score": {"mean": 105, "p10": 90, "p90": 120}},
                         "SAS": {"win_rate": .25, "score": {"mean": 100, "p10": 85, "p90": 115}}}}
    for name, when in [("first", NOW), ("repeat", NOW + timedelta(minutes=5)), ("postgame", NOW + timedelta(hours=3))]:
        folder = tmp_path / "outputs" / name
        folder.mkdir(parents=True)
        (folder / "run_metadata.json").write_text(json.dumps({**meta, "created_at": when.isoformat()}))
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "betting_analysis.json").write_text(json.dumps(edge))
    result = {"event_id": event["id"], "commence_time": event["commence_time"], "completed": True,
              "home_team": "OKC", "away_team": "SAS", "home_score": 102, "away_score": 100}
    save_results(tmp_path / "data", [result])
    report_one, target = evaluate_forward(tmp_path / "outputs", tmp_path / "data" / "results_nba.json")
    group = report_one["groups"][0]
    assert group["unique_games"] == 1
    assert group["brier_score"] == .0625
    assert group["team_score_mae"] == 1.5
    assert any("postgame" in s for s in report_one["skipped"])
    offers = pd.read_csv(target / "offers.csv")
    assert set(offers.run) == {"first"}
    assert offers.paper_selected.sum() <= 3
    report_two, _ = evaluate_forward(tmp_path / "outputs", tmp_path / "data" / "results_nba.json")
    assert report_two == report_one


def test_cli_lineup_to_export_workflow(tmp_path, monkeypatch):
    import nba_sim.cli as cli
    from nba_sim.data.demo_provider import DemoProvider
    from nba_sim.config import load_settings

    monkeypatch.setattr(cli, "_provider", lambda *args: DemoProvider())
    original_project_path = cli.project_path
    monkeypatch.setattr(cli, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(cli, "_settings", lambda args: load_settings(original_project_path("config", "settings.yaml")))
    event = event_fixture()
    current = datetime.now(timezone.utc)
    event["commence_time"] = (current + timedelta(hours=2)).isoformat()
    event["bookmakers"][0]["last_update"] = current.isoformat()
    cache = {"retrieved_at": current.isoformat(), "events": [event]}
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "odds_nba.json").write_text(json.dumps(cache))
    path = tmp_path / "lineups.csv"
    parser = cli.build_parser()
    common = ["simulate", "--home", "OKC", "--away", "SAS", "--season", "2026-27", "--event-id", event["id"]]
    assert cli.simulate_command(parser.parse_args([*common, "--write-lineups", str(path)])) == 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(status="active", reviewed_at=current.isoformat(), source="test source")
    excluded = rows[0]["player_id"]
    rows[0]["status"] = "out"
    save_rows(path, rows)
    assert cli.simulate_command(parser.parse_args([*common, "--lineups", str(path), "--simulations", "2", "--workers", "1"])) == 0
    run = next((tmp_path / "outputs").iterdir())
    assert (run / "market_probabilities.csv").exists()
    analysis = json.loads((run / "betting_analysis.json").read_text())
    assert len(analysis["offers"]) == 6
    assert not any(row["review_candidate"] for row in analysis["offers"])
    players = pd.read_csv(run / "player_distributions.csv")
    assert excluded not in set(players.player_id)
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["availability"]["reviewed"]
    assert metadata["market_data"]["event_id"] == event["id"]


def test_result_cache_preserves_players_unless_scores_are_corrected(tmp_path):
    base = {"event_id": "one", "home_score": 100, "away_score": 99}
    save_results(tmp_path, [{**base, "players": {"OKC:1": {"played": True, "points": 100}}}])
    assert save_results(tmp_path, [base])["events"]["one"]["players"]
    corrected = save_results(tmp_path, [{**base, "home_score": 101}])
    assert "players" not in corrected["events"]["one"]


def test_refresh_results_only_records_completed_games_without_key(tmp_path, monkeypatch):
    from nba_sim.evaluation.forward import refresh_results

    event = event_fixture()
    event.update(completed=True, scores=[{"name": event["home_team"], "score": "100"}, {"name": event["away_team"], "score": "99"}])
    pending = {**event, "id": "pending", "completed": False}
    def request(path, api_key, params, timeout):
        assert api_key == "secret-for-test"
        assert params["daysFrom"] == "3"
        return [event, pending], {"remaining": 400}
    monkeypatch.setattr("nba_sim.evaluation.forward._request", request)
    count, _ = refresh_results(tmp_path / "raw", tmp_path / "processed", "secret-for-test")
    assert count == 1
    cache = json.loads((tmp_path / "processed" / "results_nba.json").read_text())
    assert set(cache["events"]) == {"future-1"}
    assert "secret-for-test" not in json.dumps(cache)
