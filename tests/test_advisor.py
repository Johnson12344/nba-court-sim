from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from nba_sim.advisor.learning import advise, fit_market, fit_weight
from nba_sim.advisor.memory import load_memory, memory_summary, sync_memory, training_rows
from nba_sim.advisor.results import find_game_id, refresh_memory_results
from nba_sim.evaluation.forward import save_results

NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
VERSION = "0.5.0"


def offer(at=NOW, **changes):
    return {"event_id": "one", "bookmaker": "book-a", "market": "h2h", "team": "OKC",
            "selection": "Oklahoma City Thunder", "player_id": None, "line": None, "decimal_odds": 1.6,
            "conditional_win_probability": .9, "market_no_vig_probability": .5, "win_probability": .9,
            "loss_probability": .1, "push_probability": 0., "simulations": 5000,
            "quote_updated_at": at.isoformat(), "review_candidate": True, "blockers": [],
            "expected_profit_per_unit": .44, "mc_lower_profit_per_unit": .4, **changes}


def edge(at=NOW, **changes):
    return {"event_id": "one", "created_at": at.isoformat(), "commence_time": (at + timedelta(hours=2)).isoformat(),
            "readiness_reasons": [], "offers": [offer(at)], "min_ev": .02, "max_odds_age_minutes": 60, **changes}


def metadata(at=NOW, **changes):
    return {"created_at": at.isoformat(), "home_team": "OKC", "away_team": "SAS", "season": "2029-30",
            "model_version": VERSION, "model": "possession", "is_demo": False,
            "market_data": {"event_id": "one", "commence_time": (at + timedelta(hours=2)).isoformat()}, **changes}


def write_run(root, name, meta=None, analysis=None):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "run_metadata.json").write_text(json.dumps(meta or metadata()))
    (folder / "betting_analysis.json").write_text(json.dumps(analysis or edge()))
    return folder


def learned_memory(count=160):
    predictions = {}
    for i in range(count):
        at = NOW - timedelta(days=200 - i)
        win = i % 4 != 3  # Actual frequency .75; simulation .90; market .50.
        actual = {"event_id": str(i), "home_team": "OKC", "away_team": "SAS", "completed": True,
                  "commence_time": (at + timedelta(hours=2)).isoformat(), "home_score": 110 if win else 90,
                  "away_score": 100, "observed_at": (at + timedelta(hours=5)).isoformat()}
        predictions[str(i)] = {"event_id": str(i), "model_version": VERSION, "model": "possession", "season": "2029-30",
                                "home_team": "OKC", "away_team": "SAS", "created_at": at.isoformat(),
                                "commence_time": actual["commence_time"], "offers": [offer(at)], "result": actual}
    return {"schema_version": 1, "predictions": predictions}


def test_memory_keeps_first_pregame_run_without_counting_books_or_repeated_runs(tmp_path):
    outputs, processed = tmp_path / "outputs", tmp_path / "processed"
    rows = [offer(), offer(bookmaker="book-b"), offer(team="SAS", selection="San Antonio Spurs")]
    write_run(outputs, "first", analysis=edge(offers=rows))
    write_run(outputs, "repeat", metadata(NOW + timedelta(minutes=5)), edge(NOW + timedelta(minutes=5)))
    memory = sync_memory(outputs, processed, now=NOW + timedelta(minutes=10))
    assert len(memory["predictions"]) == 1
    prediction = next(iter(memory["predictions"].values()))
    assert prediction["created_at"] == NOW.isoformat()
    assert len(prediction["offers"]) == 1
    # Persistent memory survives moving/deleting the output archive.
    memory = sync_memory(tmp_path / "empty", processed, now=NOW + timedelta(minutes=20))
    assert len(memory["predictions"]) == 1
    assert load_memory(processed)["predictions"] == memory["predictions"]


@pytest.mark.parametrize("kind", ["demo", "postgame", "future", "event_mismatch", "stale_quote", "blocked", "opt_out"])
def test_untrusted_predictions_do_not_train(tmp_path, kind):
    m, e = metadata(), edge()
    if kind == "demo":
        m["is_demo"] = True
    elif kind == "postgame":
        m["created_at"] = (NOW + timedelta(hours=3)).isoformat()
    elif kind == "future":
        m["created_at"] = (NOW + timedelta(minutes=1)).isoformat()
    elif kind == "event_mismatch":
        e["event_id"] = "wrong"
    elif kind == "stale_quote":
        e["offers"][0]["quote_updated_at"] = (NOW - timedelta(hours=2)).isoformat()
    elif kind == "opt_out":
        m["advisor_enabled"] = False
    else:
        e["readiness_reasons"] = ["Availability unresolved"]
    write_run(tmp_path / "outputs", "run", m, e)
    memory = sync_memory(tmp_path / "outputs", tmp_path / "processed", now=NOW)
    assert not any(p["offers"] for p in memory["predictions"].values())


def test_low_ev_losses_are_retained_for_learning_and_corrections_are_applied(tmp_path):
    outputs, processed = tmp_path / "outputs", tmp_path / "processed"
    row = offer(review_candidate=False, expected_profit_per_unit=-.1)
    write_run(outputs, "run", analysis=edge(offers=[row]))
    processed.mkdir()
    actual = {"event_id": "one", "completed": True, "home_team": "OKC", "away_team": "SAS",
              "home_score": 90, "away_score": 100, "commence_time": edge()["commence_time"],
              "observed_at": (NOW + timedelta(hours=5)).isoformat()}
    path = processed / "results_nba.json"
    path.write_text(json.dumps({"events": {"one": actual}}))
    memory = sync_memory(outputs, processed, now=NOW + timedelta(days=1))
    rows = training_rows(memory, VERSION, "possession", NOW + timedelta(days=1))
    assert len(rows) == 1 and rows[0]["y"] == 0
    actual["home_score"] = 110
    path.write_text(json.dumps({"events": {"one": actual}}))
    memory = sync_memory(outputs, processed, now=NOW + timedelta(days=1))
    assert training_rows(memory, VERSION, "possession", NOW + timedelta(days=1))[0]["y"] == 1
    actual["home_team"] = "BOS"
    path.write_text(json.dumps({"events": {"one": actual}}))
    memory = sync_memory(outputs, processed, now=NOW + timedelta(days=1))
    assert not training_rows(memory, VERSION, "possession", NOW + timedelta(days=1))


def test_pushes_dnp_unresolved_and_future_results_are_not_losses():
    memory = learned_memory(4)
    for prediction in memory["predictions"].values():
        prediction["offers"][0].update(market="player_points", player_id="7", selection="Over", line=20)
        prediction["result"].update(players_observed_at=prediction["result"]["observed_at"])
    memory["predictions"]["0"]["result"]["players"] = {"OKC:7": {"played": True, "points": 20}}
    memory["predictions"]["1"]["result"]["players"] = {"OKC:7": {"played": False, "points": 0}}
    memory["predictions"]["3"]["result"].update(players={"OKC:7": {"played": True, "points": 30}},
                                                 players_observed_at=(NOW + timedelta(days=1)).isoformat())
    assert training_rows(memory, VERSION, "possession", NOW) == []
    assert not training_rows(learned_memory(), "another-version", "possession", NOW)


def test_weight_is_learned_from_outcomes_and_validated_on_later_games():
    rows = training_rows(learned_memory(), VERSION, "possession", NOW)
    result = fit_market(rows)
    assert result["status"] == "passed"
    assert result["simulation_weight"] == pytest.approx(.625)
    assert result["training_games"] == 120 and result["validation_games"] == 40
    assert result["validation_brier"]["learned"] == pytest.approx(.1875)
    # Opposite outcomes teach it to discard this unreliable simulation.
    wrong = [{**r, "y": 0} for r in rows]
    assert fit_weight(wrong) == 0
    assert fit_market(wrong)["status"] == "validation_failed"


def test_future_outcomes_cannot_leak_into_training_or_pad_game_count():
    rows = training_rows(learned_memory(), VERSION, "possession", NOW)
    for row in rows[:120]:
        row["observed_at"] = NOW.isoformat()  # Not known before validation predictions.
    result = fit_market(rows)
    assert result["status"] == "learning" and result["training_games"] == 0
    assert fit_market(rows[:1] * 1000)["games"] == 1


def test_cold_start_returns_no_bet_and_explicit_paper_watch():
    result = advise(edge(), {"predictions": {}}, VERSION, "possession", now=NOW)
    assert result["status"] == "no_bet"
    assert result["selection"] is None
    assert result["paper_watch"]["probability_source"] == "Unvalidated simulation only"


def test_ranked_candidate_uses_learned_probability_not_raw_simulation():
    result = advise(edge(), learned_memory(), VERSION, "possession", now=NOW)
    assert result["status"] == "review_candidate"
    selected = result["selection"]
    assert selected["estimated_win_probability"] == pytest.approx(.75)
    assert selected["adjusted_ev"] == pytest.approx(.2)
    assert selected["estimated_loss_probability"] == pytest.approx(.25)
    assert result["models"]["h2h"]["training_fingerprint"]


@pytest.mark.parametrize("kind", ["injury", "stale", "started", "poor_price", "no_opposite", "model_failed", "too_many_pushes"])
def test_no_bet_when_price_data_or_validation_fails(kind):
    e, memory = edge(), learned_memory()
    if kind == "injury":
        e["readiness_reasons"] = ["Availability unresolved"]
    elif kind == "stale":
        e["offers"][0]["quote_updated_at"] = (NOW - timedelta(hours=2)).isoformat()
    elif kind == "started":
        e["commence_time"] = NOW.isoformat()
    elif kind == "poor_price":
        e["offers"][0]["decimal_odds"] = 1.1
    elif kind == "no_opposite":
        e["offers"][0]["market_no_vig_probability"] = None
    elif kind == "too_many_pushes":
        e["offers"][0].update(win_probability=.45, loss_probability=.05, push_probability=.5)
    else:
        for record in memory["predictions"].values():
            record["result"]["home_score"] = 90
    result = advise(e, memory, VERSION, "possession", now=NOW)
    assert result["status"] == "no_bet" and result["selection"] is None


def test_same_current_event_cannot_train_its_own_recommendation():
    e = edge(event_id="0")
    result = advise(e, learned_memory(140), VERSION, "possession", now=NOW)
    assert result["models"]["h2h"]["games"] == 139
    assert result["status"] == "no_bet"


def test_corrupted_memory_is_preserved(tmp_path):
    path = tmp_path / "advisor_memory.json"
    path.write_text("broken")
    with pytest.raises(ValueError):
        sync_memory(tmp_path / "outputs", tmp_path, now=NOW)
    assert path.read_text() == "broken"
    path.write_text(json.dumps({"schema_version": 1, "predictions": {"bad": {"offers": []}}}))
    before = path.read_text()
    with pytest.raises(ValueError):
        sync_memory(tmp_path / "outputs", tmp_path, now=NOW)
    assert path.read_text() == before


def test_result_collection_matches_eastern_date_and_does_not_need_api_key(tmp_path, monkeypatch):
    from nba_api.live.nba.endpoints import boxscore
    memory = learned_memory(1)
    prediction = memory["predictions"]["0"]
    prediction.pop("result")
    prediction["commence_time"] = "2029-12-31T01:00:00+00:00"
    (tmp_path / "game_log_2029-30.csv").write_text(
        "game_id,game_date,team_abbreviation,matchup\n0022900001,2029-12-30,OKC,OKC vs. SAS\n")
    assert find_game_id(prediction, tmp_path) == "0022900001"
    def team(code, score):
        return {"teamTricode": code, "score": score, "players": [{"personId": 7, "played": "1",
                "statistics": {"points": score, "reboundsTotal": 5, "assists": 2}}]}
    calls = []
    class Box:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def get_dict(self):
            return {"game": {"gameStatus": 3, "gameId": "0022900001", "gameTimeUTC": prediction["commence_time"],
                             "homeTeam": team("OKC", 110), "awayTeam": team("SAS", 100)}}
    monkeypatch.setattr(boxscore, "BoxScore", Box)
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    assert refresh_memory_results(memory, tmp_path / "raw", tmp_path, now=NOW) == []
    assert len(calls) == 1
    result = json.loads((tmp_path / "results_nba.json").read_text())["events"]["0"]
    assert result["players"] and result["observed_at"]
    refresh_memory_results(memory, tmp_path / "raw", tmp_path, now=NOW)
    assert len(calls) == 1  # Six-hour refresh interval.


def test_saved_results_keep_observation_time_until_a_correction(tmp_path):
    actual = {"event_id": "one", "home_score": 100, "away_score": 99, "completed": True,
              "observed_at": "2020-01-01T00:00:00+00:00", "players_observed_at": "2020-01-01T01:00:00+00:00",
              "players": {"OKC:7": {"played": True, "points": 100}}}
    (tmp_path / "results_nba.json").write_text(json.dumps({"events": {"one": actual}}))
    saved = save_results(tmp_path, [{k: v for k, v in actual.items() if k not in {"players", "players_observed_at", "observed_at"}}])
    assert saved["events"]["one"]["observed_at"] == actual["observed_at"]
    assert saved["events"]["one"]["players_observed_at"] == actual["players_observed_at"]
    corrected = save_results(tmp_path, [{"event_id": "one", "home_score": 101, "away_score": 99, "completed": True}])
    assert corrected["events"]["one"]["observed_at"] != actual["observed_at"]
    assert "players" not in corrected["events"]["one"]


def test_advisor_console_and_saved_artifact(tmp_path, capsys, monkeypatch):
    from rich.console import Console
    from nba_sim.reporting.console_report import render_advisor
    import nba_sim.cli as cli
    from nba_sim.config import load_settings
    monkeypatch.setattr(cli, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    args = cli.build_parser().parse_args(["analyze", "--home", "OKC", "--away", "SAS", "--season", "2029-30", "--cached"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline mode must not download results")
    monkeypatch.setattr(cli, "refresh_memory_results", forbidden)
    e = edge(datetime.now(timezone.utc))
    result = cli._build_advisor(e, args, load_settings())
    render_advisor(result, Console())
    output = capsys.readouterr().out
    assert "NO BET" in output and "Paper watch only" in output
    assert (tmp_path / "data" / "processed" / "advisor_memory.json").exists()


def test_offline_cli_saves_advisor_and_remembers_run_without_extra_commands(tmp_path, monkeypatch, capsys):
    from dataclasses import replace
    import nba_sim.cli as cli
    from nba_sim.config import load_settings
    from nba_sim.data.demo_provider import DemoProvider
    settings = load_settings()
    class TestProvider(DemoProvider):
        def load(self, *args, **kwargs):
            bundle = super().load(*args, **kwargs)
            return replace(bundle, metadata=replace(bundle.metadata, is_demo=False, source="synthetic integration-test fixture"))
    current = datetime.now(timezone.utc)
    fixture = {"id": "one", "home_team": "Oklahoma City Thunder", "away_team": "San Antonio Spurs",
               "commence_time": (current + timedelta(hours=2)).isoformat(),
               "bookmakers": [{"key": "book-a", "last_update": current.isoformat(), "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Oklahoma City Thunder", "price": 2},
                                                {"name": "San Antonio Spurs", "price": 2}]}]}]}
    monkeypatch.setattr(cli, "_provider", lambda *args: TestProvider())
    monkeypatch.setattr(cli, "_settings", lambda *args: settings)
    monkeypatch.setattr(cli, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(cli, "load_odds_cache", lambda *args: {"events": [fixture], "retrieved_at": current.isoformat()})
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline command cannot use network")
    monkeypatch.setattr(cli, "refresh_memory_results", forbidden)
    args = cli.build_parser().parse_args(["analyze", "--home", "OKC", "--away", "SAS", "--season", "2029-30",
                                         "--cached", "--simulations", "2", "--workers", "1"])
    assert cli.analyze_command(args) == 0
    run = next((tmp_path / "outputs").iterdir())
    advice = json.loads((run / "advisor.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert set(summary["win_prediction"]["probabilities"]) == {"OKC", "SAS"}
    assert advice["status"] == "no_bet"
    assert advice["selection"] is None
    memory = load_memory(tmp_path / "data" / "processed")
    assert len(memory["predictions"]) == 1
    printed = capsys.readouterr().out
    assert "Learning advisor" in printed
    assert printed.rfind("estimated win probability") > printed.rfind("Exports:")


def test_result_download_failures_do_not_starve_other_games(tmp_path, monkeypatch):
    import nba_sim.advisor.results as results_module
    from nba_api.live.nba.endpoints import boxscore
    memory = learned_memory(6)
    for prediction in memory["predictions"].values():
        prediction.pop("result")
    monkeypatch.setattr(results_module, "find_game_id", lambda p, root: p["event_id"].zfill(10))
    calls = []
    def failure(**kwargs):
        calls.append(kwargs["game_id"])
        raise RuntimeError("offline")
    monkeypatch.setattr(boxscore, "BoxScore", failure)
    refresh_memory_results(memory, tmp_path / "raw", tmp_path, now=NOW)
    assert len(calls) == 5
    refresh_memory_results(memory, tmp_path / "raw", tmp_path, now=NOW + timedelta(hours=7))
    assert calls[5] == "0000000005"  # Previously unattempted fixture gets priority.


def test_missing_game_log_does_not_delay_a_later_refreshed_log(tmp_path):
    memory = learned_memory(1)
    memory["predictions"]["0"].pop("result")
    warnings = refresh_memory_results(memory, tmp_path / "raw", tmp_path, now=NOW)
    assert warnings
    assert "results_checked_at" not in memory
