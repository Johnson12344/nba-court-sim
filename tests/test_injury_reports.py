from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest
import requests

import nba_sim.data.injury_reports as feed
from nba_sim.features.automatic_availability import apply_official_report, player_key

SOURCE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-30_04_30PM.pdf"
NOW = datetime(2026, 10, 30, 21, tzinfo=timezone.utc)
HEADER = "Injury Report: 10/30/26 04:30 PM"


def report_for(teams, now=NOW):
    publication = now.astimezone(feed.EASTERN).replace(second=0, microsecond=0) - timedelta(minutes=1)
    source = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_" + publication.strftime("%Y-%m-%d_%I_%M%p.pdf")
    start = now + timedelta(hours=3)
    event = {"id": "one", "home_team": teams[0].name, "away_team": teams[1].name,
             "commence_time": start.isoformat()}
    # Demo TeamProfile names intentionally differ from the sportsbook names.
    event.update(home_team="Oklahoma City Thunder", away_team="San Antonio Spurs")
    sections = [{"team": t.abbreviation, "game_date": start.astimezone(feed.EASTERN).date().isoformat(),
                 "matchup": "SAS@OKC", "submitted": True, "players": []} for t in teams]
    player = teams[0].players[0]
    first, last = player.name.rsplit(" ", 1)
    sections[0]["players"].append({"name": f"{last}, {first}", "status": "out", "page": 1})
    return event, {"schema_version": feed.SCHEMA_VERSION, "source": source, "season": "2026-27",
                   "published_at": publication.astimezone(timezone.utc).isoformat(), "retrieved_at": now.isoformat(),
                   "teams": sections}


def test_latest_link_rejects_future_reports_and_nonofficial_hosts():
    urls = [SOURCE, SOURCE.replace("04_30PM", "05_30PM"), SOURCE.replace("ak-static.cms.nba.com", "example.com")]
    html = "".join(f'<a href="{u}">report</a>' for u in urls)
    assert feed.latest_report_url(html, "https://official.nba.com/", NOW) == SOURCE
    assert feed.report_time(SOURCE) == NOW - timedelta(minutes=30)
    # Eastern standard time in winter differs from October's daylight time.
    assert feed.report_time(SOURCE.replace("10-30", "12-30")).hour == 21
    with pytest.raises(feed.InjuryReportError):
        feed.latest_report_url("", "https://official.nba.com/", NOW)


def test_parser_handles_continuation_pages_inverted_layout_and_pending_teams():
    pages = [HEADER + "\nGame Date Game Time Matchup Team Player Name Current Status Reason\n"
             "10/30/2026 07:30 (ET) NYK@PHI New York Knicks Example Jr., Alex Out Injury/Illness - Left ankle\n"
             "Philadelphia 76ers Example, Bobby Questionable Injury/Illness - Knee\nPage 1 of 2",
             "Page 2 of 2\nNew York Knicks NOT YET SUBMITTED\n"
             "10/31/2026 07:00 (ET) PHI@NYK Philadelphia 76ers NOT YET SUBMITTED\n"
             "Other, Charlie Available Injury/Illness - Wrist\n" + HEADER]
    report = feed.parse_report_pages(pages, SOURCE)
    assert len(report["teams"]) == 4
    first_philly = report["teams"][1]
    assert [p["status"] for p in first_philly["players"]] == ["questionable", "available"]
    assert first_philly["players"][1]["page"] == 2
    assert not report["teams"][2]["submitted"]
    assert not report["teams"][3]["submitted"]
    assert report["teams"][2]["game_date"] == "2026-10-31"


@pytest.mark.parametrize("text", [
    "10/30/2026 07:30 (ET) NYK@PHI New York Knicks Example, Alex Unknown Injury",
    "10/30/2026 07:30 (ET) NYK@PHI Boston Celtics Example, Alex Out Injury",
    "Example, Alex Out Injury", "unreadable table",
])
def test_unrecognized_rows_never_become_a_clean_report(text):
    with pytest.raises(feed.InjuryReportError):
        feed.parse_report_pages([HEADER + "\n" + text], SOURCE)


def test_mismatched_publication_and_broken_pdf_fail_closed():
    with pytest.raises(feed.InjuryReportError, match="publication"):
        feed.parse_report_pages([HEADER.replace("04:30", "04:15")], SOURCE)
    with pytest.raises(feed.InjuryReportError):
        feed.parse_report_pdf(b"not a PDF", SOURCE)


def test_name_normalization_retains_suffixes_and_handles_accents():
    assert player_key("Smith Jr., Jose") == player_key("José Smith Jr.")
    assert player_key("Butler III, Jimmy") == player_key("Jimmy Butler III")
    assert player_key("Smith Jr., Alex") != player_key("Alex Smith")


def test_out_removed_and_minutes_redistributed_without_manual_inputs(demo_profiles):
    event, report = report_for(demo_profiles)
    teams, review = apply_official_report(demo_profiles, event, report, now=NOW)
    assert review["reviewed"]
    assert demo_profiles[0].players[0].player_id not in {p.player_id for p in teams[0].players}
    assert sum(p.expected_minutes for p in teams[0].players) == pytest.approx(240)
    assert all(p.expected_minutes <= 48 for p in teams[0].players)
    assert set(r["status"] for r in review["rows"]) == {"out", "not_listed"}
    assert all(r["source"] == report["source"] for r in review["rows"])
    assert review["game_date_eastern"] == "2026-10-30"  # Event's UTC date is Oct 31.
    assert review["rows"][0]["expected_minutes"] == 0


@pytest.mark.parametrize("status", ["probable", "questionable", "doubtful"])
def test_uncertain_players_are_explicit_conditional_scenarios(demo_profiles, status):
    event, report = report_for(demo_profiles)
    report["teams"][0]["players"][0]["status"] = status
    teams, review = apply_official_report(demo_profiles, event, report, now=NOW)
    assert len(teams[0].players) == len(demo_profiles[0].players)
    assert not review["reviewed"]
    assert status in review["uncertain"][0]
    assert "modeled as playing" in review["reasons"][0]


@pytest.mark.parametrize("change", ["pending", "wrong_date", "wrong_fixture", "ambiguous", "unmatched"])
def test_missing_coverage_or_name_mismatch_never_clears_availability(demo_profiles, change):
    event, report = report_for(demo_profiles)
    section = report["teams"][0]
    if change == "pending":
        section.update(submitted=False, players=[])
    elif change == "wrong_date":
        section["game_date"] = "2026-10-31"
    elif change == "wrong_fixture":
        section["matchup"] = "OKC@SAS"
    elif change == "ambiguous":
        section["players"].append(deepcopy(section["players"][0]))
    else:
        section["players"][0]["name"] = "Missing, Player"
    teams, review = apply_official_report(demo_profiles, event, report, now=NOW)
    assert not review["reviewed"]
    assert len(teams[0].players) == len(demo_profiles[0].players)
    assert all(r["status"] == "unverified" for r in review["rows"] if r["team"] == "OKC")


@pytest.mark.parametrize("change", ["old_report", "future", "missing", "no_event", "past_event"])
def test_stale_or_unverified_report_does_not_change_roster(demo_profiles, change):
    event, report = report_for(demo_profiles)
    if change == "old_report":
        old = NOW - timedelta(hours=13)
        report["published_at"] = old.isoformat()
        report["source"] = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_" + old.astimezone(feed.EASTERN).strftime("%Y-%m-%d_%I_%M%p.pdf")
    elif change == "future":
        report["retrieved_at"] = (NOW + timedelta(minutes=5)).isoformat()
    elif change == "missing":
        report = None
    elif change == "no_event":
        event = None
    else:
        event["commence_time"] = (NOW - timedelta(minutes=5)).isoformat()
    teams, review = apply_official_report(demo_profiles, event, report, now=NOW)
    assert teams == demo_profiles
    assert not review["reviewed"]


def test_too_few_remaining_players_stops_instead_of_inventing_rotation(demo_profiles):
    teams = (replace(demo_profiles[0], players=demo_profiles[0].players[:5]), demo_profiles[1])
    event, report = report_for(teams)
    with pytest.raises(ValueError, match="fewer than five"):
        apply_official_report(teams, event, report, now=NOW)


def test_refresh_downloads_and_preserves_provenance_without_odds_key(tmp_path, monkeypatch):
    calls = []
    class Response:
        text = f'<a href="{SOURCE}">Latest</a>'
        content = b"pdf-bytes"
        def raise_for_status(self):
            pass
    def get(url, **kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(feed.requests, "get", get)
    monkeypatch.setattr(feed, "parse_report_pdf", lambda data, url: feed.parse_report_pages([
        HEADER + "\n10/30/2026 07:00 (ET) SAS@OKC San Antonio Spurs NOT YET SUBMITTED\n"
        "Oklahoma City Thunder NOT YET SUBMITTED"], url))
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    report = feed.refresh_injury_report(raw, processed, "2026-27", now=NOW)
    assert len(calls) == 2
    assert report["retrieved_at"] == NOW.isoformat()
    assert (raw / SOURCE.rsplit("/", 1)[1]).read_bytes() == b"pdf-bytes"
    assert feed.refresh_injury_report(raw, processed, "2026-27", now=NOW) == report
    assert len(calls) == 2  # Five-minute reuse does not pretend to recheck.
    def fail(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(feed.requests, "get", fail)
    with pytest.raises(feed.InjuryReportError):
        feed.refresh_injury_report(raw, processed, "2026-27", now=NOW, force=True)
    assert feed.load_injury_report(processed, "2026-27") == report


def test_failed_refresh_keeps_cached_outs_but_blocks_review(demo_profiles):
    event, report = report_for(demo_profiles)
    teams, review = apply_official_report(demo_profiles, event, report, now=NOW, refresh_error="Refresh failed")
    assert review["excluded"]
    assert not review["reviewed"]
    assert "Refresh failed" in review["reasons"]


def test_unavailable_index_falls_back_to_recent_public_pdf_names(monkeypatch):
    calls = []
    class Response:
        text = ""
        content = b"public-pdf"
        def __init__(self, status):
            self.status_code = status
        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError()
    def get(url, **kwargs):
        calls.append(url)
        return Response(200 if url == SOURCE else 403)
    monkeypatch.setattr(feed.requests, "get", get)
    source, content = feed._download_report("2026-27", NOW)
    assert (source, content) == (SOURCE, b"public-pdf")
    assert len(calls) == 4  # Index, 5:00, 4:45, then published 4:30 ET report.
    assert all(feed.report_time(url) <= NOW for url in calls[1:])


def test_pdf_fallback_is_bounded_and_does_not_use_this_season_for_another(monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = 404
        return response
    monkeypatch.setattr(feed.requests, "get", get)
    with pytest.raises(feed.InjuryReportError):
        feed._download_report("2026-27", NOW)
    assert len(calls) == 9
    calls.clear()
    with pytest.raises(feed.InjuryReportError):
        feed._download_report("2025-26", NOW)
    assert len(calls) == 1


def test_corrupted_cache_is_treated_as_missing(tmp_path):
    path = tmp_path / "injuries_2026-27.json"
    path.write_text('{"schema_version": 1, "teams": [null]}')
    assert feed.load_injury_report(tmp_path, "2026-27") is None
    path.write_text("not json")
    assert feed.load_injury_report(tmp_path, "2026-27") is None


def test_analyze_automatically_applies_report_and_exports_without_csv(tmp_path, monkeypatch, capsys):
    import nba_sim.cli as cli
    from nba_sim.config import load_settings
    from nba_sim.data.demo_provider import DemoProvider
    from nba_sim.features.profile_builder import build_team_profile
    settings = load_settings()
    bundle = DemoProvider().load("2026-27")
    teams = tuple(build_team_profile(bundle.teams, bundle.players, t, settings) for t in ("OKC", "SAS"))
    event, report = report_for(teams, datetime.now(timezone.utc))
    monkeypatch.setattr(cli, "_provider", lambda *args: DemoProvider())
    monkeypatch.setattr(cli, "_settings", lambda *args: settings)
    monkeypatch.setattr(cli, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(cli, "load_odds_cache", lambda *args: {"events": [event]})
    downloads = []
    def refresh(*args, **kwargs):
        downloads.append(True)
        return report
    monkeypatch.setattr(cli, "refresh_injury_report", refresh)
    args = cli.build_parser().parse_args(["analyze", "--home", "OKC", "--away", "SAS", "--season", "2026-27",
                                         "--data-source", "csv", "--no-odds", "--simulations", "2", "--workers", "1"])
    assert cli.analyze_command(args) == 0
    assert downloads == [True]
    output = next((tmp_path / "outputs").iterdir())
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["availability"]["reviewed"]
    assert metadata["availability"]["excluded"]
    assert metadata["availability"]["source"] == report["source"]
    assert not list(tmp_path.rglob("*lineups*"))
    console = capsys.readouterr().out
    assert "Automatic player availability" in console
    assert "Combined total:" in console
    # Offline use must never reach the injury download function.
    args.cached = True
    assert cli.analyze_command(args) == 0
    assert downloads == [True]
