from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[variable] = "1"

from rich.console import Console
from rich.progress import Progress

from nba_sim.advisor.learning import ADVISOR_VERSION, advise
from nba_sim.advisor.memory import memory_summary, sync_memory
from nba_sim.advisor.results import refresh_memory_results
from nba_sim.config import load_settings, project_path
from nba_sim.data.csv_provider import CsvProvider
from nba_sim.data.demo_provider import DemoProvider
from nba_sim.data.nba_stats_provider import NbaStatsProvider
from nba_sim.data.injury_reports import InjuryReportError, load_injury_report, refresh_injury_report
from nba_sim.data.validation import validate_data
from nba_sim.evaluation.backtest import walk_forward_backtest
from nba_sim.evaluation.forward import evaluate_forward, record_boxscore, refresh_results
from nba_sim.features.availability import apply_lineups, write_lineup_template
from nba_sim.features.automatic_availability import apply_official_report
from nba_sim.features.profile_builder import build_team_profile
from nba_sim.features.blending import blend_team_profiles
from nba_sim.features.historical_weights import current_season_weight
from nba_sim.market.odds_api import build_market_comparison, load_odds_cache, refresh_nba_odds
from nba_sim.market.odds_api import NBA_TEAM_NAMES, UpcomingEventUnavailable
from nba_sim.market.events import age_minutes, select_event, upcoming_events, utc_time
from nba_sim.market.edges import build_edge_report
from nba_sim.reporting.console_report import render_report
from nba_sim.reporting.export import export_run
from nba_sim.reporting.summary import create_summary
from nba_sim.simulation.monte_carlo import run_simulations


def _provider(name: str, settings: dict[str, Any]):
    if name == "csv":
        return CsvProvider(project_path("data", "manual"))
    if name == "nba":
        return NbaStatsProvider(project_path("data", "raw"), project_path("data", "processed"), settings)
    if name == "demo":
        return DemoProvider()
    raise ValueError(f"Unknown data source: {name}")


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-source", choices=("nba", "csv"), default="nba")


def _add_simulation_options(simulate: argparse.ArgumentParser) -> None:
    simulate.add_argument("--home", required=True, type=str.upper)
    simulate.add_argument("--away", required=True, type=str.upper)
    simulate.add_argument("--season", required=True)
    simulate.add_argument("--baseline-season", help="Completed season foundation; defaults to the prior NBA season")
    simulate.add_argument("--history-season", action="append", default=[], help="Optional older season (repeatable)")
    simulate.add_argument("--no-current-season", action="store_true", help="Ignore current-season cache even when present")
    simulate.add_argument("--simulations", type=int)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--model", choices=("possession", "baseline"), default="possession")
    simulate.add_argument("--workers", type=int, default=0, help="Worker processes; 0 selects automatically")
    simulate.add_argument("--mode", choices=("real", "demo"), default="real")
    simulate.add_argument("--no-odds", action="store_true", help="Do not attach a matching locally cached market comparison")
    simulate.add_argument("--event-id", help="Exact upcoming event ID from fixtures")
    lineup = simulate.add_mutually_exclusive_group()
    lineup.add_argument("--write-lineups", type=Path, help="Write a review CSV for this fixture and exit without simulation")
    lineup.add_argument("--lineups", type=Path, help="Apply reviewed player status and expected minutes CSV")
    simulate.add_argument("--max-odds-age-minutes", type=float, default=60, help="Maximum quote age for review candidates")
    simulate.add_argument("--min-ev", type=float, default=0.02, help="Minimum modeled profit per unit for a review candidate")
    simulate.add_argument("--no-advisor", action="store_true", help="Skip the learning advisor and its local memory update")
    _add_source(simulate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba-sim", description="Local NBA possession simulation and analytics")
    parser.add_argument("--settings", type=Path, help="Optional settings YAML path")
    sub = parser.add_subparsers(dest="command", required=True)

    refresh = sub.add_parser("refresh-data", help="Explicitly download and cache official aggregate data")
    refresh.add_argument("--season", required=True)

    enrichment = sub.add_parser("refresh-enrichment", help="Cache player shot charts, PlayByPlayV3, and recent rotations")
    enrichment.add_argument("--season", required=True)
    enrichment.add_argument("--teams", nargs="+", required=True, type=str.upper)
    enrichment.add_argument("--games-per-team", type=int, default=20)
    enrichment.add_argument("--rotation-games", type=int, default=3)
    enrichment.add_argument("--force", action="store_true")

    odds = sub.add_parser("refresh-odds", help="Download and cache current NBA sportsbook lines from The Odds API")
    odds.add_argument("--home", type=str.upper, help="Optional home team abbreviation")
    odds.add_argument("--away", type=str.upper, help="Optional away team abbreviation")
    odds.add_argument("--regions", default="us", help="Bookmaker region, such as us, uk, eu, or au")
    odds.add_argument("--bookmakers", help="Optional comma-separated bookmaker keys")
    odds.add_argument("--include-player-props", action="store_true", help="Also fetch points, rebounds, and assists for this matchup")
    odds.add_argument("--event-id", help="Exact event from fixtures; requires home and away")

    fixtures = sub.add_parser("fixtures", help="List upcoming games from downloaded odds; no API credits")
    fixtures.add_argument("--timezone", default="Africa/Lagos", help="Display timezone (default Africa/Lagos)")

    sub.add_parser("refresh-results", help="Download final scores from the last three days for prediction tracking")
    record = sub.add_parser("record-result", help="Download a final NBA box score for a saved prediction")
    record.add_argument("--run", type=Path, required=True, help="Saved simulation output folder")
    record.add_argument("--nba-game-id", required=True, help="Ten-digit NBA game ID from its NBA.com link")
    sub.add_parser("evaluate", help="Compare saved pregame predictions and paper selections with cached final results")
    memory = sub.add_parser("advisor-memory", help="Show the local advisor's remembered predictions and outcomes")
    memory.add_argument("--refresh", action="store_true", help="Check pending final NBA box scores using cached game logs")

    validate = sub.add_parser("validate-data", help="Validate cached official data or manual CSV files")
    validate.add_argument("--season", required=True)
    _add_source(validate)

    simulate = sub.add_parser("simulate", help="Run a Monte Carlo matchup")
    _add_simulation_options(simulate)

    analyze = sub.add_parser("analyze", help="Refresh matchup data and odds, then show the full simulation report")
    _add_simulation_options(analyze)
    analyze.add_argument("--cached", action="store_true", help="Use downloaded data and odds without network requests")
    analyze.add_argument("--refresh", action="store_true", help="Force current statistics, enrichment, and odds refresh")
    analyze.add_argument("--games-per-team", type=int, default=10)
    analyze.add_argument("--rotation-games", type=int, default=3)
    analyze.add_argument("--regions", default="us")
    analyze.add_argument("--bookmakers", help="Comma-separated bookmaker keys")
    analyze.add_argument("--no-player-props", action="store_true", help="Download game odds only to conserve credits")

    backtest = sub.add_parser("backtest", help="Walk-forward evaluation from historical_games.csv")
    backtest.add_argument("--season", required=True)
    backtest.add_argument("--games", type=int, default=100)
    backtest.add_argument("--file", type=Path, default=project_path("data", "manual", "historical_games.csv"))
    return parser


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return load_settings(args.settings)


def refresh_command(args: argparse.Namespace) -> int:
    settings = _settings(args)
    bundle = _provider("nba", settings).load(args.season, refresh=True)
    Console().print(f"Cached [bold]{len(bundle.teams)} teams[/bold] and [bold]{len(bundle.players)} players[/bold] for {args.season}.")
    return 0


def refresh_enrichment_command(args: argparse.Namespace) -> int:
    settings = _settings(args)
    provider = _provider("nba", settings)
    console = Console()
    enrichment = provider.refresh_enrichment(
        args.season, args.teams, args.games_per_team, args.rotation_games,
        args.force, lambda message: console.print(f"[cyan]{message}[/cyan]"),
    )
    console.print(
        f"[green]Enrichment cached[/green]: {enrichment.metadata['shot_rows']:,} shots, "
        f"{enrichment.metadata['play_by_play_rows']:,} play-by-play events, "
        f"{enrichment.metadata['rotation_rows']:,} rotation rows."
    )
    return 0


def refresh_odds_command(args: argparse.Namespace) -> int:
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "THE_ODDS_API_KEY is not set. In Command Prompt run: set THE_ODDS_API_KEY=your_key"
        )
    cache = refresh_nba_odds(
        project_path("data", "raw"),
        project_path("data", "processed"),
        api_key,
        regions=args.regions,
        bookmakers=args.bookmakers,
        home=args.home,
        away=args.away,
        include_player_props=args.include_player_props,
        event_id=args.event_id,
    )
    console = Console()
    matched = select_event(cache, args.home, args.away, args.event_id) if args.home and args.away else None
    core_quota = cache.get("quota", {}).get("core", {})
    console.print(
        f"[green]Odds cached[/green]: {len(cache['events'])} active/upcoming NBA games from The Odds API. "
        f"Credits remaining: {core_quota.get('remaining', 'unknown')}."
    )
    if args.home and args.away:
        if matched:
            prop_books = len(matched.get("player_prop_bookmakers", []))
            console.print(
                f"Matched [bold]{args.away} at {args.home}[/bold] for {matched.get('commence_time')}. "
                f"Player-prop books: {prop_books}."
            )
        else:
            console.print(f"[yellow]No current sportsbook event matched {args.away} at {args.home}.[/yellow]")
    return 0


def fixtures_command(args: argparse.Namespace) -> int:
    from rich.table import Table
    cache = load_odds_cache(project_path("data", "processed"))
    if not cache:
        raise ValueError("No odds cache. Run refresh-odds first.")
    try:
        zone = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Unknown timezone; use Africa/Lagos or UTC") from exc
    table = Table(title=f"Upcoming NBA fixtures ({args.timezone})")
    for column in ("Start", "Away", "Home", "Event ID", "Books", "Prop books"):
        table.add_column(column, min_width=32 if column == "Event ID" else None,
                         no_wrap=column == "Event ID")
    abbreviations = {name: code for code, name in NBA_TEAM_NAMES.items()}
    for event in upcoming_events(cache):
        table.add_row(utc_time(event["commence_time"]).astimezone(zone).strftime("%m-%d %H:%M"),
                      abbreviations.get(event["away_team"], event["away_team"]),
                      abbreviations.get(event["home_team"], event["home_team"]), str(event["id"]),
                      str(len(event.get("bookmakers", []))), str(len(event.get("player_prop_bookmakers", []))))
    console = Console()
    console.print(table)
    console.print(f"Downloaded: {cache.get('retrieved_at')}. Run refresh-odds for updated fixtures and quotes.")
    return 0


def refresh_results_command(args: argparse.Namespace) -> int:
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("THE_ODDS_API_KEY is not set")
    count, quota = refresh_results(project_path("data", "raw"), project_path("data", "processed"), api_key)
    Console().print(f"Cached {count} final game scores. Credits remaining: {quota.get('remaining')}. Run evaluate next.")
    return 0


def record_result_command(args: argparse.Namespace) -> int:
    result = record_boxscore(args.run, args.nba_game_id, project_path("data", "raw"), project_path("data", "processed"))
    Console().print(f"Recorded final NBA result and {len(result['players'])} player records. Run evaluate next.")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    report, target = evaluate_forward(project_path("outputs"), project_path("data", "processed", "results_nba.json"))
    Console().print_json(json.dumps(report))
    Console().print(f"Evaluation saved: {target}")
    return 0


def advisor_memory_command(args: argparse.Namespace) -> int:
    memory = sync_memory(project_path("outputs"), project_path("data", "processed"))
    console = Console()
    if args.refresh:
        refresh_memory_results(memory, project_path("data", "raw"), project_path("data", "processed"),
                               force=True, progress=lambda value: console.print(value, markup=False))
        memory = sync_memory(project_path("outputs"), project_path("data", "processed"))
    summary = memory_summary(memory, str(_settings(args)["model_version"]), "possession", datetime.now(timezone.utc))
    console.print_json(json.dumps(summary))
    for warning in [*memory.get("warnings", []), *memory.get("result_warnings", [])]:
        console.print(warning, style="yellow", markup=False)
    console.print("Memory is local to this PC. Each market needs 100 training games and 40 later validation games before learned selections can qualify.")
    return 0


def _build_advisor(edge: dict, args: argparse.Namespace, settings: dict) -> dict:
    console = Console()
    try:
        memory = sync_memory(project_path("outputs"), project_path("data", "processed"))
        if args.command == "analyze" and not args.cached:
            refresh_memory_results(memory, project_path("data", "raw"), project_path("data", "processed"),
                                   force=args.refresh, progress=lambda value: console.print(value, markup=False))
            memory = sync_memory(project_path("outputs"), project_path("data", "processed"))
        result = advise(edge, memory, str(settings["model_version"]), args.model)
        result["memory_warnings"] = [*memory.get("warnings", []), *memory.get("result_warnings", [])]
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"advisor_version": ADVISOR_VERSION, "status": "unavailable", "selection": None,
                "paper_watch": None, "reasons": [f"Advisor memory could not be used ({type(exc).__name__}). Existing memory was not reset."],
                "limitations": "No learned selection is available for this run."}


def validate_command(args: argparse.Namespace) -> int:
    settings = _settings(args)
    bundle = _provider(args.data_source, settings).load(args.season, refresh=False)
    report = validate_data(bundle.teams, bundle.players)
    console = Console()
    if report.valid:
        console.print(f"[green]Valid[/green]: {len(bundle.teams)} teams, {len(bundle.players)} players ({bundle.metadata.source})")
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    for error in report.errors:
        console.print(f"[red]Error:[/red] {error}")
    return 0 if report.valid else 1


def _validate_simulation_options(args: argparse.Namespace, settings: dict) -> None:
    if args.home == args.away:
        raise ValueError("Home and away teams must differ")
    if not math.isfinite(args.min_ev) or args.min_ev < 0:
        raise ValueError("--min-ev must be a finite nonnegative number")
    if not math.isfinite(args.max_odds_age_minutes) or args.max_odds_age_minutes <= 0:
        raise ValueError("--max-odds-age-minutes must be a finite positive number")
    count = args.simulations if args.simulations is not None else int(settings["simulation"]["default_count"])
    if not 1 <= count <= int(settings["simulation"].get("max_count", 10000)):
        raise ValueError("Simulation count must be positive and within the configured maximum")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.no_odds and args.event_id:
        raise ValueError("--event-id requires odds analysis; remove --no-odds")


def analyze_command(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _validate_simulation_options(args, settings)
    if args.cached and args.refresh:
        raise ValueError("Use either --cached or --refresh, not both")
    if not 1 <= args.games_per_team <= 82 or not 1 <= args.rotation_games <= 10:
        raise ValueError("Use 1–82 games per team and 1–10 rotation games")
    if args.lineups and not args.lineups.is_file():
        raise FileNotFoundError(f"Lineup file does not exist: {args.lineups}")
    console = Console()
    if args.mode == "demo" or args.cached:
        console.print("Using downloaded inputs; no network requests.")
        return simulate_command(args)
    if args.data_source == "nba":
        provider = _provider("nba", settings)
        start = int(args.season[:4])
        baseline = args.baseline_season or f"{start - 1}-{str(start)[-2:]}"
        seasons = list(dict.fromkeys([baseline, *args.history_season] + ([] if args.no_current_season else [args.season])))
        for season in seasons:
            try:
                bundle = provider.load(season, refresh=False)
            except FileNotFoundError:
                bundle = None
            age = age_minutes(bundle.metadata.retrieved_at, datetime.now(timezone.utc)) if bundle else None
            refresh = bundle is None or (season == args.season and (args.refresh or age is None or age > 360))
            if refresh:
                console.print(f"Updating NBA statistics: {season}")
                provider.load(season, refresh=True)
            else:
                console.print(f"Using cached NBA statistics: {season}")
        enrichment_season = baseline if args.no_current_season else args.season
        enrichment = provider.load_enrichment(enrichment_season, [args.home, args.away])
        enrichment_age = age_minutes(enrichment.metadata.get("refreshed_at"), datetime.now(timezone.utc))
        if (args.refresh or not enrichment.metadata.get("available") or enrichment_age is None or enrichment_age > 360
                or int(enrichment.metadata.get("games_per_team", 0)) < args.games_per_team
                or int(enrichment.metadata.get("rotation_games", 0)) < args.rotation_games):
            console.print(f"Updating shot charts, play-by-play and rotations: {args.away} / {args.home}")
            provider.refresh_enrichment(enrichment_season, [args.home, args.away], args.games_per_team,
                                        args.rotation_games, False, lambda message: console.print(message, markup=False))
        else:
            console.print("Using recently downloaded enrichment.")
    if not args.no_odds:
        cache = load_odds_cache(project_path("data", "processed")) or {}
        try:
            event = select_event(cache, args.home, args.away, args.event_id)
        except ValueError:
            event = None  # The explicit event may be available in a fresh download.
        odds_age = age_minutes(cache.get("retrieved_at"), datetime.now(timezone.utc))
        reuse = (not args.refresh and event is not None and odds_age is not None
                 and odds_age <= min(15, args.max_odds_age_minutes)
                 and cache.get("regions") == args.regions and cache.get("bookmakers_filter") == args.bookmakers
                 and (args.no_player_props or "player_prop_bookmakers" in event))
        if reuse:
            console.print("Using recently downloaded odds.")
        elif os.getenv("THE_ODDS_API_KEY", "").strip():
            console.print("Updating game odds" + (" and available player props." if not args.no_player_props else "."))
            try:
                refresh_odds_command(argparse.Namespace(**{**vars(args), "include_player_props": not args.no_player_props}))
            except UpcomingEventUnavailable:
                if args.event_id or args.lineups or args.write_lineups:
                    raise  # Do not apply a fixture-specific review to an unverified matchup.
                note = "No upcoming sportsbook fixture found for these teams. Showing score/player projections only; no betting comparison."
                console.print(note, style="yellow")
                args.no_odds = True
                args.input_notes = [note]
        else:
            console.print("THE_ODDS_API_KEY is not set: using cached odds if available. Set it to download current prices.", style="yellow")
    console.print("Running the simulation and preparing the complete report...")
    return simulate_command(args)


def simulate_command(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _validate_simulation_options(args, settings)
    odds_cache = load_odds_cache(project_path("data", "processed")) if args.mode != "demo" else None
    event = select_event(odds_cache or {}, args.home, args.away, args.event_id) if args.mode != "demo" else None
    if (args.lineups or args.write_lineups) and event is None:
        raise ValueError("Lineup files require an upcoming fixture. Run refresh-odds, then fixtures and select --event-id.")
    source = "demo" if args.mode == "demo" else args.data_source
    provider = _provider(source, settings)
    assumptions: list[str] = list(getattr(args, "input_notes", []))
    windows: list[str] = []
    if args.mode == "demo":
        bundle = provider.load(args.season, refresh=False)
        baseline_bundle = bundle
        current_bundle = None
        baseline_season = args.season
        baseline_enrichment = provider.load_enrichment(args.season, [args.home, args.away])
    else:
        start = int(args.season[:4])
        baseline_season = args.baseline_season or f"{start - 1}-{str(start)[-2:]}"
        baseline_bundle = provider.load(baseline_season, refresh=False)
        baseline_enrichment = provider.load_enrichment(baseline_season, [args.home, args.away])
        bundle = baseline_bundle
        windows.append(f"completed baseline {baseline_season}")
        current_bundle = None
        current_enrichment = None
        if not args.no_current_season:
            try:
                current_bundle = provider.load(args.season, refresh=False)
                current_enrichment = provider.load_enrichment(args.season, [args.home, args.away])
                windows.append(f"current season-to-date {args.season}")
            except (FileNotFoundError, ValueError):
                assumptions.append(f"No valid local current-season data for {args.season}; using completed-season evidence.")
                current_enrichment = None
        older_bundles = [provider.load(value, refresh=False) for value in args.history_season]
        older_enrichments = [provider.load_enrichment(value, [args.home, args.away]) for value in args.history_season]
        if args.history_season:
            windows.extend(f"older season {value}" for value in args.history_season)
    if args.mode == "demo":
        older_bundles = []
        older_enrichments = []
        current_enrichment = None
    if not baseline_enrichment.metadata.get("available") and args.mode != "demo":
        assumptions.append(
            f"No local play-by-play/shot-chart enrichment for {baseline_season}; run refresh-enrichment for richer shooter and location modelling."
        )
    available = set(baseline_bundle.teams.team_abbreviation.astype(str).str.upper())
    if current_bundle is not None:
        available |= set(current_bundle.teams.team_abbreviation.astype(str).str.upper())
    for older in older_bundles:
        available |= set(older.teams.team_abbreviation.astype(str).str.upper())
    missing = [team for team in (args.home, args.away) if team not in available]
    if missing:
        raise ValueError(f"Team(s) not found for {args.season} in {source}: {', '.join(missing)}. Available: {', '.join(sorted(available))}")
    def profile(team: str):
        in_baseline = team in set(baseline_bundle.teams.team_abbreviation.astype(str).str.upper())
        historical = None
        decay = float(settings["blending"]["older_season_decay"])
        for older, enrichment in zip(older_bundles, older_enrichments):
            if team not in set(older.teams.team_abbreviation.astype(str).str.upper()):
                continue
            candidate = build_team_profile(
                older.teams, older.players, team, settings,
                enrichment.shots, enrichment.play_by_play, enrichment.rotations,
            )
            historical = candidate if historical is None else blend_team_profiles(historical, candidate, 1.0 / (1.0 + decay), candidate.season)
        if in_baseline:
            candidate = build_team_profile(
                baseline_bundle.teams, baseline_bundle.players, team, settings,
                baseline_enrichment.shots, baseline_enrichment.play_by_play,
                baseline_enrichment.rotations if baseline_season == args.season else None,
            )
            historical = candidate if historical is None else blend_team_profiles(historical, candidate, 1.0 / (1.0 + decay), baseline_season)
        if current_bundle is None or team not in set(current_bundle.teams.team_abbreviation.astype(str).str.upper()):
            if historical is None:
                raise ValueError(f"{team} has no usable profile")
            return historical
        recent = build_team_profile(
            current_bundle.teams, current_bundle.players, team, settings,
            current_enrichment.shots if current_enrichment else None,
            current_enrichment.play_by_play if current_enrichment else None,
            current_enrichment.rotations if current_enrichment else None,
        )
        if historical is None:
            assumptions.append(f"{team} has no {baseline_season} profile; current-season evidence is used without a completed-season prior.")
            return recent
        weight = current_season_weight(recent.games, float(settings["blending"]["current_season_prior_games"]))
        return blend_team_profiles(historical, recent, weight, args.season, float(settings["blending"]["recent_minutes_weight"]))

    home = profile(args.home)
    away = profile(args.away)
    if args.write_lineups:
        write_lineup_template(args.write_lineups, (home, away), str(event["id"]))
        Console().print(f"Lineup review file: {args.write_lineups}. Review all rows before using --lineups.")
        Console().print("Set status to active/out/questionable; optional expected_minutes and starter. Add source and reviewed_at (ISO time with timezone).")
        return 0
    availability = {"reviewed": False, "reasons": ["Player availability has not been reviewed for this fixture"], "rows": []}
    if args.lineups:
        (home, away), availability = apply_lineups(args.lineups, (home, away), str(event["id"]))
    elif args.mode != "demo":
        injury_report = load_injury_report(project_path("data", "processed"), args.season)
        injury_error = None
        if args.command == "analyze" and not args.cached and event:
            Console().print("Checking the official NBA injury report (no API key or Odds API credits needed).")
            try:
                injury_report = refresh_injury_report(project_path("data", "raw"), project_path("data", "processed"),
                                                      args.season, force=args.refresh)
            except (InjuryReportError, OSError) as exc:
                injury_error = f"Automatic injury refresh failed: {exc}"
                Console().print(injury_error, style="yellow", markup=False)
        (home, away), availability = apply_official_report((home, away), event, injury_report,
                                                          refresh_error=injury_error)
    readiness = list(availability["reasons"])
    if args.model != "possession":
        readiness.append("Score-only baseline does not model player availability")
    latest_data = current_bundle.metadata if current_bundle is not None else baseline_bundle.metadata
    data_age = age_minutes(latest_data.retrieved_at, datetime.now(timezone.utc))
    if current_bundle is None or data_age is None or data_age > 48 * 60:
        readiness.append("Refresh current-season NBA data (maximum age for review candidates: 48 hours)")
    if current_enrichment is None or not current_enrichment.metadata.get("available"):
        readiness.append("Refresh current-season enrichment for both teams")
    else:
        enrichment_age = age_minutes(current_enrichment.metadata.get("refreshed_at"), datetime.now(timezone.utc))
        if enrichment_age is None or enrichment_age > 48 * 60:
            readiness.append("Current-season enrichment is older than 48 hours or undated")
    for team in (home, away):
        if not team.data_evidence.get("rotation_games"):
            readiness.append(f"No recent verified rotations for {team.abbreviation}")
    count = args.simulations if args.simulations is not None else int(settings["simulation"]["default_count"])
    max_count = int(settings["simulation"].get("max_count", 10000))
    if count > max_count:
        raise ValueError(f"Simulation count is capped at {max_count:,}")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    workers = args.workers or (1 if count < 500 else min(4, max(1, (os.cpu_count() or 2) - 1)))
    with Progress() as progress:
        task = progress.add_task("Simulating", total=count)
        games, representative = run_simulations(home, away, count, args.seed, settings, args.model, lambda _: progress.advance(task), workers)
    summary, scores, player_distributions = create_summary(games, home, away)
    summary.update({
        "disclaimer": "Simulation distributions are model outputs, not guaranteed outcomes or betting recommendations.",
        "model": args.model,
    })
    market_comparison = None
    if not args.no_odds and args.mode != "demo":
        if odds_cache and event:
            market_comparison = build_market_comparison(
                odds_cache, home.abbreviation, away.abbreviation, summary, player_distributions,
                event=event,
            )
            if market_comparison:
                summary["market_comparison"] = market_comparison
            summary["betting_analysis"] = build_edge_report(
                odds_cache, event, games, home, away, readiness_reasons=readiness,
                max_odds_age_minutes=args.max_odds_age_minutes, min_ev=args.min_ev,
            )
        else:
            assumptions.append("No upcoming cached odds match. Run refresh-odds and fixtures to select an event.")
    metadata_assumptions = list(bundle.metadata.assumptions) + assumptions
    active_enrichment = current_enrichment if current_enrichment and current_enrichment.metadata.get("available") else baseline_enrichment
    if active_enrichment.metadata.get("available"):
        metadata_assumptions = [
            value for value in metadata_assumptions
            if "Shot-zone and play-by-play enrichment are unavailable" not in value
        ]
        metadata_assumptions.append(
            "PlayByPlayV3 and ShotChartDetail enrichment is active; defender identity remains unavailable."
        )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "home_team": home.abbreviation, "away_team": away.abbreviation,
        "season": args.season, "simulations": count, "seed": args.seed, "model": args.model,
        "workers": workers,
        "model_version": settings["model_version"], "data_source": bundle.metadata.source,
        "data_retrieved_at": latest_data.retrieved_at, "is_demo": bundle.metadata.is_demo,
        "availability": availability, "model_settings": settings,
        "advisor_enabled": not args.no_advisor,
        "data_snapshots": [{"season": b.metadata.season, "retrieved_at": b.metadata.retrieved_at}
                           for b in [baseline_bundle, *older_bundles, *([current_bundle] if current_bundle is not None else [])]],
        "historical_windows": windows or [args.season],
        "enrichment": active_enrichment.metadata,
        "market_data": ({
            "provider": market_comparison["provider"],
            "retrieved_at": market_comparison["retrieved_at"],
            "event_id": market_comparison["event_id"],
            "commence_time": market_comparison["commence_time"],
        } if market_comparison else None),
        "coverage": bundle.metadata.coverage, "assumptions": metadata_assumptions,
        "reproduction_command": " ".join(sys.argv),
    }
    if summary.get("betting_analysis") and not args.no_advisor and not metadata["is_demo"]:
        summary["advisor"] = _build_advisor(summary["betting_analysis"], args, settings)
    output = export_run(project_path("outputs"), summary, scores, player_distributions, representative, metadata)
    if summary.get("advisor"):
        try:
            sync_memory(project_path("outputs"), project_path("data", "processed"))
        except (OSError, ValueError, KeyError, TypeError):
            Console().print("The report was saved, but advisor memory could not be updated. Keep the output folder for a later retry.", style="yellow")
    render_report(summary, player_distributions, representative, metadata, home, away, str(output))
    return 0


def backtest_command(args: argparse.Namespace) -> int:
    report, target = walk_forward_backtest(args.file, args.season, args.games, project_path("outputs"))
    Console().print_json(json.dumps(report))
    Console().print(f"Exports: [bold]{target}[/bold]")
    return 0


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = {
            "refresh-data": refresh_command, "refresh-enrichment": refresh_enrichment_command,
            "refresh-odds": refresh_odds_command,
            "fixtures": fixtures_command, "refresh-results": refresh_results_command,
            "record-result": record_result_command, "evaluate": evaluate_command,
            "advisor-memory": advisor_memory_command,
            "validate-data": validate_command,
            "simulate": simulate_command, "backtest": backtest_command,
            "analyze": analyze_command,
        }[args.command](args)
    except (OSError, RuntimeError, ValueError) as exc:
        Console(stderr=True).print(f"[red]Error:[/red] {exc}")
        raise SystemExit(2) from exc
    raise SystemExit(code)
