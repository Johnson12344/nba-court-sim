from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[variable] = "1"

from rich.console import Console
from rich.progress import Progress

from nba_sim.config import load_settings, project_path
from nba_sim.data.csv_provider import CsvProvider
from nba_sim.data.demo_provider import DemoProvider
from nba_sim.data.nba_stats_provider import NbaStatsProvider
from nba_sim.data.validation import validate_data
from nba_sim.evaluation.backtest import walk_forward_backtest
from nba_sim.features.profile_builder import build_team_profile
from nba_sim.features.blending import blend_team_profiles
from nba_sim.features.historical_weights import current_season_weight
from nba_sim.market.odds_api import build_market_comparison, find_matchup, load_odds_cache, refresh_nba_odds
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

    validate = sub.add_parser("validate-data", help="Validate cached official data or manual CSV files")
    validate.add_argument("--season", required=True)
    _add_source(validate)

    simulate = sub.add_parser("simulate", help="Run a Monte Carlo matchup")
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
    _add_source(simulate)

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
    )
    console = Console()
    matched = find_matchup(cache["events"], args.home, args.away) if args.home and args.away else None
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


def simulate_command(args: argparse.Namespace) -> int:
    if args.home == args.away:
        raise ValueError("Home and away teams must differ")
    settings = _settings(args)
    source = "demo" if args.mode == "demo" else args.data_source
    provider = _provider(source, settings)
    assumptions: list[str] = []
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
    count = args.simulations or int(settings["simulation"]["default_count"])
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
        odds_cache = load_odds_cache(project_path("data", "processed"))
        if odds_cache:
            market_comparison = build_market_comparison(
                odds_cache, home.abbreviation, away.abbreviation, summary, player_distributions,
            )
            if market_comparison:
                summary["market_comparison"] = market_comparison
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
        "data_retrieved_at": bundle.metadata.retrieved_at, "is_demo": bundle.metadata.is_demo,
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
    output = export_run(project_path("outputs"), summary, scores, player_distributions, representative, metadata)
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
            "validate-data": validate_command,
            "simulate": simulate_command, "backtest": backtest_command,
        }[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        Console(stderr=True).print(f"[red]Error:[/red] {exc}")
        raise SystemExit(2) from exc
    raise SystemExit(code)
