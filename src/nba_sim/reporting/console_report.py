from __future__ import annotations

from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nba_sim.models import SimulatedGame, TeamProfile


def render_advisor(advisor: dict, console: Console) -> None:
    console.print("[bold]Learning advisor[/bold]")
    selected = advisor.get("selection")
    console.print("Decision: candidate for review" if selected else "Decision: NO BET", style="cyan" if selected else "yellow")
    memory = advisor.get("memory", {})
    if memory:
        console.print(f"Memory: {memory['saved_fixtures']} saved fixtures; {memory['completed_fixtures']} completed; "
                      f"{memory['pending_fixtures']} awaiting results.")
        if not selected:
            console.print("Learning requires 100 training games plus 40 later validation games per market, and must pass the probability checks.")
    for reason in advisor.get("reasons", []):
        console.print(reason, markup=False)
    row = selected or advisor.get("paper_watch")
    if row:
        label = f"{row['market']} | {row.get('player') or row.get('team') or ''} {row['selection']}"
        if row.get("line") is not None:
            label += f" {row['line']:+g}"
        table = Table(title="Highest estimated win chance among qualifying offers" if selected else "Paper watch only - learning is not ready for this selection")
        for column in ("Selection", "Book", "Odds", "Win %", "Loss %", "Push %", "EV %"):
            table.add_column(column)
        table.add_row(label, row["bookmaker"], f"{row['decimal_odds']:.2f}",
                      f"{100 * row['estimated_win_probability']:.1f}", f"{100 * row['estimated_loss_probability']:.1f}",
                      f"{100 * row['push_probability']:.1f}", f"{100 * row['adjusted_ev']:+.1f}")
        console.print(table)
        console.print(row["probability_source"], markup=False)
        fit = advisor.get("models", {}).get(row["market"], {})
        console.print(f"Learning for this market: {fit.get('games', 0)} usable games. {fit.get('reason', '')}", markup=False)
    if advisor.get("memory_warnings"):
        console.print(f"{len(advisor['memory_warnings'])} memory/result warnings saved in advisor.json; unresolved outcomes are not counted as losses.", style="yellow")
    console.print(advisor.get("limitations", ""), style="dim", markup=False)


def render_report(
    summary: dict[str, Any],
    players: pd.DataFrame,
    representative: SimulatedGame,
    metadata: dict[str, Any],
    home: TeamProfile,
    away: TeamProfile,
    output_path: str,
) -> None:
    console = Console()
    title = "DEMO DATA — NBA SIMULATION" if metadata["is_demo"] else "NBA SIMULATION"
    console.print(Panel.fit(
        f"[bold]{away.abbreviation} at {home.abbreviation}[/bold]\n"
        f"{metadata['simulations']:,} runs | seed {metadata['seed']} | {metadata['model']} model\n"
        f"Source: {metadata['data_source']} | model {metadata['model_version']}",
        title=title, border_style="yellow" if metadata["is_demo"] else "blue",
    ))
    if summary.get("advisor"):
        render_advisor(summary["advisor"], console)
    availability = metadata.get("availability", {})
    if availability.get("mode") == "automatic":
        console.print("[bold]Automatic player availability[/bold]")
        if availability.get("source"):
            console.print(f"NBA report published: {availability.get('published_at')} | Checked: {availability.get('retrieved_at')}", markup=False)
        excluded = availability.get("excluded", [])
        console.print("Excluded (out): " + ("; ".join(excluded) if excluded else "none applied"), markup=False)
        for reason in availability.get("reasons", []):
            console.print(f"Availability warning: {reason}", style="yellow", markup=False)
        console.print(availability["note"], style="dim", markup=False)
    table = Table(title="Team scoring distributions (simulated, not guaranteed)")
    for column in ("Team", "Mean", "Median", "SD", "P10", "P90", "Win %"):
        table.add_column(column, justify="right" if column != "Team" else "left")
    for team in (away, home):
        data = summary["teams"][team.abbreviation]
        score = data["score"]
        table.add_row(team.abbreviation, f"{score['mean']:.1f}", f"{score['median']:.1f}", f"{score['std_dev']:.1f}", f"{score['p10']:.0f}", f"{score['p90']:.0f}", f"{100*data['win_rate']:.1f}")
    console.print(table)
    total = summary.get("game_total")
    if total:
        console.print(Panel.fit(
            f"[bold]Projected score: {away.abbreviation} {summary['teams'][away.abbreviation]['score']['mean']:.1f}"
            f" — {home.abbreviation} {summary['teams'][home.abbreviation]['score']['mean']:.1f}[/bold]\n"
            f"[bold]Combined total: {total['mean']:.1f} points[/bold]\n"
            f"Median total: {total['median']:.1f} | Middle 80% of simulated totals: {total['p10']:.0f}–{total['p90']:.0f}",
            title="Score projection (simulation averages)",
        ))
    market = summary.get("market_comparison")
    if market:
        market_table = Table(title="Sportsbook market comparison (context only, not a recommendation)")
        market_table.add_column("Measure")
        market_table.add_column("Simulation")
        market_table.add_column("Market consensus")
        market_table.add_column("Difference")
        home_spread = market.get("home_spread")
        market_margin = -home_spread if home_spread is not None else None
        market_table.add_row(
            f"{home.abbreviation} margin",
            f"{market['model_home_margin']:+.1f}",
            f"{market_margin:+.1f} implied ({home.abbreviation} {home_spread:+.1f})" if market_margin is not None else "Unavailable",
            f"{market['model_minus_market_margin']:+.1f}" if market.get("model_minus_market_margin") is not None else "—",
        )
        market_table.add_row(
            "Game total",
            f"{market['model_total']:.1f}",
            f"{market['total']:.1f}" if market.get("total") is not None else "Unavailable",
            f"{market['model_minus_market_total']:+.1f}" if market.get("model_minus_market_total") is not None else "—",
        )
        market_table.add_row(
            f"{home.abbreviation} win probability",
            f"{100 * market['model_home_win_probability']:.1f}%",
            f"{100 * market['home_no_vig_win_probability']:.1f}% (no-vig)" if market.get("home_no_vig_win_probability") is not None else "Unavailable",
            "—",
        )
        console.print(market_table)
        console.print(
            f"[dim]The Odds API: {market['sportsbook_count']} sportsbook(s); event {market.get('commence_time')}; "
            f"downloaded {market.get('retrieved_at')}.[/dim]"
        )
        if market.get("is_stale"):
            console.print("[yellow]Market cache is more than 6 hours old or has no valid timestamp; run refresh-odds before relying on the comparison.[/yellow]")
        props = sorted(market.get("player_props", []), key=lambda row: (row["player"], row["stat"]))[:12]
        if props:
            props_table = Table(title="Player projection versus consensus prop line (context only)")
            for column in ("Player", "Stat", "Model avg", "Market line", "Difference", "Books"):
                props_table.add_column(column)
            for row in props:
                props_table.add_row(
                    row["player"], row["stat"], f"{row['model_mean']:.1f}", f"{row['market_line']:.1f}",
                    f"{row['model_minus_line']:+.1f}", str(row["sportsbook_count"]),
                )
            console.print(props_table)
    analysis = summary.get("betting_analysis")
    if analysis:
        console.print("[bold]Apparent betting edges — model probabilities are not yet validated.[/bold]")
        for reason in analysis["readiness_reasons"]:
            if availability.get("mode") == "automatic" and reason in availability.get("reasons", []):
                continue  # Already displayed above, including runs without odds.
            console.print(f"Needs review: {reason}", style="yellow", markup=False)
        offers = analysis["offers"]
        candidates = [row for row in offers if row["review_candidate"]]
        display = candidates[:12] if candidates else offers[:8]
        table = Table(title="Raw simulation candidates (advisor may reject)" if candidates else "Price comparisons — no offers passed all review checks")
        for column in ("Book", "Market / selection", "Odds", "Win %", "Push %", "EV %", "MC lower %", "Status"):
            table.add_column(column)
        for row in display:
            selection = f"{row['market']} {row.get('player') or row.get('team') or ''} {row['selection']}"
            if row["line"] is not None:
                selection += f" {row['line']:+g}"
            table.add_row(row["bookmaker"], selection, f"{row['decimal_odds']:.2f}",
                          f"{100 * row['win_probability']:.1f}", f"{100 * row['push_probability']:.1f}",
                          f"{100 * row['expected_profit_per_unit']:+.1f}", f"{100 * row['mc_lower_profit_per_unit']:+.1f}",
                          "Review" if row["review_candidate"] else "Blocked" if row["blockers"] else "Below threshold")
        console.print(table)
        console.print("EV is modeled profit per 100 units staked. MC lower bounds cover simulation noise only, not model error. Review candidates are not proven profitable bets.")
        console.print(f"{len(candidates)} review candidates / {len(offers)} quotes. All offers and blockers are saved in market_probabilities.csv.")
        if analysis["skipped"]:
            console.print(f"Skipped {len(analysis['skipped'])} types of unmatched/unsupported offers; details in betting_analysis.json.")
    if not players.empty:
        leaders = players.loc[players.stat == "points"].sort_values(["team", "mean"], ascending=[True, False])
        averages = {(r.team, r.player_id, r.stat): r.mean for r in players.itertuples()}
        ptable = Table(title="Player projections — all modeled players (simulation averages)")
        for column in ("Player", "Team", "Min", "Points", "Reb", "Ast", "Points P10–P90"):
            ptable.add_column(column)
        for row in leaders.itertuples():
            def average(stat):
                value = averages.get((row.team, row.player_id, stat))
                return f"{value:.1f}" if value is not None else "—"
            ptable.add_row(row.player_name, row.team, average("minutes"), f"{row.mean:.1f}", average("rebounds"),
                           average("assists"), f"{row.p10:.0f}–{row.p90:.0f}")
        console.print(ptable)
    quarters = " | ".join(
        f"{'Q' if q['period'] <= 4 else 'OT'}{q['period'] if q['period'] <= 4 else q['period']-4}: {q['away']}-{q['home']}"
        for q in representative.quarter_scores
    )
    console.print(Panel(
        f"[bold]{representative.away} {representative.away_score}, {representative.home} {representative.home_score}[/bold]\n{quarters}\n"
        "Chosen as the run nearest the median total and margin. It is generated, not an actual game.",
        title="Representative simulated game",
    ))
    factors = Table(title="Model factors and evidence status")
    factors.add_column("Factor")
    factors.add_column("Value / interpretation")
    factors.add_row("Estimated pace", f"{(home.pace + away.pace)/2:.1f} possessions per team; derived from aggregate observations")
    factors.add_row("Turnover rates", f"{away.abbreviation} {away.turnover_rate:.3f}; {home.abbreviation} {home.turnover_rate:.3f}")
    factors.add_row("3-point attempt share", f"{away.abbreviation} {away.zone_shares['corner_three'] + away.zone_shares['above_break_three']:.1%}; {home.abbreviation} {home.zone_shares['corner_three'] + home.zone_shares['above_break_three']:.1%}")
    factors.add_row(
        "Shooter selection",
        f"PlayByPlayV3 coverage: {away.abbreviation} {away.data_evidence.get('play_by_play_games', 0)} games; "
        f"{home.abbreviation} {home.data_evidence.get('play_by_play_games', 0)} games",
    )
    factors.add_row(
        "Player shot locations",
        f"Shot-chart attempts: {away.abbreviation} {away.data_evidence.get('shot_chart_attempts', 0):,}; "
        f"{home.abbreviation} {home.data_evidence.get('shot_chart_attempts', 0):,}",
    )
    factors.add_row(
        "Recent rotations",
        f"Verified box-score games: {away.abbreviation} {away.data_evidence.get('rotation_games', 0)}; "
        f"{home.abbreviation} {home.data_evidence.get('rotation_games', 0)}",
    )
    factors.add_row("Defense", "Team zone factor is neutral when opponent zone-defense data is unavailable; no defender assignments are claimed")
    console.print(factors)
    for assumption in metadata.get("assumptions", []):
        console.print(f"[yellow]Assumption/limitation:[/yellow] {assumption}")
    console.print(f"Exports: [bold]{output_path}[/bold]")
