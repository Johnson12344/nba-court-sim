# Upcoming-game analysis

This workflow estimates probabilities for upcoming NBA fixtures and compares them with actual bookmaker prices. An apparent positive expected value (EV) is a model estimate, not evidence that the model is profitable. No bets are placed. Player availability is reviewed manually; the program does not automatically read injury reports.

All commands below are for **CMD**, opened in the project folder. On another PC, first run `git pull` and `.venv\Scripts\python.exe -m pip install -e ".[dev]"` to update the program.

## 1. Find a fixture

Set your key in the terminal if needed, then download odds and list upcoming games:

```cmd
set "THE_ODDS_API_KEY=your_key_here"
.venv\Scripts\python.exe -m nba_sim refresh-odds
.venv\Scripts\python.exe -m nba_sim fixtures
```

The list uses **Africa/Lagos time** and shows the away team, home team, and full event ID. Only future events are listed. Use `fixtures --timezone UTC` for UTC. The list comes from cached bookmaker events, so it may not include every scheduled NBA game.

The remaining examples use **SAC at NOP**. Substitute your chosen teams and replace `EVENT_ID` with the event ID shown. Matching the event prevents using a different meeting between the same teams.

## 2. Refresh basketball data

Download the baseline once if it is not on this PC:

```cmd
.venv\Scripts\python.exe -m nba_sim refresh-data --season 2025-26
```

Before analysing your matchup, update the current season and the teams' detailed data:

```cmd
.venv\Scripts\python.exe -m nba_sim refresh-data --season 2026-27
.venv\Scripts\python.exe -m nba_sim refresh-enrichment --season 2026-27 --teams NOP SAC --games-per-team 10 --rotation-games 3
```

These NBA.com downloads do not use Odds API credits. Enrichment refresh now downloads the latest season game log and shot charts each time; completed per-game play-by-play and box scores are reused unless `--force` is supplied. Availability of NBA.com endpoints can vary; do not interpret a failed refresh as fresh data.

## 3. Review availability and minutes

Generate a CSV containing the players currently modeled:

```cmd
.venv\Scripts\python.exe -m nba_sim simulate --home NOP --away SAC --season 2026-27 --event-id EVENT_ID --write-lineups data\manual\nop-sac.csv
```

This command writes a file and stops; it does not simulate games or overwrite an existing lineup file. Open `data\manual\nop-sac.csv` in Excel or another CSV editor. Save it as CSV after editing.

Check the latest [official NBA injury report](https://official.nba.com/nba-injury-report-2026-27-season/) and official team lineup announcements near tip-off. The CSV is a review of the modeled roster, not a guarantee that all current roster members are present. If a needed player is absent, refresh the current-season data; do not invent their historical statistics.

| Column | What to enter |
|---|---|
| `event_id`, `team`, `player_id`, `player_name` | Leave these identifiers unchanged. |
| `status` | `active`, `out`, or `questionable`. The default `unreviewed` needs your attention. |
| `suggested_minutes` | Reference only: the model's starting estimate. Editing this does not change the simulation. |
| `expected_minutes` | Optional target, such as `28`. Leave blank to distribute remaining minutes automatically. An out player must have blank or zero minutes. |
| `starter` | Optional `true` or `false`; blank retains the observed starter estimate. No more than five per team. |
| `reviewed_at` | Actual review date/time with timezone, e.g. `2026-10-30T19:00:00+01:00`. Replace the example with the time you really checked. |
| `source` | Link or clear reference to the report/announcement you checked. |

Every modeled player must have a row, including out players. A questionable player remains in the simulation as a scenario assumption, but unresolved status blocks all review candidates for that run. Resolve the status when confirmed, or make separate clearly sourced scenarios.

Out players are removed **after** historical blending. Explicit minute targets are preserved and the remaining allocation is adjusted to 240 regulation minutes. If you fix every player's target, the team total must equal 240. At least five players must remain. Targets describe expected regulation minutes, not hard playing-time caps: actual simulated minutes vary with lineup scheduling and overtime. Known strict playing-time restrictions therefore still need careful interpretation.

The availability review must be within 12 hours for review candidates. The program records your source and timestamp but does not independently verify them. A file is tied to one fixture; generate a new one for the next game.

## 4. Get current prices and run

```cmd
.venv\Scripts\python.exe -m nba_sim refresh-odds --home NOP --away SAC --event-id EVENT_ID --include-player-props
.venv\Scripts\python.exe -m nba_sim simulate --home NOP --away SAC --season 2026-27 --event-id EVENT_ID --lineups data\manual\nop-sac.csv --simulations 5000 --seed 42
```

Player props use additional credits. To restrict the downloaded prices to bookmakers you can use, add `--bookmakers bookmaker_key` to `refresh-odds`; use the provider's bookmaker key, not a display name. Changing bookmaker or region filters changes the available markets. A normal odds refresh replaces the odds cache, so a later refresh without player props will remove the previous prop quotes.

The report calculates each quoted side of moneylines, spreads, totals, and player points/rebounds/assists. It shows win and push probabilities, decimal odds, fair odds, same-book no-vig comparisons where both sides have matching lines, and expected profit per unit staked.

For example, a modeled 55% win chance at decimal odds 2.00, with no pushes, implies `0.55 × 1 − 0.45 = 0.10` units expected profit per unit staked. That is **10% modeled EV**, not a guaranteed 10% return.

Whole-number ties are pushes and refund the unit stake. Full-game markets are assumed to include overtime. Bookmaker participation, void, and settlement rules must match these assumptions. No parlay probabilities or stake recommendations are calculated.

## 5. Read the review list

A quote can appear as a **review candidate** only when:

- The fixture has not started and is within the next 24 hours.
- Both the downloaded cache and the quote's own timestamp are no older than 60 minutes.
- All modeled players have a sourced availability review within 12 hours, with no unresolved status.
- Current-season aggregate data and enrichment are no older than 48 hours, with rotation coverage for both teams.
- At least 500 simulations were run using the possession model.
- Modeled EV is at least 2%, and a conservative lower bound for simulation sampling error is above zero.

The quote-age and EV thresholds are configurable with `--max-odds-age-minutes` and `--min-ev`. They are screening rules, not validated optimal settings. Freshness timestamps mean download/review time; they do not guarantee the upstream feed contains every newly completed game.

The lower bound only addresses Monte Carlo randomness. It does not cover incorrect injury assumptions, model bias, weak player samples, or searching many offers. All runs retain `validation_status: unvalidated`; the software never declares a proven betting edge automatically.

Open the new timestamped folder under `outputs`:

- `market_probabilities.csv`: every supported quote, probability, price, EV, and any blockers.
- `betting_analysis.json`: the full analysis, skipped offers, thresholds, and assumptions.
- `run_metadata.json`: event, data timestamps, settings, and a snapshot of your lineup review.
- Existing score, player-distribution, and representative-game files remain available.

No matching upcoming event means no betting analysis. Unknown/excluded/ambiguous player names are skipped instead of assigned zero statistics. A simulation remains runnable without a reviewed lineup, but its offers will not pass the review checks.

## 6. Measure whether the predictions actually help

After games finish, download final team scores and evaluate saved predictions:

```cmd
.venv\Scripts\python.exe -m nba_sim refresh-results
.venv\Scripts\python.exe -m nba_sim evaluate
```

`refresh-results` uses the Odds API scores endpoint, consumes its API credits, and retrieves completed scores from the past three days. Run it regularly; saved results accumulate locally. It does not require historical-odds access.

For player results, open the completed game's NBA.com page and copy its ten-digit NBA game ID. This is different from the Odds API event ID. Supply it with the saved prediction folder:

```cmd
.venv\Scripts\python.exe -m nba_sim record-result --run "outputs\YOUR_RUN_FOLDER" --nba-game-id NBA_GAME_ID
.venv\Scripts\python.exe -m nba_sim evaluate
```

`record-result` downloads the final NBA box score without Odds API credits. It checks final status, home/away teams, scheduled time, and player-point totals before recording results. Do not pass a different game if the matching game is postponed; the command rejects mismatches.

Evaluation reports score error, probability accuracy (Brier score and calibration groups), range coverage, and player errors when available. It also computes **hypothetical** one-unit returns under a fixed rule: the first saved pregame run for each event/model/version, choosing its highest-EV eligible quote per market/player. Later runs do not replace losing forecasts or increase the sample count. All offers, including losses, are exported.

Player props remain pending until player results are recorded. A player who did not participate is treated as a void, not an under win. Other bookmaker-specific void rules are not modeled. Model versions are evaluated separately, and selections within a game are correlated. Paper returns do not establish that those prices were executable or that a small positive sample will persist.

This evaluates predictions made **before upcoming games**, without rerunning those games with future data. The older `backtest` command remains a separate evaluation of simpler score estimators.
