# Local NBA Monte Carlo Simulation and Basketball Analytics Engine

A Windows-first Python CLI that turns locally cached NBA observations into an event-driven, possession-style game simulation. It supports any two team abbreviations present in the selected dataset; OKC–SAS is an example, not a fixed matchup.

The program keeps real, user-supplied, and synthetic data paths separate. It never replaces a failed NBA.com request with invented observations. Demo output is prominently labelled `DEMO DATA`.

## What is implemented

- NBA.com aggregate team, advanced-team (including pace), and player refresh through `nba_api`.
- Explicit NBA.com enrichment refresh for `ShotChartDetail`, `PlayByPlayV3`, and recent `BoxScoreTraditionalV3` rotations.
- Timestamped raw JSON preservation plus normalized local CSV caches.
- A documented manual-CSV provider for API outages or curated inputs.
- Validation for schemas, IDs, duplicates, negative values, impossible makes/attempts, missing teams, and empty seasons.
- Completed-season, optional older-season, and sample-size-weighted current-season windows.
- Verified recent starters/minutes blended with the longer-term rotation baseline, with exactly 240 regulation player-minutes per team.
- PBP-driven shooter involvement and assisted-shot passer-to-shooter tendencies with sample-size shrinkage.
- Player-specific shot-zone rates and empirical coordinates sampled from each shooter's observed shot chart.
- Event simulation for turnovers, field-goal attempts, five shot zones, makes/misses, assists, shooting fouls, free throws, offensive/defensive rebounds, retained possessions, quarters, and overtime.
- Reproducible child seeds for each Monte Carlo game and exact replay of the representative game.
- A deliberately simpler score-level baseline model for comparison.
- Rich console reporting and the five required CSV/JSON exports.
- Walk-forward team-score backtesting that uses only rows strictly earlier than each holdout game.
- Automated tests for data, cache behavior, shrinkage, probabilities, minutes, event accounting, overtime, reproducibility, demo labelling, and future-data leakage.

The simulator now supports upcoming fixtures, reviewed availability/minutes, bookmaker-specific probabilities and expected value, and evaluation of saved pregame predictions. Follow the [upcoming-game workflow](BETTING_WORKFLOW.md) for CMD commands and the limitations of each step. It does not place bets or claim a proven betting advantage. More simulations reduce Monte Carlo sampling noise; they do not repair weak data or assumptions.

## Windows setup

Open PowerShell in this project directory.

```powershell
python --version
```

Python 3.11 or newer is required. This project was tested with Python 3.13.5.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The execution-policy command affects only the current PowerShell process. If activation is not permitted, prefix commands with `.\.venv\Scripts\python.exe` instead.

Run the tests:

```powershell
python -m pytest
```

## Run on another Windows PC

Install Python 3.13 and Git, then open PowerShell and run:

```powershell
git clone https://github.com/Johnson12344/nba-court-sim.git
cd nba-court-sim
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m nba_sim simulate --home OKC --away SAS --season 2026-27 --mode demo --simulations 100 --seed 42
```

The repository excludes downloaded data, generated reports, and API keys. For real simulations, copy the contents of this PC's `data/processed/` folder into the same folder in the new clone, or download the NBA data:

```powershell
.\.venv\Scripts\python.exe -m nba_sim refresh-data --season 2025-26
.\.venv\Scripts\python.exe -m nba_sim refresh-data --season 2026-27
```

If downloading fresh data, follow the enrichment commands below to add shot charts, play-by-play, and recent rotations for your teams. Set the odds key separately in the new PC's terminal, refresh the matchup's lines, and run:

```powershell
$env:THE_ODDS_API_KEY="your_key_here"
.\.venv\Scripts\python.exe -m nba_sim refresh-odds --home OKC --away SAS --include-player-props
.\.venv\Scripts\python.exe -m nba_sim simulate --home OKC --away SAS --season 2026-27 --simulations 5000 --seed 42
```

Replace the matchup as needed. Keep the key out of source files; the application reads it from the terminal environment, not a `.env` file. Set it again when opening a new terminal. Copying the existing NBA cache preserves its original data freshness; refresh it when needed. Do not copy `.venv` between PCs.

## Quick start: offline demo

```powershell
python -m nba_sim simulate --home OKC --away SAS --season 2026-27 --mode demo --simulations 100 --seed 42
```

The demo uses fictional `OKC Demo Player ...` and `SAS Demo Player ...` observations. It tests the software without making a factual statement about either NBA team. The built-in demo has only those two teams; real and CSV modes support every team present in their data.

## Real NBA.com data

Downloading is always explicit. Refresh the completed baseline first:

```powershell
python -m nba_sim refresh-data --season 2025-26
python -m nba_sim validate-data --season 2025-26 --data-source nba
python -m nba_sim refresh-data --season 2026-27
python -m nba_sim refresh-enrichment --season 2025-26 --teams OKC SAS --games-per-team 20 --rotation-games 3
python -m nba_sim refresh-enrichment --season 2026-27 --teams OKC SAS --games-per-team 10 --rotation-games 3
```

`refresh-data` caches aggregates. `refresh-enrichment` separately caches player shot charts, the selected teams' recent PlayByPlayV3 games, and recent verified box-score rotations. The league game log and season shot charts are refreshed on every call; existing per-game files are reused unless `--force` is supplied. Increase `--games-per-team` up to 82 for deeper PBP coverage; start with 10–20 while developing.

Then run any cached matchup by abbreviation:

```powershell
python -m nba_sim simulate --home OKC --away SAS --season 2026-27 --simulations 5000 --seed 42
python -m nba_sim simulate --home BOS --away LAL --season 2026-27 --simulations 5000 --seed 7
```

## Optional sportsbook market comparison

For a single command that updates inputs and prints the full matchup report, use:

```cmd
.venv\Scripts\python.exe -m nba_sim analyze --home NOP --away SAC --season 2026-27
```

It reuses recent downloads, fetches missing/stale data and available odds/player props, and runs 5,000 simulations. Add `--cached` to use only the data already downloaded. See the [full workflow](BETTING_WORKFLOW.md) for availability review and other options. The report includes the combined total and player points, rebounds, assists, and minutes.

Create an API key at [The Odds API](https://the-odds-api.com/). Keep the key outside the project files. In Command Prompt, set it for the current terminal window and download current NBA lines:

```cmd
set THE_ODDS_API_KEY=your_key_here
python -m nba_sim refresh-odds --home OKC --away SAS
```

To also download available player points, rebounds, and assists lines for that matchup:

```cmd
python -m nba_sim refresh-odds --home OKC --away SAS --include-player-props
```

PowerShell uses `$env:THE_ODDS_API_KEY="your_key_here"` instead of `set`. The core refresh requests moneyline, spread, and game-total markets. Player props are opt-in because each additional market consumes API credits.

The cache is written to `data/processed/odds_nba.json`; timestamped raw responses are retained in `data/raw/`. The API key is never written to either file. A normal simulation automatically attaches an exact home/away market match and displays simulation-versus-market context. Use `--no-odds` to suppress it. Market differences are diagnostic context, not betting recommendations.

Runs of 500 or more automatically use up to four local worker processes. Set `--workers 1` for strictly single-process execution or `--workers 2` to cap CPU use. Seeded results remain ordered and reproducible across worker counts.

Simulation count is intentionally capped at 10,000. The default remains 5,000 because more draws reduce Monte Carlo noise but do not improve the underlying basketball model.

For target season `2026-27`, the CLI automatically uses `2025-26` as the completed baseline. If a validated `2026-27` cache exists, it blends that season-to-date evidence using `games / (games + prior_games)`; otherwise the report states that it used only the completed baseline. Override the foundation when needed:

```powershell
python -m nba_sim simulate --home NYK --away MIA --season 2026-27 --baseline-season 2025-26 --history-season 2024-25 --simulations 5000 --seed 11
python -m nba_sim simulate --home NYK --away MIA --season 2026-27 --no-current-season --simulations 5000 --seed 11
```

Refresh every requested historical season before using it. A simulation does not initiate downloads.

Use the simpler comparator:

```powershell
python -m nba_sim simulate --home OKC --away SAS --season 2026-27 --model baseline --simulations 5000 --seed 42
```

Baseline mode intentionally produces no player distributions or event narrative because it samples team scores rather than possessions.

## Manual CSV mode

The schemas are documented in [`data/manual/README.md`](data/manual/README.md). Place these files there:

- `team_stats.csv`
- `player_stats.csv`

Both may contain multiple seasons. Then run:

```powershell
python -m nba_sim validate-data --season 2025-26 --data-source csv
python -m nba_sim simulate --home DEN --away MIN --season 2026-27 --data-source csv --simulations 5000 --seed 42
```

Manual source provenance cannot be independently verified by the application, so the report says that explicitly. Optional player shot-zone fields are `rim_fga/rim_fgm`, `paint_fga/paint_fgm`, `midrange_fga/midrange_fgm`, `corner_three_fga/corner_three_fgm`, and `above_break_three_fga/above_break_three_fgm`.

## Outputs

Each run creates a unique UTC timestamped directory under `outputs/` containing:

- `summary.json` — team score and quarter distributions.
- `game_scores.csv` — one row per simulated game with its child seed.
- `player_distributions.csv` — player distributions, sample sizes, starter status, shooter-selection source, and shot-location source.
- `representative_game.json` — selected generated game, box leaders, quarter scores, and generated event sequence. Enriched shot events include player-specific sampled coordinates, distance, zone, and evidence source.
- `run_metadata.json` — source, refresh time, model version, seed, windows, limitations, and reproduction command.

The representative game minimizes standardized distance from the median total and median home margin. It is replayed using its original child seed with event capture enabled; the text is not invented after the fact.

## Backtesting

Place `historical_games.csv` in `data/manual/` with:

```text
game_id,game_date,season,home_team,away_team,home_score,away_score
```

Then run:

```powershell
python -m nba_sim backtest --season 2025-26 --games 100
```

Every holdout uses only games with dates strictly before its game date. The exported report compares a league-mean baseline with a pregame team for/against estimator and reports team-score MAE and 10th–90th percentile coverage. This initial evaluator does not claim player-stat backtesting without player box-score history.

## Model details and honest limitations

Observed aggregate fields drive pace, free throws, rebounds, turnovers and stable priors. When enrichment is cached, PlayByPlayV3 shot events influence shooter selection, assisted-shot descriptions influence passer-to-shooter selection, ShotChartDetail drives each player's zones and empirical coordinates, and recent box scores drive starters/minutes. All noisy rates are shrunk toward longer-term or league priors. Model constants live in `config/settings.yaml`.

NBA.com PlayByPlayV3 and ShotChartDetail do not establish the on-ball defender on every shot, screen actions, switches, or exact defensive assignments. The simulator therefore applies neutral zone-defense factors unless supported defensive fields are supplied and never presents generated defenders as observed assignments. The generated event sequence samples from observed player tendencies; it is not a reconstruction of an actual game.

Current-season recent box scores identify observed starters and minutes and are blended with season evidence before normalization to 240 minutes. They still do not constitute an injury report or confirmed future lineup. For a specific future game, provide verified expected-minute and availability overrides rather than interpreting the model rotation as confirmed.

## Troubleshooting

**NBA.com timeout or blocked request**

Retry later and avoid rapid repeated refreshes. Successful raw responses remain in `data/raw/`. Simulation never silently falls back to demo data. Use documented manual CSVs if the endpoint remains inaccessible.

**No cached NBA.com data**

Run `refresh-data` for the completed baseline shown in the error. The simulator intentionally does not download during a run.

**Current season missing**

This is not fatal when the completed baseline exists. The console and metadata state that current-season evidence was unavailable. Refresh the current season when enough games exist.

**Team not found**

Use official three-letter abbreviations and validate the season. The error lists available abbreviations. Matchups are not hardcoded.

**Package installation fails**

Confirm that the virtual environment is active, upgrade pip, and retry `python -m pip install -e ".[dev]"`. Corporate proxies may block PyPI or NBA.com independently.

**PowerShell displays odd dash characters**

Use a UTF-8 terminal (`chcp 65001`) or Windows Terminal. Exported JSON and CSV files are UTF-8.

## Project layout

```text
config/                 model settings and aliases
data/cache/             local cache support
data/raw/               immutable timestamped NBA.com responses
data/processed/         normalized official caches
data/manual/            user-supplied CSV inputs
outputs/                timestamped run and backtest exports
src/nba_sim/data/       providers, normalization, validation, cache
src/nba_sim/features/   profiles, shrinkage, historical blending
src/nba_sim/simulation/ lineups, possessions, games, Monte Carlo
src/nba_sim/reporting/  summaries, console, narrative, exports
src/nba_sim/evaluation/ walk-forward backtesting and metrics
tests/                  automated tests
```

## Commands verified in this environment

```powershell
python -m pytest
python -m nba_sim refresh-data --season 2025-26
python -m nba_sim refresh-data --season 2026-27
python -m nba_sim refresh-enrichment --season 2025-26 --teams OKC SAS --games-per-team 3 --rotation-games 2
python -m nba_sim refresh-enrichment --season 2026-27 --teams OKC SAS --games-per-team 10 --rotation-games 2
python -m nba_sim validate-data --season 2025-26 --data-source nba
python -m nba_sim simulate --home OKC --away SAS --season 2026-27 --mode demo --simulations 100 --seed 42
python -m nba_sim simulate --home OKC --away SAS --season 2026-27 --simulations 100 --seed 42
python -m nba_sim simulate --home OKC --away SAS --season 2026-27 --simulations 5000 --seed 42
```

The official refresh was accessible during development. The enriched 2026-27 OKC/SAS verification loaded 471 shot-chart attempts, 1,440 PBP events, and 85 recent rotation rows. Endpoint availability can change independently of this code.
