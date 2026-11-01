# Learning advisor

The advisor runs automatically with a real `analyze` report when matching odds are available. It is a small local statistical learning model, requires no extra AI key or subscription, and never places bets. Existing injury, data-freshness and simulation-quality checks still apply.

Use your normal matchup command:

```cmd
.venv\Scripts\python.exe -m nba_sim analyze --home HOME_CODE --away AWAY_CODE --season 2026-27
```

Replace `HOME_CODE` and `AWAY_CODE` with the actual home and away team abbreviations. The report now starts with **Learning advisor**. Its output is one of:

- **NO BET**: nothing passed all the learning, availability, price and freshness checks. This is a normal outcome.
- **Paper watch only**: a simulation-only candidate to observe while collecting evidence, not a learned recommendation. The decision remains NO BET.
- **Candidate for review**: among qualifying positive-EV quotes, the selection with the highest estimated win probability, then higher EV. It is not described as a safe or guaranteed bet.

Win probability and value are different: a likely winner can still be overpriced. The advisor requires at least 55% estimated outright win probability, at least 2% adjusted modeled EV (or your higher `--min-ev`), and the existing Monte Carlo/data checks. These are engineering screening thresholds, not proven optimal betting rules. Win, loss and push probabilities are displayed separately.

Every simulation report ends with **Predicted winner** and **Estimated win probability** for both teams. Percentages below 50% are red, 50% through 80% yellow, and above 80% green. Showing both teams also shows the underdog's probability, which is normally below 50%. An equal probability produces "No clear favourite". These describe the chance of winning this particular matchup, not historical accuracy, the chance of predicting the exact score, or betting safety. A high percentage does not override NO BET.

The footer uses the learned **moneyline** estimate only when that model has passed preliminary validation and current data/quote checks pass. It uses a fresh home-side quote from the first bookmaker in alphabetical order, consistently with the training quote tie-break. Otherwise it clearly labels the simulation estimate as not yet calibrated; demo percentages remain labelled demo. A player-prop or spread probability is never presented as a team's win probability. Both percentages and their source are saved as `win_prediction` in `summary.json`.

## Memory and learning

Each saved pregame prediction, its selected training quotes and any advisor decision are remembered in `data/processed/advisor_memory.json`. Sources and the original run location remain attached. Memory persists after closing CMD. It is local to each PC, excluded from GitHub, and is not uploaded to an AI service. Keep backups of `outputs`, `advisor_memory.json` and `results_nba.json`. To move memory to a new PC, copy both JSON files into that PC's `data/processed` folder before running the advisor there; this replaces that PC's memory, so preserve its existing files first. There is no automatic cross-PC merge or synchronization.

The advisor imports compatible saved reports automatically. Predictions from different simulator versions remain separate. The simulator itself stays at model version 0.5.0 in this release because its numerical mechanics have not changed; the application is 0.6.1 and the advisor has its own version 1.0.0. Reports without saved bookmaker probabilities cannot be reconstructed into training examples using later odds.

For every event/model/version, only the **first saved pregame run** is used. Re-running a game with more seeds or more bookmakers cannot inflate its experience or replace a losing forecast. An opted-out `--no-advisor` run is not imported later. Demo and postgame predictions are excluded. Predictions with unresolved availability, stale quotes or other blockers are remembered but do not supply training observations.

For each market separately (moneyline, spread, total, points, rebounds, assists), the learner fits a weight between the simulation probability and the bookmaker's same-book no-vig probability:

`learned probability = weight × simulation + (1 − weight) × market`

The weight minimizes historical Brier loss. At most one training quote per market per game is chosen deterministically before consulting outcomes: home/Over side, the most widely quoted player/line, then stable tie breaks. Low-EV predictions and losses are retained; training is not restricted to previously attractive bets. Player markets pool evidence across players, so the fit is not a claim of player-specific reliability.

Each market needs **at least 100 training games plus 40 later validation games**. Outcomes used for training must have been recorded before the first validation prediction. The latest 40 eligible games are held out, and their outcomes never fit the weight used in that report. The weight must do no worse than the original simulation on those games and improve on the market benchmark by at least 0.002 Brier points. The training weight is used unchanged for the current prediction. Parameters, counts, benchmark scores, chronology and a training-data fingerprint are saved in `advisor.json`.

This is preliminary probability validation, not proof of positive betting returns or a formal confidence guarantee. It does not establish reliability for every player, line or probability range. Market conditions can change. There is no promise that more observations will produce a qualifying selection. Push probabilities remain simulation estimates; pushes and DNP voids are excluded from conditional-win calibration, while genuine losses are included.

## Automatic final results

On later online `analyze` runs, the advisor uses the downloaded league game logs to match past saved fixtures by teams, home/away order and Eastern-time game date. It fetches final NBA box scores, verifies final status, game ID, teams, timing and player-point totals, and records outcomes with their observation time. This uses NBA.com, not additional Odds API credits. It cannot learn an outcome that has not been verified.

Result checks normally occur at most once per six hours, with up to five box-score downloads per run. Unfinished, unmatched or unavailable games remain pending and are retried; they are not counted as losses. `analyze --refresh` forces a fresh check and current game logs. `analyze --cached` and `simulate` use cached results without network requests. Nothing runs while the program is closed.

View memory without running a simulation:

```cmd
.venv\Scripts\python.exe -m nba_sim advisor-memory
```

Check pending results using the game logs already on this PC:

```cmd
.venv\Scripts\python.exe -m nba_sim advisor-memory --refresh
```

The existing `refresh-results`, `record-result` and `evaluate` commands remain available for explicit result collection/evaluation. The advisor imports their results on its next run. Result corrections replace old settlements; a score correction invalidates player totals until verified again.

Memory reports separate hypothetical one-unit returns for learned review selections and paper-only watches. These do not represent actual wagers, account balances or executable prices. Existing `evaluate` output measures its own raw-EV paper strategy, separately from the advisor's selections. Shortlist alternatives are not combined into accumulators, and stake sizing is not provided.

Add `--no-advisor` to a run to skip the advisor and memory collection. Corrupted memory produces an unavailable/NO BET result; it is never silently reset. The saved run remains available for later recovery.

For background on assessing probabilities against observed outcomes and separating fitting from validation, see the [probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html). This implementation uses NumPy already installed with the simulator; it does not add scikit-learn or an LLM dependency.
