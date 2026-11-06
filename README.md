# Kalshi Model Lab

An offline research project comparing machine-learning methods on collected
Kalshi order-book data. The focus is reproducible model comparison, including
prediction quality, training cost, and hypothetical bid/ask execution economics.
The existing GRU remains part of the project.

**Correctness audit:** The initial benchmark used a strict floating-point
threshold that incorrectly excluded some exact 0.5-cent moves. Its saved models
and scores are retained as legacy results, not a finalized comparison. The audit
is in `artifacts/benchmark_correctness_audit_verified/REPORT.md`. Active code now shares a
roundoff-safe label policy and rejects reuse/resume of legacy GRU runs. A fresh
corrected training run and independent benchmark audit completed on November 4,
2026 using the same frozen splits. See
[the checked results summary](reports/model_lab_corrected_results_20261104.md).
Do not silently change the reserved holdout boundaries.

## Models

- GRU neural network: full 60-step, seven-feature input sequence.
- Decision tree and random forest.
- Histogram gradient boosting.
- Logistic regression, including a version trained with stochastic gradient descent.
- Class-prior baseline to expose misleading accuracy on mostly unchanged prices.
- Monte Carlo: a stochastic historical-increment bootstrap, alongside the
  supervised models. For every validation example, it simulates 2,048 ten-step
  price paths using only that example's past 59 midpoint increments, clipping
  prices to [0,1] at every step. It fits no training labels. It assumes independent,
  locally representative increments; it does not simulate confirmed executions.

Gradient descent is a training method, not a separate model family. The initial
tabular models use 50 fixed causal summaries of the same 60-step history; their
representation differs from the GRU's sequence input and is disclosed in reports.

## Retrain with corrected labels

Reuse the original frozen data and exact split boundaries while fitting fresh
models and scalers. This workflow verifies independent dataset copies, trains the
GRU, runs all tabular models and Monte Carlo, and audits the resulting benchmark:

```powershell
python -u -m src.rerun_model_lab --source artifacts/v3_20261104_overnight
```

It creates a new `artifacts/model_lab_corrected_TIMESTAMP/` training directory
and sibling `..._benchmark/` report directory. It does not ingest newly synced
recordings, change splits, evaluate holdout outcomes, or reuse legacy checkpoints.
After successful completion `artifacts/model_lab_latest.json` points to the report.

## Run the comparison on an already corrected GRU run

From the project root with the existing virtual environment activated:

```powershell
python -m src.model_benchmark --source-run artifacts/YOUR_COMPLETED_CORRECTED_RUN
```

This requires a completed frozen v3 run with the current label policy, including its shards, split metadata,
GRU predictions, checkpoints and config. It uses the exact sampled GRU training
population (up to 300,000 windows) and the same non-overlapping validation
examples. Direction is trained/scored on meaningful moving examples; execution
signals use predictions only. Movement means an absolute midpoint change of at
least 0.5 cents at the 10-second horizon, computed in float64.

Results go to a new `artifacts/model_benchmark_TIMESTAMP/` directory, outside
Syncthing's shared data folder. Open its `index.html` directly in a browser for
interactive task/signal-fraction controls. The directory also contains
`REPORT.md`, `leaderboard.csv`, `execution.csv`, `metrics.json`, `protocol.json`,
model files, row identities, validation predictions, and `status.json`.
The report is standalone and requires no web server, account, or external assets.

## Evaluation boundaries

- Chronological splits, disjoint sessions, 70-second embargo.
- Reserved v2 test period/sessions excluded; v3 holdout outcomes remain untouched.
- Scaling is fitted only to training data. Tree/SGD early stopping does not create
  a random internal validation split.
- Fixed initial settings and seed; no automated hyperparameter search.
- The reused GRU checkpoint was selected on development validation. Benchmark
  rankings are exploratory, not unbiased held-out performance estimates.
- Non-overlapping windows within a market can still be correlated across events.
- Gross execution assumes hypothetical taker entry/exit at recorded quotes.
  Fees, slippage, latency, fill capacity and impact are excluded. Positive gross
  results would not establish a profitable executable strategy.

## Large-swing entry/exit experiment

This separate experiment retains valid wide spreads and predicts at least five
cents of quoted gross return on YES or NO over 30/60 seconds, with no-trade as an
option. It uses a one-observation entry delay, one-contract depth checks, and a
predeclared four-cent additional round-trip cost scenario. Other cost/delay and
first-crossing bracket exits are sensitivity checks, not parameters selected on
validation results. Historical market-specific fees are not available in the
feature recordings, so cost buffers must not be presented as actual fee-adjusted
profits. Missing future quotes remain censored; low exit depth is unresolved and
conservatively marked to zero contract value, not treated as a confirmed fill.

```powershell
python -u -m src.analyze_swing_opportunities --output artifacts/swing_opportunities_NEW
python -u -m src.swing_strategy_experiment --diagnostics artifacts/swing_opportunities_NEW --output artifacts/swing_strategy_NEW
```

The first script diagnoses opportunities using explicitly labeled hindsight
upper bounds. The second trains new models, chooses thresholds from a purged
chronological calibration subset of original training, simulates entry/exit
rules, and independently audits the saved results. It never uses peak exits or
future directions to select trades. Neither script evaluates the reserved v2/v3
holdouts or submits orders.

The initial local run is `artifacts/swing_strategy_20261104/index.html`.
It did not establish a robust profitable strategy. Sparse positive results and
SGD iteration-limit warnings require care; see its report and `audit.json`.

## Past-only swing return experiment

The corrected experiment selects entries from the preceding 60 observations,
without requiring future quote coverage. Missing entry or exit observations stay
in the evaluation with explicit payoff bounds. Models predict conservative net
returns for YES and NO; a fixed expected-return margin controls entry, rather
than a probability of a large win. The primary cost scenario is an assumed
four-cent round-trip buffer, not measured historical fees.

```powershell
python -u -m src.swing_return_experiment --output artifacts/swing_returns_NEW
```

This compares mean, ridge, SGD, decision tree, random forest, gradient boosting,
GRU and Monte Carlo forecasts against no trade at 30/60 seconds. Source arrays
are reconstructed from frozen CSVs before training and during the final audit.
Reports include missing outcomes and block subsequent entries when a position's
outcome is uncertain. Reserved v2/v3 outcomes remain excluded. Outputs are local
under the new artifact directory, with progress in `status.json`.

The earlier classifier experiment above is retained for comparison; its cohort
conditions on future quote availability and its win probabilities do not measure
expected net returns. Its results should be interpreted with those limitations.

To use newly collected feature files, freeze a separate development source first:

```powershell
python -u -m src.expanded_swing_source --output artifacts/swing_source_all_NEW
python -u -m src.swing_return_experiment --source artifacts/swing_source_all_NEW --output artifacts/swing_returns_all_NEW
```

This considers every local feature CSV, excludes reserved v2/v3 sessions and time
ranges, and skips recent, active, conflicting or changing files. The source has a
chronological 80/20 session split with a 70-second embargo; calibration remains
inside training. Its validation dates differ from the older experiment, so score
differences alone cannot isolate the benefit of additional training data. Both
sources and their reports remain separate. Remote Syncthing completion is not
assumed from local file stability.

## Live prediction dashboard

```powershell
.\.venv\Scripts\python.exe -m src.live_dashboard
```

Open `http://127.0.0.1:8765`. The local dashboard connects directly to Kalshi's
authenticated order-book WebSocket and displays all eight model forecasts at 30/60 seconds, midpoint and spread
history, YES/NO net-return comparisons, and a rolling prediction log. It defaults
to the audited run in `artifacts/swing_return_latest.json`. To choose another run:

```powershell
python -m src.live_dashboard --run artifacts/swing_returns_all_20261106 --port 8765
```

The backend samples reconstructed books once per second and updates inference
every five seconds using saved weights, scalers and calibration corrections.
It discovers up to eight open markets, or accepts `--tickers TICKER1 TICKER2`.
The `.env` credentials stay on the server and are used only for market-data
authentication. No account, balance, fill or order channel is subscribed.
Reconnections and subscription confirmations reset books and history; sequence
gaps force a fresh connection. Forecasting needs about one minute of continuous
history after connecting. Unchanged books remain valid between deltas while the
connection is healthy. Market selection refreshes every 30 minutes.

For the previous Syncthing mode, use `--feed files`. This monitors up to eight
markets from the six newest CSVs, reads bounded tails and ignores partial rows.
Both modes require a valid continuous 60-row history inside one session and
exclude reserved research periods. Observations older than 30 seconds pause
new forecasts; cached results are marked stale. The direct WebSocket mode keeps
observations in memory and never writes to the shared data folder or changes
the collector on the other computer.

The service binds only to localhost and uses the existing signing helper.
There are no order routes or trading controls. Forecasts are conservative return
estimates in cents, not win probabilities. The primary 4-cent cost assumption is
not measured fees, and these models have not demonstrated a profitable strategy.
Prediction history is bounded and kept in memory; it resets when the server exits.

## Tests and collection

```powershell
python -m pytest -q -p no:cacheprovider
```

Live authentication is an explicitly enabled integration test and is skipped by
default. Research scripts do not contact Kalshi or submit orders. The separate
read-only collector runs on another computer; only its data folder is synced.
Code changes here do not update that machine automatically.

The previous phase-based experiments and frozen artifacts are retained.
See `reports/collector_status_fix_rollout.md` for the separately prepared collector
status-scheduling fix. No Git commit or push is part of the benchmark workflow.
