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
connection is healthy. Market selection refreshes every 10 minutes. Dashboard
discovery accepts markets with at least 10 minutes until close; use
`--min-close-minutes` to change this (minimum three). Known scheduled closes
pause forecasts 90 seconds before closing. The collector's one-hour discovery
default is unchanged. Each rescan reconnects and needs a fresh minute of history.
Use `--rescan-seconds` to change the rescan interval. These policies do not make
the models validated on near-close markets; evaluate that regime separately.

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

## Close-time metadata and existing recordings

New collector session CSVs include `scheduled_close_time`,
`close_time_observed_at`, and `seconds_to_scheduled_close`, populated from the
existing background status reviews. Earlier rows remain blank until metadata is
observed; no historical rows are rewritten and no additional per-tick requests
are made. Existing seven-input models ignore the added columns. To admit markets
with ten minutes until close, set `COLLECTOR_MIN_CLOSE_MINUTES=10` for a deliberate
collector restart; the default is still 60. See
`reports/near_close_collection_handoff.md` before changing the other computer.

Existing features can be joined to public live/historical market schedules for
a descriptive coverage audit:

```powershell
python -u -m src.backfill_close_metadata --output artifacts/near_close_backfill_NEW --max-markets 40
```

This queries development-market metadata only and discards settlement/results
and price fields. Retrospective schedules are explicitly unverified as-of the
original observations; this audit does not authorize them as causal model inputs.
Missing second-level book observations cannot be reconstructed from close times.

For a complete, resumable coverage audit of the frozen development snapshot:

```powershell
python -u -m src.audit_near_close_coverage --output artifacts/near_close_full_NEW
```

Rerun the same command to resume cached schedules. The audit reuses the initial
40-market cache, limits public API traffic to six requests per second across at
most four workers, validates source hashes, and excludes reserved sessions/time
ranges. Outputs include `status.json`, `row_coverage.csv`, `window_coverage.csv`,
retrospective window annotations, `summary.json`, and `REPORT.md`. Windows reuse
the audited 120-second-spaced development cohort; spacing does not eliminate
correlation between related markets. No model training or collector restart is
performed. Give `reports/collection_computer_near_close_prompt.md` to Codex on
the collection computer for the prepared rollout workflow.

## Model explanations and training traces

Click a model row in the live dashboard for its frozen-observation explanation.
The panel shows exact prediction calculations, saved training replay where
available, and a model-specific "How this model works" visualization.
Monte Carlo plays actual diverging simulation paths and random increments;
trees branch through their visited decisions; forests combine tree outputs;
boosting adds stage corrections; Ridge uses a signed contribution waterfall;
SGD replays recorded updates; GRU shows a labeled recurrent architecture/input
sequence schematic; the mean baseline shows aggregation into recorded means.
Unrelated models no longer receive a generic gradient-descent animation.
SGD shows actual recorded mini-batch updates; GRU shows actual Adam epoch/probe
observations. A two-dimensional projection is not the full loss landscape.
Linear contributions, tree decision paths, forest outputs and boosting stages
reconcile with the model's raw forecast, calibration and clipping.

Monte Carlo displays real seeded simulation paths, sampled historical increments,
return distributions and convergence. Its download contains all 2,048 paths and
draw indices; scenario dispersion is not a calibrated confidence interval.
Explanations load lazily at `/api/explanations/{forecast_id}/{model}` with a
30/60-second horizon. Forecast provenance is frozen; expired observations return
an explicit error. The 240-observation cache bounds retained input history.

The audited seven-input run is selected for live inference; eight-input timing
models remain offline comparisons. Feature-schema mismatches are rejected.
The new instrumented SGD procedure is versioned, not a claim of exact equivalence
to the older sklearn optimizer. Existing models and reserved holdouts are retained.

The paired comparison uses identical verified-timing fit/calibration/validation
identities for seven/eight inputs. Its optional `--exclude-near` supplement removes
only fitting examples with more than 10 and at most 60 minutes remaining, keeping
the common calibration/validation populations. Calibration still includes that
band, so this is fit-only exclusion, not a model never exposed to near-close data.
The supplemental artifacts are separate from the original paired results:

```powershell
python -m src.timing_comparison --base artifacts/model_lab_traced_20261109
python -m src.timing_comparison --base artifacts/model_lab_traced_20261109 --exclude-near
```

Verification commands:

```powershell
python -m pytest tests -q -p no:cacheprovider
node --test tests/frontend_explanations.test.cjs
```

The frontend tests exercise DOM interactions, not visual browser layout. No
profitable strategy has been demonstrated by the updated comparisons.

## Hourly readiness audit

Windows task `KalshiResearch-ReadinessHourly` runs the offline readiness audit
hourly as the logged-in research user, using hidden project `pythonw.exe`.
It remains independent of Codex and reads synced CSVs without network requests.
The task does not wake the PC or run while the user is logged out; start-when-
available allows a missed run to catch up. Power and Syncthing settings stay intact.

Read `artifacts/readiness/LATEST.md`, `status.json`, and `last_run.json`. Reports
are replaced atomically; the incremental cache avoids rereading unchanged files.
The newest file and files modified within two minutes are skipped. Errors and
stale data prevent a ready screen. No price-change targets or reserved outcomes
are calculated. No collector processes, model files, or recordings are modified.

```powershell
# Manual run:
.\.venv\Scripts\python.exe -m src.readiness_audit
# Register/repair the reviewed hourly task and run it once:
.\scripts\register_readiness_audit.ps1
# Stop future automatic runs:
Disable-ScheduledTask -TaskName 'KalshiResearch-ReadinessHourly'
```

The pilot screen asks for 1,000 valid near-close histories, 50 sessions, 20 event
proxies and three UTC dates, with zero metadata violations/file errors and recent
completed data. These are transparent heuristic review thresholds, not a power
calculation or proof of model quality. All 60 observations must contain schedules
known by their own timestamps. Signals are spaced by 120 seconds within each
market/session. Event grouping is a ticker-prefix proxy. A ready screen triggers
no training; chronological split design and related-market dependence still need
review. A Windows file lock and task IgnoreNew policy prevent overlapping runs.

## Tests and collection

```powershell
python -m pytest -q -p no:cacheprovider
```

Live authentication is an explicitly enabled integration test and is skipped by
default. Research scripts do not contact Kalshi or submit orders. The separate
read-only collector runs on another computer; only its data folder is synced.
Code changes here do not update that machine automatically.

The previous phase-based experiments and frozen artifacts are retained.

## Return-size accuracy study

The offline accuracy pipeline compares the original seven features with verified
schedule/depth/spread features and 180-observation context. Ridge, SGD, gradient
boosting and GRU settings are selected using three chronological development
folds. Mean, tree, forest and separate 60/180-observation Monte Carlo controls
remain available. SGD/GRU also compare a conservative target with interval-Huber
training; unknown returns remain intervals rather than invented point labels.
See [the study protocol](reports/accuracy_improvement_protocol_20261109.md).

The study uses a separately frozen source and a new final period; it does not
evaluate reserved v2/v3 outcomes. Choices and refit models are frozen before final
labels are opened once. A consumed final period cannot serve as fresh evidence
for later tuning. Do not start a second experiment in the active output folder.

Read `artifacts/accuracy_experiment_20261109/status.json` or tail
`artifacts/agent_status/research.log` for actual training progress. After the run
has written `final_completed.json`, the following commands independently audit
the artifacts and generate a report; they do not retrain or select a new winner:

```powershell
.\.venv\Scripts\python.exe -m src.accuracy_audit --output artifacts/accuracy_experiment_20261109 --allow-final
.\.venv\Scripts\python.exe -m src.accuracy_reporting --output artifacts/accuracy_experiment_20261109
```

Before completion, omit `--allow-final` for a development-only audit. The report
generator requires a passed final-scope audit. Final outputs include observed
and closed-only return errors, unknown-outcome coverage, interval diagnostics,
seed robustness, paired event-proxy bootstrap intervals and quote-based execution
scenarios. Improved return error can reflect spread/cost mechanics and does not
establish direction accuracy, confirmed fills or trading profit. New feature
schemas remain offline; the live model pointer is unchanged.
See `reports/collector_status_fix_rollout.md` for the separately prepared collector
status-scheduling fix. Research scripts never commit or push automatically.
