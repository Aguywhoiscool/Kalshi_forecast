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
