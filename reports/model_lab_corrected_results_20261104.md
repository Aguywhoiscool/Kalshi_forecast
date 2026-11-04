# Corrected Model Lab benchmark — November 4, 2026

All eight methods completed; the independent correctness audit passed. These
are exploratory development-validation results, not an untouched test result.
No live orders were submitted. Training artifacts and source data remain local.

## Experiment

- Frozen input: 1,732 feature recordings / 5,371,883 rows, unchanged from the
  original v3 research snapshot. No new recordings were ingested for the rerun.
- Same chronological splits, disjoint sessions, and 70-second embargo.
- Supervised training: the same 300,000 sampled windows spanning 1,750 markets
  and 1,048 sessions; 55,571 moving examples for conditional direction.
- Validation: the same 8,058 non-overlapping windows within each market/session,
  spanning 428 markets and 233 sessions; 1,275 moving examples for direction.
- Training period: October 26–November 2 UTC. Validation: November 2–3 UTC.
- Input: 60 observations of bid, ask, spread, midpoint, bid/ask depth, imbalance.
- Target: midpoint movement of at least 0.5 cents after 10 seconds. Float64
  arithmetic with a 1e-9-cent roundoff tolerance includes exact threshold moves.
- The GRU sees the full sequence; tabular methods see 50 fixed causal history
  summaries. Monte Carlo simulates 2,048 bounded paths from historical increments
  without fitting labels. Representations and training budgets differ.
- The GRU uses up to eight epochs and validation-based checkpoint selection;
  tabular methods use fixed settings. One seed; no hyperparameter search.

## Results

Movement average precision is higher-is-better. Direction accuracy is conditional
on actual meaningful movement. The top-10% gross simulation ranks movement scores
and chooses direction from predictions, not future labels.

| Method | Movement average precision | Direction accuracy | Top-10% gross cents/contract |
|---|---:|---:|---:|
| Gradient boosting | 0.5607 | 65.96% | -3.3752 |
| GRU | 0.5573 | 61.33% | -3.3975 |
| Random forest | 0.5572 | 65.33% | -3.3776 |
| Decision tree | 0.5326 | 61.57% | -3.1808 |
| Monte Carlo bootstrap | 0.4989 | 43.69% | -3.6286 |
| Logistic regression, SGD | 0.4940 | 61.88% | -3.5255 |
| Logistic regression, L-BFGS | 0.4932 | 62.12% | -3.5280 |
| Class-prior baseline | 0.1582 | 51.22% | -2.8695 |

The leading movement scores are close; no significance or clear-winner claim is
made. All top-10% gross simulations lost money before fees, slippage, latency,
capacity constraints or market impact. Quote-based taker fills are hypothetical.
Signal fractions rank the whole validation period retrospectively; they are not
deployable online thresholds. Cutoff ties use uniform expected selection.
Related markets/events can remain correlated despite within-market spacing.

## Audit and preserved boundaries

The audit reproduced sample/source hashes, original v2 protection from timestamp
metadata, temporal/session separation, every validation history and execution
quote from hashed CSVs, saved model predictions, training-only scaler means and
counts, all reported classification metrics, and complementary YES/NO gross
P&L using independent Decimal arithmetic. Reserved v2 test outcomes and v3
holdout outcomes were not evaluated.

The legacy threshold had missed 4,866 moving training examples and 114 moving
validation examples. Its original checkpoints, predictions and numerical scores
were preserved and marked as legacy. Corrected models were fitted from scratch.
The old chronological tie-breaking behavior was also replaced and tested.

Validation was used for GRU checkpoint selection, so further development-period
replications and fixed-protocol holdout evaluation are needed before generalization
claims. Recommended next experiments: multiple seeds, day/market-group analysis,
and a training-size comparison while keeping reserved boundaries fixed.

## Local artifacts and verification

- Training: `artifacts/model_lab_corrected_20261104/`
- Interactive report: `artifacts/model_lab_corrected_20261104_benchmark/index.html`
- Audit: `artifacts/model_lab_corrected_20261104/audit/REPORT.md`
- Current verified run pointer: `artifacts/model_lab_latest.json`
- Offline suite: **151 passed, 1 skipped** (opt-in live authentication test).

Artifacts are excluded from Git. See the repository README for rerun commands;
a completed local frozen dataset is required. This summary is the portable
record of the completed experiment.
