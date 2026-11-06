# Spread forecasting feasibility — November 4, 2026

This is an offline feasibility analysis, not a trained forecasting model.
It assumes the proposed target is bid–ask spread widening, rather than a large
midpoint move. Both are legitimate but different prediction tasks.

## Why the old 10-cent filter is limited

A wide spread is not corrupt data. The original midpoint experiment treated
spread <=10 cents as a research-quality/liquidity heuristic, not a universal
exchange rule or empirically optimized cutoff. It removes valid observations
and reduces the experiment's coverage of illiquid conditions.

The old builder filters wide rows before constructing complete 60-history plus
10-future-observation windows. Consequently eligibility is conditioned partly on
future spread quality: windows crossing widening episodes are removed. The old
benchmark is a conditional development benchmark, not an unbiased simulation of
what an online entry-time spread filter would experience. Its predictors do not
receive future quotes, but its evaluation cohort is selected using future quote
conditions. Do not interpret it as evidence that spread-widening episodes cannot
be forecast. An execution evaluation should apply entry-time eligibility rules
and retain subsequent widening in outcomes/costs.

## Proposed tasks

1. Widening onset: current spread <=10 cents; any spread >10 cents during the
   following ten valid one-second observations.
2. Narrowing onset: current spread >10 cents; any spread <=10 cents in that horizon.
3. Future spread regression/state: predict spread at ten seconds or its change.

Training must include valid quotes regardless of spread width. A model for
onset should be scored only in its relevant current-state cohort; counting a
spread that is already wide as a newly predicted widening would be misleading.
Persistence/current-spread baselines are required. Rare onset events call for
average precision, precision/recall, probability calibration and alert burden;
plain accuracy can be dominated by no-event predictions.

## Development-only counts

Source: the same frozen corrected Model Lab snapshot. Original development time
boundaries and known session assignments were preserved. Reserved v2 sessions
and the known v3 holdout sessions were excluded before target construction.
Additional wide-only sessions were eligible only if entirely within an existing
development time range. Valid two-sided quotes and timing continuity remain
required. Missing quote intervals are not imputed or labeled as no-event cases.

| Cohort | Train | Validation |
|---|---:|---:|
| All valid windows, including wide spreads (overlapping) | 2,579,213 | 526,802 |
| Spaced windows | 43,602 | 9,084 |
| Currently <=10 cents, spaced | 40,481 | 8,435 |
| Widening within 10 seconds, spaced | 1,110 | 162 |
| Currently >10 cents, spaced | 3,121 | 649 |
| Narrowing within 10 seconds, spaced | 728 | 107 |

Widening onset prevalence in spaced initially-narrow validation examples is
about 1.92%. A model predicting no widening would therefore score about 98.08%
accuracy while detecting none of the events. Spacing prevents overlap inside a
market/session but does not make related market events independent.

There is enough observed variation for an exploratory benchmark, but 162 spaced
validation positives are limited evidence for cross-market generalization. The
new cohort is broader than the old midpoint benchmark and requires its own
clearly labeled results. Models have not yet been fitted to these targets.

## Reproduce

```powershell
python -u -m src.analyze_spread_transitions --output artifacts/spread_forecast_NEW_RUN
python -m pytest tests/test_spread_transition_labels.py -q -p no:cacheprovider
```

Full counts: `artifacts/spread_forecast_feasibility_20261104/summary.json`.
No collector changes, data repairs, trades, commits, pushes or reserved outcome
evaluation were performed by this analysis.
