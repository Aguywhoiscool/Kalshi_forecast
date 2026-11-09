# Return-size accuracy study — 2026-11-09

## Scope

Implemented a reproducible offline study of quote-based YES/NO return-size
forecasts over 30 and 60 seconds. This task does not change the live model
pointer, place trades, restart the collector, or evaluate reserved v2/v3 outcomes.

The source snapshot contains 2,806 copied feature files and 8,868,787 rows.
Source-level embargo checks leave 2,805 eligible files. The development cohort
has 6,054 candidates with common 180-observation histories across 488 sessions
and 854 ticker-derived event proxies. This is a filtered paired cohort, not
8.9 million independent training examples. Verified schedule knowledge must
exist at every history row; missing historical timing is not imputed.

The final period was selected using timestamps strictly after the previous
source's development maximum plus 70 seconds. Three expanding chronological
development folds use separate sessions and inner calibration with 70-second
embargoes. Existing reserved outcomes remain excluded.

## Implementation and checks

Compared seven-feature 60-observation baselines with engineered 60/180-observation
variants, conservative and interval-Huber SGD/GRU pipelines, and separate genuine
60/180-observation Monte Carlo controls. Ridge, SGD, gradient boosting and GRU
use a staged feature/objective comparison followed by fixed parameter grids.
Mean, decision tree and random forest remain controls.

The complete registry is 90 configurations and 270 fold evaluations. Primary seed 42
determines selection; seeds 7 and 101 are robustness refits. Both primary winner
and strongest default baseline are chosen from development results before final
labels are opened. Model manifests and choices are hashed before final scoring.

Independent source review and actual development-artifact replay were performed.
The full Python suite passed 287 tests with one skipped; all 21 frontend tests
passed. Tests include numerical loss-gradient checks, actual tiny GRU fits,
causal feature/timing validation, chronological leakage checks, Monte Carlo
prefix/context behavior, audit tamper rejection and frozen-winner reporting.
Passing tests do not prove an absence of defects.

Source milestones: `eb32508` and `bc9a573`, both pushed to origin/main. Collected
data, trained binaries, logs and experiment artifacts are excluded from Git.

## Results and acceptance

Training and once-only final scoring completed at 16:03:08 UTC (11:03 AM EST).
All 90 configurations/270 development-fold evaluations and all 78 refits
completed. The independent reviewer reproduced the complete registry, staged
minima and frozen choices before final labels were opened. Both horizons selected
the original seven-feature 60-observation GRU with interval-Huber training;
engineered features and longer context did not win the GRU development comparison.
The strongest preselected default comparator was Monte Carlo with 180 observations.

Final assessment uses 1,413 candidates and 200 event proxies, from the 109-session
timestamp-reserved segment. Primary seed-42 results:

| Horizon | Selected GRU observed MAE | Default MC180 observed MAE | Relative gain | Paired 95% improvement interval | Observed sides |
|---|---:|---:|---:|---:|---:|
| 30 seconds | 1.5959 cents | 2.5289 cents | 36.89% | [0.8379, 1.0354] cents | 2,679 / 2,826 |
| 60 seconds | 2.1892 cents | 5.0921 cents | 57.01% | [2.7208, 3.0792] cents | 2,504 / 2,826 |

Both declared exploratory screens pass, with 197/193 observed event groups.
Unknown sides remain 147/322 respectively; 26 known rejected sides are included
in each observed score. Closed-only GRU MAE is 1.5462/2.1439 cents. Across seeds
7, 42 and 101, GRU observed MAE ranges from 1.5696–1.6354 cents at 30 seconds and
2.1395–2.2118 cents at 60 seconds; seed 42 remains primary.

This is an accuracy result, not a profitable strategy: the primary GRU generated
zero simulated attempts at 30 seconds and one at 60 seconds, whose quoted
scenario return was -5 cents after the fixed cost buffer. No real orders occurred.

The completed-final artifact audit passed at 18:13:53 UTC. It reconciled 90
configurations, 270 fold artifacts, 94 final prediction outputs and all 1,413
final source histories; checked 36 fitted scalers; and replayed all 26 primary
seed calibration corrections. Saved-model prediction replay uses sampled rows;
robustness-seed corrections are checked against frozen calibration files/counts.
The script's bootstrap replay uses the frozen unit-tested implementation. A
separate reviewer subsequently reconstructed scenario outcomes and all 94
prediction MAEs, independently implemented event-group aggregation and 1,000
bootstrap resamples, replayed every primary-GRU final prediction, and verified
the zero/one-attempt economics directly. That independent final review passed.
See `audit.json` for the script's full limits; its bootstrap limitation describes
the script alone, not this additional review.

## Interpretation

Observed-side MAE covers known quote scenarios, including separately counted
rejected entries. Unknown returns remain intervals and retain population and
execution diagnostics. Interval-Huber is not a statistical correction for
informative missingness or generally an expected-return estimator. Improved
return error can reflect spread/cost mechanics rather than directional alpha.

The declared exploratory screen requires at least 5% relative observed-MAE gain,
a positive paired 95% event-proxy bootstrap interval and at least 20 observed
event groups, separately by horizon. It is not a power calculation, a global
significance claim, or proof of executable profits. Quote touches and simulated
scenarios are not confirmed fills. No automatic live promotion is authorized
by passing this screen.

## Artifacts and next steps

Run directory: `artifacts/accuracy_experiment_20261109`. `status.json` now says
`complete`. Its generated `REPORT.md` contains all primary-seed and frozen-old
reference results; `seed_spread.json`, `subgroup_diagnostics.csv` and `audit.json`
retain robustness, subgroup and acceptance details. `src.accuracy_reporting`
requires a passed final-scope audit. See README and the predeclared protocol for
exact commands and boundaries.

No collector deployment is requested. A completed final period is consumed
evidence; later studies must reserve a newer final segment. New feature schemas
remain offline pending live inference and explanation-parity integration review.
