# Return-size accuracy experiment

Implementation protocol, declared before new final-evaluation outcomes are
scored. Existing audited models and reserved v2/v3 data remain untouched.
No collector change, real trading or automatic dashboard promotion is authorized
by this experiment's scripts.

## Data and evaluation boundary

Freeze stable completed recordings with immutable hashes and exclusion reasons.
Use timestamp-only discovery to reserve the latest chronological segment,
strictly later than the prior source development maximum plus a 70-second gap.
The prior source boundary reported by the research implementation is
2026-11-09T02:11:39.820570Z; verify that value from source timestamps.
If the reserved segment has fewer than 20 ticker-derived event proxies, publish
development-only results and keep final outcomes unscored.

Use three chronological walk-forward development folds, session separation and
70-second gaps after planned target endpoints. Longer context is 180 observations,
not a 180-second prediction horizon. Keep 30/60-second forecasts and 120-second
signal spacing. Overlapping context and related markets remain correlated.

## Staged selection, before final scoring

Compare fixed defaults first, on a common valid 180-observation cohort:
- Seven inputs from its last 60 observations.
- Verified timing and engineered features from its last 60 observations.
- Engineered features using the full 180-observation context.

Keep the existing full-history seven-input model as a secondary frozen baseline;
also refit the default seven-input model on matched identities. Older histories
without valid timing are not assigned retrospective/imputed schedules. Timing
must be known at every history row and countdowns must reconcile within .001s.

For SGD and GRU, compare conservative-target defaults with a versioned
interval-Huber objective (5-cent transition). Known labels retain exact scenario
returns, unknown outcomes retain their feasible intervals. Predictions inside an
unknown interval produce no data-loss gradient. This does not identify expected
returns under informative missingness or correct that missingness statistically.

Conservative calibration uses the fixed mean lower-target residual. Interval
calibration uses one fixed mean projection residual: mean(clip(raw, lower, upper)
minus raw), zero for predictions already inside an unknown interval. These are
declared pipeline variants, not a pure loss-only ablation.

Select feature family/objective by development-fold observed MAE, then tune only
that chosen family/objective, avoiding a Cartesian search over every ablation:
- Ridge: alpha 0.1, 1, 10, 100.
- SGD: learning rate 0.0001/0.001 crossed with alpha 0.0001/0.01.
- Boosting: learning rate 0.03/0.1, leaves 7/15, L2 1/10; at most 300 iterations.
- GRU: layers 32/16, 64/32 or 96/48 crossed with dropout 0.1/0.3;
  learning rate .001, at most 30 epochs, calibration patience five.

Keep mean, tree, forest and Monte Carlo comparisons. Monte Carlo's separate
60/180-observation variants retain genuine bootstrap draws and flat-history
behavior; no fabricated volatility is added.

## Gates and reporting

Seed 42 is primary. Seeds 7 and 101 are robustness refits, never selected using
final scores. Freeze choices/checksums before scoring the final segment once.
Freeze the strongest matched-cohort default baseline per horizon using only
development-fold observed MAE. Compare the development-selected winner against
that baseline, not a conveniently weak final-scoring comparator. Also report
family-default and frozen-old-model comparisons as secondary diagnostics.
Primary observed scenario-return MAE includes known closed/rejected outcomes,
with rejected-zero returns reported separately; closed-only MAE is secondary.
Unknown candidates remain in population counts, interval diagnostics and global
execution-bound evaluation. Their error is not silently treated as observed.

Report each horizon separately, plus timing/series groups, coverage, missingness,
seed variation and 1,000 event-proxy bootstrap resamples of paired primary-seed
loss differences. A release candidate needs at least 5% relative observed-MAE
improvement and a positive paired 95% improvement interval. These are review
gates, not proof of profitability, independent events or statistical power.

Retain one-second entry, 1-cent margin, 4-cent cost scenario, quote/depth checks
and uncertainty blocking. No target/threshold is chosen for best final trading
profit. New schemas are experimental offline artifacts; live promotion requires
separate source, explanation-parity and final integration review. Existing live
model pointers must not be overwritten during the search.
