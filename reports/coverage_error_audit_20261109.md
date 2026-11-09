# Development coverage and prediction-error audit

The audit exactly reproduced all **6,054 development identities**, including every metadata column, before opening development outcome arrays. They represent 488 sessions and 854 event proxies. Error analysis uses **3,820 distinct out-of-fold observations**, each counted once per horizon/model, rather than treating all 6,054 training/development observations as out-of-sample predictions.

Only the 2,695 authorized development feature CSVs were read: 8,485,705 recorded rows. The source snapshot's 111 other CSVs were inventoried through manifest/session metadata without reading their prices. Reserved and final outcome arrays were not opened. No data, model, collector or live setting was changed, and no training or trades occurred.

## Why eligible counts are smaller than recorded rows

| Past-input requirement | Eligible ends before spacing | Spaced observations | Sessions | Event proxies |
|---|---:|---:|---:|---:|
| 60 observations, valid quotes | 6,523,003 | 63,821 | 2,689 | 3,770 |
| 60 observations, valid quotes and verified timing | 1,031,584 | 11,077 | 516 | 895 |
| 180 observations, valid quotes | 3,447,446 | 40,855 | 2,414 | 3,524 |
| 180 observations, valid quotes and verified timing | 456,978 | 6,054 | 488 | 854 |

Every comparison keeps session boundaries, planned target boundaries, protected periods and 120-second spacing. Counts depend only on past inputs and clocks; future quote availability never determines eligibility. These populations differ, so the table establishes coverage, not model improvement.

For the reproduced 180-observation verified-timing cohort, the sequential window funnel removes 4,567,587 ends with insufficient same-session history, 3,450,660 with missing timing in the remaining history, 9,870 for gaps/elapsed history, 610 for the development boundary and 450,924 for spacing, leaving 6,054. These exclusions sum exactly to 8,485,705 potential input ends. Separate overlapping-flag outputs count all applicable problems; those totals deliberately overlap.

The row-level validator found zero invalid timestamps, duplicate timestamps, invalid quotes or inconsistent populated timing values in this authorized scope. Missing timing affected 7,147,903 rows; 1,337,802 rows had complete timing. This validates the recorded feature values under the existing checks, not raw WebSocket reconstruction or confirmed executions. Quote/timing-invalid rows are not removed before forming windows; their positions remain available to gap/history checks. Invalid timestamps and duplicate observations follow the original preprocessing rules; neither occurred here.

Reducing the verified history requirement from 180 to 60 observations adds 5,023 spaced observations, or 83.0%. Diversity increases more modestly: **28 sessions (+5.7%) and 41 event proxies (+4.8%)**. Neither rows nor overlapping windows establish an equivalent number of independent samples.

## Matched prediction errors

All entries below use the same unique OOF observations and observed YES/NO sides. Values are cents. A positive paired difference means the learned model is worse than the constant-book forecast, `-spread_in_cents - 4`.

| Horizon | Model | Observed MAE | Median absolute error | P90 absolute error | Paired MAE minus constant-book |
|---|---|---:|---:|---:|---:|
| 30s | Constant-book | 1.565231 | 0.000000 | 4.000000 | 0.000000 |
| 30s | Development-selected MLP | 1.679855 | 0.214640 | 4.429350 | +0.114624 |
| 30s | Three-architecture ensemble | 1.861087 | 0.723913 | 4.460049 | +0.295856 |
| 60s | Constant-book | 2.156559 | 0.000000 | 6.000000 | 0.000000 |
| 60s | Development-selected GRU | 2.302599 | 0.729914 | 6.178451 | +0.146040 |
| 60s | Three-architecture ensemble | 2.382349 | 0.859528 | 6.051789 | +0.225790 |

Thirty-second evaluation includes 7,212 quoted closes, 49 known rejected-entry sides and 379 unknown sides; 60-second evaluation includes 6,827 closes, 49 rejections and 764 unknowns. Observed-outcome event proxies number 507 and 499 respectively. Unknown intervals stay separate from observed MAE. Group outputs flag fewer than 20 observed event proxies as low support.

Stable outcomes explain much of the learned models' disadvantage. When entry-to-exit midpoint was unchanged, the selected model added 0.16337 cents of MAE at 30 seconds and 0.36275 cents at 60 seconds relative to constant-book. For midpoint movements of at least one cent, the 60-second GRU reduced MAE by only 0.06552 cents. These movement groups are retrospective diagnostics and are never inputs or eligibility filters. Development model/subgroup selection makes these discovery statistics optimistic.

On quoted closes, the constant-book error decomposes exactly into midpoint and spread terms. Mean absolute midpoint/spread components were 1.27681/0.80527 cents at 30 seconds and 1.87586/0.91036 cents at 60 seconds. Components can cancel, so they are not additive causal shares. Correlations and per-observation algebra are saved as descriptive associations.

## One recommended next experiment

Run a **prospective paper comparison of a 60-second constant-book/GRU gate**, using the frozen GRU only when current spread is at or above the training-fit 75th percentile, and constant-book otherwise. Freeze that rule and model before collecting its evaluation segment. Use canonical spread arithmetic, the same execution assumptions and unknown-outcome accounting, and a genuinely later unused evaluation period. This is a proposed experiment; it was not run by this audit.

This is the strongest supported input-defined discovery: the canonical upper-spread group contains 891 OOF observations and 252 observed event proxies, with GRU paired MAE difference **-0.07309 cents**. The fit thresholds were approximately four cents. The effect reversed in one fold: differences were -0.12552, +0.14875 and -0.19684 cents. The aggregate posthoc gate would reduce overall OOF MAE by only 0.01716 cents, about 0.8%, so the evidence does not justify expecting a 5% improvement or profitability. Its value is a cheap, falsifiable test of keeping the strong baseline on stable cases while checking the small supported exception. No further architecture search is recommended before that test.

Canonical spread bins round arithmetic to nine decimal cents before fitting quantiles and assigning groups. This avoids assigning economically identical four-cent spreads to different bins because of floating subtraction. Movement thresholds receive the same arithmetic normalization. Quotes are not filled or altered. Original first-pass outputs are preserved; interpretation uses `canonical_fold_error_groups.csv`, `pooled_error_groups.csv` and `canonical_fold*_schema.json`.

## Verification and artifacts

The final offline suite passed **317 tests**, with one live integration test explicitly deselected; 13 focused coverage/summary tests passed. Tests exercise exact reconstruction, sequential/overlapping reconciliation, gaps and protected sessions, all four history/timing variants' invariance to future prices or recording-end availability, OOF identity rejection, independent error arithmetic and equal-price-tick bin invariance.

Versioned artifacts: `artifacts/coverage_error_audit_20261109/`. The identity and audit JSONs record the 6,054/3,820 denominators. `provenance.json` hashes source/protection metadata, development arrays, OOF payloads, fold identities and audit implementation. CSVs separate row/window funnels, overlapping flags, group coverage, signed bias/median/P90 errors and decomposition. The side-level table contains 45,840 records: three models × two horizons × 3,820 observations × two sides. An independent reviewer checked exact identities, reconciliation, pooled metrics, canonical-bin arithmetic and quote-return algebra. Findings concern development data and are not a new holdout performance claim.

Reproduce into a new output directory with these two offline stages; the second publishes canonical pooled diagnostics and provenance:

```powershell
.\.venv\Scripts\python.exe -m src.coverage_error_audit --source artifacts/accuracy_source_20261109 --assignment artifacts/accuracy_experiment_20261109 --study artifacts/residual_accuracy_20261109 --output artifacts/coverage_error_audit_NEW
.\.venv\Scripts\python.exe -m src.coverage_error_summary --source artifacts/accuracy_source_20261109 --assignment artifacts/accuracy_experiment_20261109 --study artifacts/residual_accuracy_20261109 --output artifacts/coverage_error_audit_NEW
```
