# Broader neural accuracy study: completed

The broader study completed 108 development-fold fits, 18 refits and one fresh final evaluation. Neither preselected winner passed the release screen. The constant-book forecast remained more accurate than either primary neural model on observed outcomes. No model was promoted; trading and the collector were unchanged.

## Primary final results

All MAEs below are cents per observed YES/NO side. Winners and comparisons were frozen using development data before fresh-final outcomes were opened. Seed 42 is primary; other seeds are robustness checks.

| Horizon | Preselected neural winner | Winner MAE | Previous GRU MAE | Constant-book MAE | Improvement over previous GRU |
|---|---|---:|---:|---:|---:|
| 30 seconds | Residual MLP, 1-cent interval Huber | 2.11456 | 2.13896 | 2.01304 | 1.14% |
| 60 seconds | Direct GRU, 1-cent interval Huber | 2.69844 | 2.80732 | 2.67543 | 3.88% |

The constant-book forecast is `-spread_in_cents - 4` for each side. It was selected as the strongest default on the exact development folds; its final performance did not determine that choice. The three-architecture equal-weight ensemble was evaluated on aligned out-of-fold observations and retained as a final comparison, but development did not select it as primary.

Paired, 1,000-resample event-proxy bootstrap intervals for absolute MAE improvement were:

| Horizon | Versus previous GRU | Versus constant-book |
|---|---:|---:|
| 30 seconds | [-0.09307, 0.11697] cents | [-0.13589, -0.06806] cents |
| 60 seconds | [0.05502, 0.16850] cents | [-0.11259, 0.10714] cents |

Positive values favor the neural winner. The 60-second improvement over the previous GRU had a positive interval but fell below the predeclared 5% improvement threshold. Both horizons also failed the comparison against the strongest default. These are unadjusted exploratory intervals using event proxies, not a guarantee of statistical power or independence.

## Data and uncertainty

Development reused the previous frozen 6,054-observation cohort and identical three chronological folds, with session separation and 70-second embargoes. Its preceding final outcomes were excluded from this round. A new, stable source snapshot contained 2,866 files and 9,097,402 rows. The new final boundary was declared at 2026-11-09 13:28:12.778966 UTC, 70 seconds after the previously consumed declared final end.

The actual fresh final cohort contained 966 observations across 114 event proxies. Input histories began at 13:37:23.879050 UTC; the latest planned target end was 20:10:34.325356 UTC on November 9. At 30 seconds, 1,841 of 1,932 sides were observed and 91 remained unknown; at 60 seconds, 1,750 were observed and 182 remained unknown. Observed-outcome bootstrap groups numbered 113 and 107 respectively.

Observed-side MAE includes quoted closes and known rejected entries. Unknown outcomes stayed as intervals in training and population diagnostics. Thus observed-side improvements do not establish accuracy on missing outcomes or profitable trading. Training-defined spread, volatility and time-to-close quartiles, training-series categories, availability diagnostics, signed bias, median error and 90th-percentile error are saved in the run artifacts.

The fixed hypothetical taker policy generated zero signals for either primary model. The secondary TCN produced losing quoted closes: 34 closed at 30 seconds totaled -192 cents, with total outcome bounds [-266, -66] cents after two unresolved exits; 19 closed at 60 seconds totaled -110 cents, with total bounds [-212, 88] cents after three unresolved exits. The 30-second ensemble had one quoted close at -5 cents. These are offline quote-based simulations with a fixed 4-cent cost buffer, not orders, confirmed fills or demonstrated profitability.

## Verification and artifacts

Implementation tests included real tiny GRU/TCN/MLP fitting and reload, TensorFlow Huber/L1 gradients, TCN prefix causality, fit-only scaling, residual reconstruction, ensemble row-identity rejection and source/comparator tamper rejection. The final full offline suite passed 304 tests in 29.38 seconds; one live integration test was explicitly deselected. The unchanged frontend's 21 Node tests also passed. Five auditor tests included saved-prediction tampering and missing zero-trade execution-record rejection.

The completed final auditor exited successfully and wrote `artifacts/residual_accuracy_20261109/audit.json` with `status=passed` and `scope=completed_final`. It checked all 36 configuration summaries and 108 fold payloads, actual saved-model replay for 36 first folds plus six primary-seed refits, all 30 final prediction/metric records, exact refit populations, frozen choices and weights, diagnostic fit boundaries, 14 execution records, source/final boundaries and both bootstrap gates. Twelve frozen feature-CSV histories were independently reconstructed (six development, six final); this is sampled feature-source reconstruction, not an exhaustive replay of every recorded row or raw WebSocket messages. A separate reviewer verified selections before final opening and replayed actual checkpoint predictions, all primary final predictions and event-proxy confidence intervals.

Run directory: `artifacts/residual_accuracy_20261109/`. Important files include `protocol.json`, `choices.json`, `refit_manifest.json`, `final_metrics.json`, `primary_gates.json`, `seed_spread.json`, `execution.json`, `subgroup_diagnostics.csv` and `audit.json`. Training started at 20:56:07 UTC and completed at 22:37:32 UTC. Standard logs are alongside the run folder; local milestone history is in `artifacts/agent_status/research.log`.

No automatic live promotion is justified. Any further round must preserve this consumed final period and declare another genuinely later evaluation segment before model selection.
