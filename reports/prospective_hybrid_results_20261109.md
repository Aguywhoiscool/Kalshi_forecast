# Prospective hybrid: completed, primary screen failed

The once-only prospective evaluation completed at **2026-11-10 02:15:53 UTC** (November 9, 21:15:53 EST). The hybrid did not beat its primary constant-book comparator. **No live promotion is justified by this test.**

| Forecast | Observed MAE, cents |
|---|---:|
| Constant-book | **1.442586** |
| Spread-gated hybrid | 1.516377 |
| Frozen GRU | 1.605128 |

The frozen rule used the GRU when current spread, rounded to nine decimal cents, was at least **4 cents**; otherwise it used `-spread_in_cents - 4` on both sides. The threshold came only from the prior 4,629 refit-training observations. The checkpoint, scaler, saved correction, code and rule were fixed before the evaluation period. There was no retraining or threshold search.

Against constant-book, the hybrid's observed MAE was **5.12% worse**, a difference of +0.073791 cents. The paired 1,000-resample event-proxy bootstrap interval for *improvement* was **[-0.169019, +0.021556] cents**, crossing zero. The primary requirement—at least 5% improvement with a positive lower confidence bound—failed.

Against the frozen GRU, the hybrid improved observed MAE by **5.53%**, with an improvement interval of **[+0.047588, +0.150751] cents**. That secondary comparison supports the fallback's advantage over always using this GRU, but it does not replace the failed primary comparison or establish superiority to constant-book.

## Cohort and uncertainty

The evaluation contained **116 observations across 12 sessions and 37 event proxies**. The GRU handled 53 observations; constant-book handled the other 63. Histories retained the strict 180-observation verified-timing requirement and 120-second spacing. Entry delay was one second, forecast horizon 60 seconds and the cost buffer four cents.

There were **174 observed sides out of 232 (75%)**, all quoted closes, with zero known rejected-entry sides. **58 sides (25%) remained unknown** and were retained as intervals rather than filled. Confidence intervals used **33 event proxies with observed outcomes**, exceeding the predeclared minimum of 20; that count is a heuristic, not a power guarantee or proof of independence.

Results describe the observed subset. Missing outcomes may be nonrandom, so these conditional errors and intervals do not identify performance on the complete population. Accuracy comparisons are not evidence of executable fills or trading profitability.

The prospective input boundary was **2026-11-10 00:11:39.787550 UTC**. Actual histories began no earlier than 00:12:37.397336 UTC; the latest planned target ended at 01:12:17.291838 UTC. The declared evaluation period ends at **01:17:30.338604 UTC** and is consumed, including its unknown outcomes. Another study must respect that declared end plus the established embargo and its own prospective freeze boundary.

## Verification and next action

Evaluation was invoked once. Frozen protocol SHA-256 is `dd8516ac4a83105649622f2dfaa05f8742509eba481cd63266078b82c7fdf6f4`. Completed results are in `artifacts/prospective_hybrid_20261109/metrics.json`, `gates.json`, `final_completed.json` and the saved predictions/routing mask under `final/`. No evaluation command was rerun for this report; verification replays saved artifacts without selecting another model or threshold.

Independent verification passed for all 116 frozen feature-CSV histories and their timing/future-quote alignment from 13 source files; the actual network, scaler, saved correction and 53 routing decisions; scalar outcome metrics; and independently implemented event-proxy bootstrap intervals. Hashes, prospective boundaries and once-only marker chronology also passed. The verification record is `reviewer_final_verification.json` in the run directory. This reconstructs recorded features and quotes, not raw WebSocket messages or confirmed executions.

Keep the result as a failed prospective primary hypothesis and retain the strong constant-book reference. Do not reuse this evaluation period for another release claim. No collector settings, live prediction pointer or trading behavior changed.
