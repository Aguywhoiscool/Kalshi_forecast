# Near-close coverage findings — November 6, 2026

The frozen development snapshot already contains substantial observations near
the retrospectively reported market close. Recollecting the entire price history
is unnecessary. Historical schedules are not verified as known at observation
time, however, so this audit does not approve seconds-to-close as a causal model
input or prove a near-close trading edge.

## Audited scope

All 3,478 eligible development markets in 2,078 frozen feature files were checked.
All 3,478 schedules were recovered, with zero failures, covering 6,570,008 rows.
Reserved v2/v3 sessions and periods remained excluded. Source hashes were checked.
Forty previously fetched schedules were reused; the remaining public metadata
requests were limited to six per second. An offline replay reproduced the audit
from its complete cache without additional API requests.

| Retrospective time remaining | Recorded rows | Training windows | Validation windows |
|---|---:|---:|---:|
| Over 60 minutes | 5,237,877 | 30,818 | 8,377 |
| 30–60 minutes | 643,559 | 4,195 | 675 |
| 10–30 minutes | 505,882 | 3,279 | 573 |
| 0–10 minutes | 182,384 | 1,119 | 230 |
| After reported close | 306 | 1 | 0 |

The within-hour total is 1,331,825 rows, 8,593 training windows and 1,478
validation windows. Windows are spaced by 120 seconds per market/session;
related markets/events can still be correlated. These are descriptive groups
inside the existing audited dataset, not newly verified timing features.

## What the existing forecasts show

No models were retrained. The original global position blocking, fixed 1-cent
entry margin and 4-cent cost scenario were preserved when grouping saved
validation trades. The GRU generated no trades in any within-hour group at
either horizon. Monte Carlo's within-hour 30-second trades had a combined
simulated loss of 78 cents; its 60-second results included unresolved positions.
These post-hoc subgroup results do not demonstrate a profitable strategy.

## Why schedule verification matters

3,475 of 3,478 markets permit early closing. The market's `updated_time` is later
than its last recorded observation for 3,213 markets. An update does not prove
the close schedule changed, but neither field proves what schedule was known
at the time. The 306 observations after the reported close are retained as a
diagnostic anomaly, not automatically declared invalid or rewritten. All fetched
timing records remain `asof_verified=false`.

## Next execution step

Apply the prepared collector patch on the collection computer, preserving its
local flags and observer changes. Record close schedules together with their
observation times in new sessions. Set `COLLECTOR_MIN_CLOSE_MINUTES=10` for a
deliberate restart and verify ten minutes of recording. No remote deployment or
restart has been performed from this research computer.

Use `reports/collection_computer_near_close_prompt.md` for that handoff and
`reports/near_close_experiment_protocol.md` for the controlled future comparison.
The existing seven-feature price histories remain usable; the prospective data
is needed to test adding trustworthy timing information, not to replace all
historical books. Do not promote a new time-aware model to the dashboard until
that comparison has been evaluated.

Evidence remains local under `artifacts/near_close_full_20261106/`: `summary.json`,
`REPORT.md`, `row_coverage.csv`, `window_coverage.csv`,
`model_subgroup_diagnostics.csv`, `window_schedule_diagnostics.parquet`, and the
resumable `schedules.jsonl` cache. No trades, collector changes, reserved-outcome
evaluation or model retraining were performed by this audit.
