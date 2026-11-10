# Prospective 60-second hybrid: initial frozen setup

This report preserves the initial setup and waiting snapshot. The once-only evaluation has since completed; see the [completed results](prospective_hybrid_results_20261109.md). The primary accuracy screen failed.

At this initial milestone, the prospective test was prepared and frozen with status **`waiting_for_fresh_data`**, zero eligible observations and zero eligible event proxies in the input-only readiness check. Evaluation outcomes were unopened and performance results were unavailable at that time.

The prospective input boundary is **2026-11-10 00:11:39.787550 UTC**, equivalent to **November 9 at 19:11:39.787550 EST**. Every evaluated 180-observation input history must begin at or after that boundary. It is the later of protocol creation and every previously consumed declared final end plus 70 seconds. Existing final periods therefore cannot be reused. Newly consumed periods are checked again before readiness and evaluation; stale pending cohorts are rejected.

## Frozen hypothesis

Use the existing residual study's `gru_60_seed42` forecast when current spread is **at least 4.0 cents**, and the constant-book forecast otherwise. The threshold is the 75th percentile of canonical spread, rounded to nine decimal cents, over exactly the prior **4,629 refit-training observations**. Calibration and validation observations do not determine the threshold.

The GRU checkpoint, scaler, specification and saved two-side calibration correction were copied into the versioned run and hashed. The previous study's choices, protocol, segment declarations, completion/consumption markers, refit manifest and protected-period metadata were checked through their existing hash bindings. Those anchors and the inference/evaluation code are frozen in `protocol.json`. Raw GRU outputs are clipped to [-104, 96] cents, the saved correction is added, and the result is clipped again. The constant-book forecast is `-100 * (ask - bid) - 4` cents on both sides.

There is no retraining, threshold search, seed selection or live promotion. The cohort keeps the prior strict 180-observation verified-timing histories, same-session gap checks and 120-second spacing. The forecast uses a one-second entry delay, 60-second horizon, minimum one-contract quote depth and four-cent cost buffer. Future quote availability does not decide input eligibility.

The primary comparison is hybrid versus constant-book. A positive accuracy screen requires at least 5% observed-side MAE improvement, a positive lower endpoint of the paired 95% event-proxy bootstrap interval from 1,000 resamples, and at least 20 observed-outcome event groups. That group count is a heuristic rather than a power guarantee; event proxies can remain dependent. The frozen GRU is a secondary comparison. Known rejected entries and unknown outcome intervals remain separately counted. This conditional accuracy screen does not establish profitability.

## Readiness and once-only evaluation

Readiness snapshots stable, completed feature CSVs into a new versioned source directory. The newest file, sync-conflict files and files modified within 30 minutes are omitted. Timestamp/protected-period checks precede copying price data; copies are checked against original SHA-256 and size/modification metadata. Remote Syncthing completion is not independently confirmed.

Readiness examines past feature histories only. At least 20 actually eligible event proxies are required before any future quote alignment. Once `final_started.json` exists, this run refuses another evaluation or a changed cohort, including after interruption. The consumed-final registry uses `final_labels_opened.json`: once that marker exists, the declared period stays consumed for subsequent studies even if scoring fails. A failure before labels are opened does not enter that registry, but this run still refuses automatic re-entry after its start marker.

Initial readiness completed at 2026-11-10 00:12:26.721229 UTC: zero candidates and zero event proxies, with `outcomes_opened=false`. The new prospective boundary and 30-minute file-stability requirement mean waiting is expected. There is no unattended polling, service or scheduled job. The collector and Syncthing settings were not changed.

Historical manual commands from the research workspace are shown below. The status command remains read-only; the completed run now refuses readiness/evaluation re-entry:

```powershell
# Inspect the current state without reading outcomes:
.\.venv\Scripts\python.exe -m src.prospective_hybrid status --output artifacts/prospective_hybrid_20261109

# Later, take another input-only source snapshot and check eligibility:
.\.venv\Scripts\python.exe -m src.prospective_hybrid readiness --output artifacts/prospective_hybrid_20261109

# Only after readiness reports ready_for_once_only_evaluation:
.\.venv\Scripts\python.exe -m src.prospective_hybrid evaluate --output artifacts/prospective_hybrid_20261109
```

The freeze command has already run; repeating it verifies the existing protocol rather than changing the boundary. Preserve the frozen code and artifacts for reproducibility.

## Verification

The full offline suite passed **333 tests** in 31.25 seconds; one live integration test was explicitly excluded. Sixteen focused tests cover threshold scope, equal-price-tick routing, protocol/consumed-period boundaries, hashes, zero/19-event waiting without outcome alignment, future-price invariance, 20-event quote alignment, once-only rejection, newly consumed periods and prior-control/protection tampering. A synthetic end-to-end evaluation mocks only GRU inference; it exercises actual quote outcomes, routing, metrics and second-look refusal. Tests submit no orders and make no account requests.

Actual copied-model parity was also checked on three frozen training histories with spreads **1, 4 and 5 cents**. Original and copied GRU predictions matched exactly after the saved correction; routing was constant-book, GRU and GRU respectively. The fit-only threshold independently reproduced 4.0 cents. This check used historical training inputs, not evaluation outcomes.

An initial freeze attempt rejected Windows path separators in the prior manifest before writing any protocol or copying the model. Portable path normalization and a duplicate-path regression test fixed that issue before the actual freeze.

Artifacts are under `artifacts/prospective_hybrid_20261109/`: `protocol.json`, `protocol.sha256`, copied `model/`, `protected_periods.json`, `model_parity.json`, immutable source/readiness attempts, `latest_readiness.json` and `status.json`. No `final_started.json`, `final_labels_opened.json` or final outcome folder exists at this waiting milestone.
