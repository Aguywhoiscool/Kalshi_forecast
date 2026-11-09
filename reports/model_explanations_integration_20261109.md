# Trained models and explanation dashboard integration

Frozen source: 2,696 files, 8,488,495 rows; 2,695 files eligible after the session
embargo. Reserved v2/v3 sessions and periods remain excluded. Primary run:
`artifacts/model_lab_traced_20261109`, with 63,798 reconstructed histories and
40,888 fit / 10,198 calibration / 12,712 validation examples. Source/execution
and actual training-trace audits passed. No model/dataset files are committed.

The prospective paired seven/eight-feature comparison uses identical identities:
4,600 fit / 1,148 calibration / 1,574 validation. All 7,322 histories have timing
known throughout their input sequences. Independent audit checked 32 saved
model/variant/horizon prediction artifacts. Adding time did not improve GRU MAE
in this comparison: 15.9382 to 16.1576 cents at 30s, and 18.6652 to 18.8495 cents
at 60s. Neither GRU selected trades under the primary policy. No robust profit
was demonstrated. A supplemental exclude-near-close-fit comparison is separate.

The supplement completed and passed its source/prediction audit: 3,343 fitting
examples after removing 1,257 examples strictly above 10 and at most 60 minutes
until close, with the same 1,148 calibration and 1,574 validation identities.
Calibration still includes near-close observations; this is fit-only exclusion.
The supplemental GRU selected three closed 30-second trades totaling -18 cents
and two closed 60-second trades totaling -9 cents under the assumed cost policy.
Its MAE was 15.7389 cents at 30s and 18.7487 cents at 60s. Fewer fitting examples
are a confound; these exploratory results do not prove a feature/regime effect.

The local dashboard was deliberately restarted with the audited seven-input
model run and connected directly to Kalshi market data. Verified page HTTP 200,
live state with three markets, no reported errors, and an actual live Monte Carlo
explanation HTTP 200. The browser was opened at http://127.0.0.1:8765. No collector
restart, trading, account changes or Syncthing changes occurred.

Model rows open frozen-forecast explanations and actual training replay plus a
separate educational gradient demo. Monte Carlo downloads expose all 2,048 actual
paths/draws. Forecast IDs, content-hashed artifacts, explicit expiry and bounded
cache prevent explanations silently switching to another forecast or model run.
The live dashboard rejects eight-input artifacts; timing comparisons are offline.

Root reran 244 local Python tests successfully, with one integration test skipped,
and five Node interaction tests. That local count includes previously installed
exchange tests, which remain separate uncommitted bootstrap work. Independent
review also replayed old/new model explanations, 120 additional linear cases,
varied Monte Carlo paths/payoffs and the full paired source audit. Owners fixed
reviewer-found gradient/objective, float32 tolerance, stale UI state and paired
audit coverage issues before these checks passed.

No browser executable or automation package was available for automated visual
layout QA. Opening the user's browser and DOM tests do not substitute for that
verification. The user can inspect the responsive layout and interactive plots;
do not claim zero possible errors or actual executable fills from these checks.
