# Model windows and relative leaderboard

Dashboard presentation work requested by the user: each model gets its own animated
"window" that shows how it reaches its forecast, a leaderboard of relative
profitability normalized to the cross-model mean, and a less green palette.
Read-only local inference only; no trading, collector, Syncthing or reserved-outcome changes.

## Changes

- **`src/model_scoreboard.py`** (new): read-only leaderboard from the live model run's
  saved validation predictions and outcome bounds.
  - **Policy:** always take the side with the higher forecast.
  - **Closed mean:** mean return over the outcomes that are known.
  - **Lower/upper means:** bound every row, unknown outcomes included.
  - **Relative score:** closed mean minus the eight-model mean.
  - **Intervals:** 1,000 session-cluster bootstrap resamples, seed 20261109.
- **`src/live_dashboard.py`:**
  - `GET /api/scoreboard`: cached and tied to the run fingerprint; returns 503 on failure.
  - `GET /api/windows/{forecast_id}`: every model's actual `explanation()` for one
    forecast. The parity assertions are kept, per-model errors are isolated, and an LRU
    cache sits behind the expiry and artifact checks.
- **Bug fix in `Dashboard.update()`:** live histories arrived column-ordered while the
  stored copy was row-ordered. `summarize()` differs by about 1e-16 between the two
  layouts, which flipped gradient-boosting splits by up to 0.6¢ between a forecast and
  its later explanation. Histories are now C-ordered before prediction, matching the
  training `history.npy` exactly.
- **Frontend:**
  - New `frontend/model_windows.js`: eight data-driven windows, each with a
    raw → calibration → final strip. Schematic geometry is labelled.
    - mean: training-mean line
    - ridge: contribution waterfall
    - SGD: tug of war
    - decision tree: visited path
    - random forest: tree-output dots
    - gradient boosting: stage stairs
    - GRU: input-sensitivity strip
    - Monte Carlo: actual path fan, with no jitter
  - Leaderboard with intervals and "within noise" tags.
  - Slate palette with colour-vision-validated accents; no green remains.
  - Copy reframed away from trading signals.

## Results shown (validation split, closed-outcome means)

All models average about −7.2 to −7.5¢ per hypothetical trade after the 4¢ buffer.

| Horizon | Above average, outside noise | Below average, outside noise |
|---|---|---|
| 30s | ridge +0.068, random forest +0.051 | mean −0.082, Monte Carlo −0.160 |
| 60s | gradient boosting +0.086, random forest +0.057, ridge +0.050 | mean −0.119, Monte Carlo −0.178 |

Everything else is within noise. The upper bounds that count unknown outcomes are
positive (+0.06 to +3.46¢), so the page says "no profit on known outcomes" rather than
claiming unprofitability for the complete population.

## Checks

- **Offline pytest:** 346 passed, 1 integration test deselected. The baseline was
  333; the change adds 13 tests.
  - Must run with `--basetemp` inside a writable directory here, because the default
    pytest temp directory is sandbox-blocked.
- **Node:** `frontend_explanations` + `frontend_windows`: 41 tests, 40 pass, 1 optional
  fixture test skipped. With the local real-data fixtures it passes too.
- **Independent data audit**, which reimplemented the scoreboard separately: point
  estimates are identical and the interval endpoints differ by at most 1.8e-15. 78 of 78
  checks on real-forecast window payloads passed: parity, tree paths, Monte Carlo replay,
  GRU inputs.
- **Two independent review rounds** each of the backend and the frontend. All findings
  were fixed and re-verified.

## Limitations

- **Population:** the leaderboard uses the exploratory validation split that was
  repeatedly used during development. It is not fresh evidence. Differences between
  models are at most ±0.2¢, and there is no multiple-comparison correction.
- **Bootstrap unit:** "within noise" is sensitive to it for borderline models
  (30s SGD, 60s ridge and SGD). The extremes hold under session, ticker, event and row
  clustering.
- **Per-model populations:** closed means cover slightly different rows per model
  (known fraction 0.904–0.907). Restricting to rows closed for every model changes the
  relative values by at most 0.009¢.
- **Not seen in a browser:** none was available offline. Layout was checked by
  computed geometry and DOM-harness tests only. Effective SVG text is 9.1–12px from
  360px up; 8.1px at 320px.
- **No live end-to-end run:** the Syncthing file feed arrives about 140s late, beyond
  the 30s freshness rule, and the direct Kalshi feed was not started. Windows were
  exercised with real recorded histories through the real models.

## Next steps

- Open the dashboard in a browser on the live feed and review the animation and layout.
- Optional: a recorded/offline demo mode, clearly badged, so the windows can be shown
  without a live feed.
