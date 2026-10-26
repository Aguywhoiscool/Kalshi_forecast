# Kalshi Forecasting & Arbitrage Engine — Project Roadmap

**Project:** `kalshi-forecasting`  
**Status:** In development  
**Current milestone: Phase 2 — Kalshi REST client  
**Primary prediction target:** Kalshi YES contract price movement over the next **1, 5, and 10 seconds**  
**Primary stack:** Python 3.12, TensorFlow/Keras Sequential GRU, pandas, NumPy, scikit-learn, WebSockets, FastAPI, React  
**Operating mode:** Data collection, research, and paper trading first. **Live trading disabled by default.**


## Project goals and success criteria

1. Collect synchronized, timestamped, reproducible Kalshi market and order-book data.
2. Forecast 1-, 5-, and 10-second changes in a consistently defined YES quote or midpoint using only information available at prediction time.
3. Compare GRU forecasts against naive and statistical baselines on chronological held-out data.
4. Build a separate, fee-aware mispricing/arbitrage scanner using valid settlement relationships.
5. Simulate trades with spreads, depth, fees, latency, and partial fills; measure realistic out-of-sample results.
6. Present methods, limitations, and measured results in a clean GitHub repository and optional React dashboard.

## Current setup checkpoint


## Recommended repository structure

```text
kalshi-forecasting/
├── PROJECT_ROADMAP.md            # This file: update continuously
├── README.md
├── .gitignore
├── .env                        # Local secrets; never commit
├── .env.example                # Placeholder config; safe to commit
├── requirements.txt
├── .venv/                      # Local only
├── src/
│   ├── __init__.py
│   ├── kalshi/
│   │   ├── __init__.py
│   │   ├── rest_client.py
│   │   ├── websocket_client.py
│   │   └── authentication.py
│   ├── market_data/
│   │   ├── __init__.py
│   │   ├── market_discovery.py
│   │   ├── orderbook.py
│   │   ├── normalizer.py
│   │   └── data_collector.py
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── dataset_builder.py
│   │   ├── baseline_model.py
│   │   ├── gru_model.py
│   │   ├── train.py
│   │   ├── calibration.py
│   │   └── evaluate.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── complement_arbitrage.py
│   │   ├── multi_outcome_arbitrage.py
│   │   └── forecast_mispricing.py
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtester.py
│   │   ├── paper_trader.py
│   │   └── performance.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── fee_calculator.py
│   │   ├── position_limits.py
│   │   └── risk_manager.py
│   └── api/
│       ├── __init__.py
│       └── routes.py
├── scripts/
│   ├── collect_data.py
│   ├── build_dataset.py
│   ├── train_model.py
│   └── run_backtest.py
├── tests/
│   ├── test_rest_client.py
│   ├── test_orderbook.py
│   ├── test_features.py
│   ├── test_dataset_builder.py
│   ├── test_fee_calculator.py
│   └── test_backtest.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── models/.gitkeep
├── notebooks/
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── forecasting_methodology.md
│   ├── arbitrage_math.md
│   └── research_results.md
└── frontend/                     # Add when reaching dashboard phase
```

Create only the modules you need for the active milestone; the tree is the eventual target, not a requirement to create dozens of empty files immediately.

## Phase 1 — Environment and Git setup

- [x] Confirm 64-bit Python 3.12 and project `.venv` are selected in VS Code.
- [x] Install and import core dependencies: `requests`, `websockets`, `python-dotenv`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `tensorflow`, `matplotlib`, `pytest`, `fastapi`, `uvicorn`.
- [x] Keep `.env` local and `.env.example` free of credentials.
- [x] Ensure `.gitignore` excludes `.venv/`, `.env`, `*.pem`, `*.key`, large data, and model artifacts.
- [x] Create `README.md`, this roadmap, initial `src/`, and `tests/`.
- [x] Save the initial Git commit and push to your chosen remote.

**Done when:** `python --version`, imports, `pytest`, and `git status` run as expected; repository is available on GitHub without secrets.

## Phase 2 — Kalshi REST API and market discovery

- [ ] Read current **official** Kalshi developer docs: <https://docs.kalshi.com/>.
- [ ] Check the documented production and demo base URLs before hardcoding any environment setting.
- [ ] Implement `src/kalshi/rest_client.py` with timeouts, structured errors, and pagination.
- [ ] Add an unauthenticated request for open markets and print ticker, title, status, and current quote fields.
- [ ] Implement `market_discovery.py` to select 5–10 markets by recent activity, valid quotes, and liquidity.
- [ ] Keep selection criteria in configuration instead of hardcoding individual tickers.
- [ ] Add unit tests with recorded sample responses; don't require live network access to run tests.

**Done when:** A single command lists 5–10 appropriate markets reliably and unit tests pass.

**Suggested next Git commit:** `Add Kalshi REST market discovery client`

## Phase 3 — Streaming order books and durable collection

- [ ] Implement authenticated WebSocket connection using current Kalshi requirements and channels.
- [ ] Handle initial book snapshots and subsequent incremental updates.
- [ ] Reconstruct order books correctly, including YES/NO complement conventions and valid price increments.
- [ ] Check sequence continuity if provided; reconnect and resynchronize if data becomes inconsistent.
- [ ] Record the exact time your machine received each message, plus exchange timestamps if available.
- [ ] Save the unmodified incoming message or normalized event record **before** aggregation.
- [ ] Write data to append-only files (e.g., daily Parquet partitions) or a durable database.
- [ ] Derive 1-second snapshots from the event stream; do not discard raw updates.
- [ ] Detect stale books, crossed/impossible reconstructed quotes, duplicate messages, and gaps.
- [ ] Monitor disconnections, data-writing failures, and storage growth.
- [ ] Collect multiple days across several liquid markets without overwriting earlier data.

**Done when:** You can stop/restart the collector, reconstruct a valid book, and replay saved records reproducibly without losing earlier data.

**Suggested next Git commit:** `Implement timestamped Kalshi market data collector`

## Phase 4 — Market features

- [ ] Define a common YES-side representation with explicit executable bid, ask, midpoint, and depth.
- [ ] Compute spread and available depth at multiple price levels.
- [ ] Compute order-book imbalance `(bid_depth - ask_depth) / (bid_depth + ask_depth)` with zero-depth handling.
- [ ] Compute recent returns over 1, 5, 10, and 30 seconds.
- [ ] Compute short-term volatility and rolling statistics without peeking forward.
- [ ] Compute trade rate, volume, order additions/cancellations when data supports them.
- [ ] Include time since most recent genuine market update; mark stale/closed markets.
- [ ] Represent market-specific tick sizes explicitly.
- [ ] Handle irregular event time and repeated unchanged snapshots.
- [ ] Fit standardization **only on training data**, transform validation/test data with those fitted parameters.
- [ ] Preserve both original and transformed values for debugging.

**Done when:** Every feature has a documented formula, expected units, and tests for edge cases.

## Phase 5 — Forecast labels and dataset

- [ ] Pick a price definition **before** modeling: start with YES midpoint where both sides exist.
- [ ] Define target horizons of 1, 5, and 10 seconds.
- [ ] Label DOWN / FLAT / UP by change in valid ticks (or define continuous returns as a separate experiment).
- [ ] Use the book as of the precise future cutoff; never accidentally select later information.
- [ ] Construct rolling sequences, e.g., 60 timesteps × 12 features.
- [ ] Mark gaps and incomplete sequences rather than silently backfilling from the future.
- [ ] Split chronologically with an embargo around boundaries; group related contracts/events to prevent leakage.
- [ ] Preserve an untouched final test set and separately track class balance by market.
- [ ] Record metadata, dataset version, feature definition, and split timestamps.

**Done when:** A deterministic dataset-build script emits `(samples, timesteps, features)` plus labels and reproducible split manifests.

## Phase 6 — Baselines first

- [ ] Always-FLAT baseline.
- [ ] Recent price-momentum baseline.
- [ ] Order-book imbalance threshold baseline.
- [ ] Logistic regression using engineered summary features.
- [ ] Optional gradient-boosted trees.
- [ ] Evaluate each model on identical chronological holdouts.
- [ ] Report confusion matrix, balanced accuracy, class precision/recall, multiclass Brier score, log loss, and calibration.

**Done when:** Simple model benchmark results are saved and repeatable. Never judge success using raw accuracy alone if most observations are FLAT.

## Phase 7 — Sequential GRU model

- [ ] Implement a first Keras model with an input sequence, GRU(64), optional second GRU(32), Dense(16), and Dense(3, softmax).
- [ ] Compile with sparse categorical cross-entropy; use early stopping and a time-based validation set.
- [ ] Compare smaller models (e.g., GRU 16/32) and larger models (64/128); don't assume more neurons is better.
- [ ] Compare 15, 30, and 60-second lookback windows.
- [ ] Compare one vs. two GRU layers, dropout, input standardization, and normalization layer choices.
- [ ] Keep training-seed, hyperparameters, model version, scaler, and feature schema with each artifact.
- [ ] Measure inference latency on the machine that will run the model.
- [ ] Evaluate the selected architecture once on untouched test data.

**Done when:** The GRU has defensible out-of-sample results and is compared fairly with baselines.

## Phase 8 — Forecast quality and economic relevance

- [ ] Calculate accuracy, balanced accuracy, per-class precision, confusion matrix.
- [ ] Calculate multiclass Brier score, log loss, and reliability/calibration plots.
- [ ] Inspect performance by market, spread, liquidity, activity, and prediction horizon.
- [ ] Examine errors around news shocks, market closures, and stale quotes.
- [ ] Compare midpoint prediction to **future executable** bid/ask changes.
- [ ] Record signal frequency and how quickly apparent edges disappear.

**Done when:** Performance is described honestly, including where the model fails.

## Phase 9 — Separate arbitrage and mispricing engine

- [ ] Represent valid payout constraints for complementary YES/NO outcomes.
- [ ] Compare exhaustive mutually exclusive outcomes where contract rules truly support the relationship.
- [ ] Add logically nested threshold market checks only after verifying settlement definitions.
- [ ] Calculate executable cost from the full relevant order-book depth, not midpoint-only quotes.
- [ ] Include accurate fees, rounding, slippage, and incomplete-fill risk.
- [ ] Keep model-forecast mispricing **separate** from guaranteed-payoff arbitrage.
- [ ] Log all rejected signals and why they were rejected.

**Done when:** Every candidate opportunity is explainable with explicit payoff math, order-book depth, fees, and assumptions.

## Phase 10 — Backtesting and paper trading

- [ ] Replay raw market events in timestamp order.
- [ ] Simulate actions using only information observed by the decision time.
- [ ] Model bid/ask spread, trade fees, order latency, available size, partial fills, and cancellations.
- [ ] Track cash, positions, realized/unrealized P&L, and risk exposure.
- [ ] Compare net results against a no-trade baseline, including opportunity frequency.
- [ ] Validate on unseen periods and different activity regimes.
- [ ] Document assumptions that may make simulation more optimistic than live execution.

**Done when:** Paper-trading reports give net P&L, drawdown, fill assumptions, costs, and reproducible trade logs.

## Phase 11 — Risk controls and optional execution

- [ ] Live trading remains disabled by default.
- [ ] Implement maximum order size, per-market exposure, total exposure, and loss limits.
- [ ] Reject invalid, stale, or desynchronized books.
- [ ] Add duplicate-order prevention, order reconciliation, and a kill switch.
- [ ] Handle partial fills and unhedged positions explicitly.
- [ ] Test all order states in Kalshi's demo environment before any live use.
- [ ] Verify exchange rules, API permissions, trading fees, and regulatory requirements.

**Done when:** Risk controls are tested under forced disconnects, stale data, and partial fill conditions. Live use is a separate opt-in decision.

## Phase 12 — API and React dashboard

- [ ] Add FastAPI read-only endpoints for markets, features, forecasts, and simulated performance.
- [ ] Build React views for monitored markets, order-book state, price history, and 1/5/10-second forecasts.
- [ ] Display DOWN/FLAT/UP probabilities and distinguish simulated from actual results.
- [ ] Display arbitrage candidates with assumptions, fees, and depth.
- [ ] Display connection health, stale-data warnings, and system status.
- [ ] Keep API credentials server-side; never send private keys to the React frontend.

**Done when:** The dashboard can monitor running experiments without controlling real orders by default.

## Phase 13 — Documentation and portfolio

- [ ] README: installation, running collector, building data, training models, evaluating results.
- [ ] Architecture diagram and data dictionary.
- [ ] Explanation of target definition, time splits, leakage prevention, and feature engineering.
- [ ] Baseline-vs-GRU comparison, calibration plots, and test-set performance.
- [ ] Backtest methodology with fees, spreads, fills, and latency assumptions.
- [ ] Honest limitations and future experiments.
- [ ] Clean `.gitignore`, no private keys/secrets, and clear project history.

**Done when:** Another developer can reproduce the core results from sample data without secret credentials.

## Immediate next actions (do these in order)

1. [ ] In VS Code PowerShell, confirm you're in `C:\Users\jacob\kalshi-forecasting`.
2. [ ] Activate `.venv` and verify Python 3.12.
3. [ ] Install missing FastAPI/Uvicorn; run the successful import test.
4. [ ] Add this file as `PROJECT_ROADMAP.md` to project root.
5. [ ] Ensure `.env` is excluded from Git; commit this roadmap and environment config.
6. [ ] Create `src/kalshi/rest_client.py` and write a minimal public-market query.
7. [ ] Create one focused unit test using a saved example response.
8. [ ] Run the first REST query, inspect 5–10 active markets, and record the result in `docs/development_log.md`.
9. [ ] Implement the WebSocket collector and storage **before** training GRU models.

### Commands for the immediate milestone (PowerShell)

```powershell
cd C:\Users\jacob\kalshi-forecasting
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install fastapi uvicorn
python -c "import tensorflow, pandas, sklearn, websockets, fastapi; print('All imports successful')"
python -m pytest

git status
git add PROJECT_ROADMAP.md requirements.txt .gitignore .env.example
git commit -m "Add project roadmap and finalize environment setup"
git pull origin main --rebase
git push origin main
```

If `pytest` finds no tests yet, that is expected at the very beginning; create tests during Phase 2. If `git commit` reports nothing to commit, inspect `git status` rather than treating that as an error. If a pull encounters a conflict, resolve it before pushing. Never force-push just to overcome a rejected push.

## Milestone log — keep updating

| Date | Phase | Change / evidence | Next action |
|---|---|---|---|
| 2026-10-25 | 1 | TensorFlow/pandas/scikit-learn/websockets imports worked; FastAPI missing at last check | Install FastAPI; rerun complete import test |
| | | | |
| | | | |

## Reference links

- Kalshi developer docs: <https://docs.kalshi.com/>
- TensorFlow/Keras API: <https://www.tensorflow.org/api_docs/python/tf/keras>
- scikit-learn model evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Git documentation: <https://git-scm.com/doc>

**Definition of project completion:** Reliable collection + reproducible feature and labeling pipeline + leakage-aware baseline and GRU evaluation + fee-aware simulated strategy results + documented limitations. Real-money profitability is *not* a requirement for a successful research project.
