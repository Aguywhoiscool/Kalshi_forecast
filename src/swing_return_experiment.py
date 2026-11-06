"""Past-only swing entries with expected conservative net-return forecasts.

Unobserved outcomes remain in the cohort with explicit payoff bounds. No orders.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from src.causal_swing_data import HORIZONS, prepare_causal
from src.swing_strategy_experiment import inner_split
from src.v3_research import digest, save_json
from src.verify_swing_inputs import verify_derived_inputs


COST = 4.
MARGINS = (0., 1., 2.)
MODEL_NAMES = {"mean": "Mean-return baseline", "ridge": "Ridge regression", "sgd": "SGD regression",
    "decision_tree": "Decision tree", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
    "gru": "GRU", "monte_carlo": "Monte Carlo", "no_trade": "No trade"}
STATUS = {0: "closed", 1: "rejected_entry", 2: "unknown_entry", 3: "missing_exit_quote", 4: "insufficient_exit_depth"}


def return_bounds(future, available, horizon, cost=COST):
    """Bounds for hypothetical one-contract, best-level all-or-none execution.

    Unknown entry: worst-case -100c-cost, best-case +100c-cost. Known rejected
    entry: zero. Unresolved exit: contract value anywhere in [0,1], not a fill.
    """
    future = np.asarray(future, dtype=float)
    available = np.asarray(available, dtype=bool)
    n = len(future)
    if future.shape != (n, 61, 4) or available.shape != (n, 61) or horizon not in HORIZONS:
        raise ValueError("Invalid quote/availability shape or horizon")
    lower, upper = np.full((n, 2), -100. - cost), np.full((n, 2), 100. - cost)
    status = np.full((n, 2), 2, dtype=np.int8)
    entry_prices = np.column_stack([future[:, 1, 1], 1 - future[:, 1, 0]])
    exit_values = np.column_stack([future[:, horizon, 0], 1 - future[:, horizon, 1]])
    entry_depth = future[:, 1, [3, 2]]
    exit_depth = future[:, horizon, [2, 3]]
    for side in range(2):
        entry_known = available[:, 1]
        rejected = entry_known & (entry_depth[:, side] < 1)
        held = entry_known & ~rejected
        closed = held & available[:, horizon] & (exit_depth[:, side] >= 1)
        missing = held & ~available[:, horizon]
        thin = held & available[:, horizon] & (exit_depth[:, side] < 1)
        lower[rejected, side] = upper[rejected, side] = 0.
        lower[held, side] = -entry_prices[held, side] * 100 - cost
        upper[held, side] = (1 - entry_prices[held, side]) * 100 - cost
        gross = (exit_values[closed, side] - entry_prices[closed, side]) * 100
        lower[closed, side] = upper[closed, side] = gross - cost
        status[closed, side], status[rejected, side] = 0, 1
        status[missing, side], status[thin, side] = 3, 4
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
        raise ValueError("Invalid outcome bounds")
    return lower, upper, status


def expected_entry(prediction, history, margin=1.):
    """Return choice is based on expected net scenario value, not P(big win)."""
    value = np.asarray(prediction, dtype=float).copy()
    if value.shape != (len(history), 2) or not np.isfinite(value).all():
        raise ValueError("Invalid expected-return prediction")
    value[history[:, -1, 5] < 1, 0] = -np.inf
    value[history[:, -1, 4] < 1, 1] = -np.inf
    side = np.argmax(value, axis=1)
    score = value[np.arange(len(value)), side]
    selected = (score >= margin) & (value[:, 0] != value[:, 1])
    return side, score, selected


def evaluate(model, horizon, prediction, history, meta, future, available, margin, cost=COST):
    sides, scores, mask = expected_entry(prediction, history, margin)
    lower, upper, status = return_bounds(future, available, horizon, cost)
    blocked, busy = set(), {}
    records = []
    for position in np.flatnonzero(mask):
        row = meta.iloc[position]
        side = int(sides[position])
        timestamp = int(row.input_end)
        # Once a simulated position is uncertain, don't assume it disappeared
        # before permitting a later position in the same market.
        cannot_enter = row.ticker in blocked or timestamp <= busy.get(row.ticker, -1)
        state = int(status[position, side])
        lo, hi = (0., 0.) if cannot_enter else (float(lower[position, side]), float(upper[position, side]))
        if not cannot_enter:
            busy[row.ticker] = timestamp + horizon * 10**9 + 500_000_000
            if state >= 2:
                blocked.add(row.ticker)
        records.append({"session_id": row.session_id, "ticker": row.ticker,
            "input_end": timestamp, "side": side + 1, "predicted_net_cents": float(scores[position]),
            "status": "blocked_uncertain_or_pending" if cannot_enter else STATUS[state],
            "net_lower_cents": lo, "net_upper_cents": hi,
            "planned_exit_time_ns": timestamp + horizon * 10**9,
            "model": model, "horizon": horizon, "margin_cents": margin, "cost_buffer_cents": cost})
    columns = ["session_id", "ticker", "input_end", "side", "predicted_net_cents", "status",
        "net_lower_cents", "net_upper_cents", "planned_exit_time_ns", "model", "horizon", "margin_cents", "cost_buffer_cents"]
    trades = pd.DataFrame(records, columns=columns)
    active = trades.loc[~trades.status.eq("blocked_uncertain_or_pending")]
    closed = active.loc[active.status.eq("closed")]
    result = {"model": model, "horizon": horizon, "margin_cents": margin, "cost_buffer_cents": cost,
        "signals": len(trades), "blocked": len(trades) - len(active), "attempts": len(active),
        "closed": len(closed), "rejected_entries": int(active.status.eq("rejected_entry").sum()),
        "unknown_entries": int(active.status.eq("unknown_entry").sum()),
        "unresolved_exits": int(active.status.isin(["missing_exit_quote", "insufficient_exit_depth"]).sum()),
        "net_lower_total_cents": float(active.net_lower_cents.sum()),
        "net_upper_total_cents": float(active.net_upper_cents.sum()),
        "closed_quote_net_cents": float(closed.net_lower_cents.sum())}
    return trades, result


def regressors():
    return {"mean": DummyRegressor(), "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
        "sgd": make_pipeline(StandardScaler(), MultiOutputRegressor(SGDRegressor(
            loss="squared_error", alpha=.001, learning_rate="invscaling", eta0=.001,
            max_iter=2000, tol=.001, average=True, random_state=42))),
        "decision_tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=30, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=64, max_depth=12,
            min_samples_leaf=20, n_jobs=2, random_state=42),
        "gradient_boosting": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, l2_regularization=1., early_stopping=False, random_state=42))}


def mc_expected(histories, metadata, paths=2048):
    predictions = {h: np.zeros((len(histories), 2)) for h in HORIZONS}
    for index, row in enumerate(metadata.itertuples()):
        past = histories[index]
        spread = past[-1, 1] - past[-1, 0]
        increments = np.diff(past[:, 3])
        mid = np.full(paths, past[-1, 3])
        import hashlib
        seed = int.from_bytes(hashlib.sha256(f"return42|{row.session_id}|{row.ticker}|{row.input_end}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        for step in range(1, 61):
            mid = np.clip(mid + rng.choice(increments, size=paths), spread / 2, 1 - spread / 2)
            if step == 1:
                entry_yes, entry_no = mid + spread / 2, 1 - (mid - spread / 2)
            if step in HORIZONS:
                predictions[step][index] = [float(((mid - spread / 2) - entry_yes).mean() * 100 - COST),
                    float(((1 - mid - spread / 2) - entry_no).mean() * 100 - COST)]
    return predictions


def train_gru(histories, target, fit, cal, val, output, horizon, progress, epochs):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    folder = output / f"gru_{horizon}"
    folder.mkdir()
    scaler = StandardScaler()
    for start in range(0, len(fit), 256):
        batch = histories[fit[start:start + 256]]
        scaler.partial_fit(batch.reshape(-1, 7))
    joblib.dump(scaler, folder / "scaler.joblib")

    class Batches(tf.keras.utils.Sequence):
        def __init__(self, ids, labels=True, shuffle=False):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.ids, self.labels, self.shuffle = ids.copy(), labels, shuffle
            self.rng = np.random.default_rng(42)
        def __len__(self):
            return (len(self.ids) + 255) // 256
        def __getitem__(self, index):
            ids = self.ids[index * 256:(index + 1) * 256]
            raw = histories[ids]
            x = scaler.transform(raw.reshape(-1, 7)).reshape(raw.shape).astype(np.float32)
            return (x, (target[ids] / 100).astype(np.float32)) if self.labels else x
        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.ids)

    model = tf.keras.Sequential([tf.keras.layers.Input((60, 7)),
        tf.keras.layers.GRU(64, return_sequences=True), tf.keras.layers.Dropout(.2),
        tf.keras.layers.GRU(32), tf.keras.layers.Dropout(.2), tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse")
    progress("training_gru", horizon=horizon, fit=len(fit), calibration=len(cal))
    model.fit(Batches(fit, True, True), validation_data=Batches(cal), epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            tf.keras.callbacks.ModelCheckpoint(str(folder / "best.keras"), save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.CSVLogger(str(folder / "epochs.csv"))])
    model = tf.keras.models.load_model(folder / "best.keras")
    result = (model.predict(Batches(cal, False), verbose=0) * 100,
              model.predict(Batches(val, False), verbose=0) * 100)
    tf.keras.backend.clear_session()
    return result


def report(output, rows, metrics):
    save_json(output / "metrics.json", metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "strategy_results.csv", index=False)
    primary = frame.loc[(frame.cost_buffer_cents == COST) & (frame.margin_cents == 1.)]
    lines = ["# Past-only expected-return experiment", "",
        "Primary entry: predicted conservative net scenario return >=1 cent after a 4-cent cost buffer.",
        "Approximately 1-second delayed entry; fixed 30/60-second exit. No probability-of-win gate.",
        "Inputs and entry population depend on history and fixed administrative boundaries only.", "",
        "| Model | Horizon | Attempts | Closed | Unknown entry | Unresolved exit | Net lower cents | Net upper cents |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {MODEL_NAMES[row.model]} | {row.horizon}s | {row.attempts} | {row.closed} | {row.unknown_entries} | {row.unresolved_exits} | {row.net_lower_total_cents:+.3f} | {row.net_upper_total_cents:+.3f} |")
    lines += ["", "## Scope", "- Return targets include loss magnitude and unknown-outcome conservative bounds.",
        "- A lower scenario bound is not a guaranteed expected return: the predictive model can be wrong.",
        "- Missing future quotes never remove an entry. Histories stay inside one connection session; no book/session stitching.",
        "- Quote times match nominal schedule within 0.5 seconds. Missing entry means unknown execution, not a verified rejection.",
        "- Unknown exits span contract value [0,1]; no zero-price fill is claimed. A market is blocked after an uncertain position.",
        "- Positive closed-only P&L cannot establish profit while uncertain positions remain.",
        "- Cost buffers are scenarios, not actual market-specific fees; quoted prices do not guarantee fills or queue priority.",
        "- Regression fits only original training. Checkpoints and global mean-bias corrections use chronological calibration inside training.",
        "- No reserved outcomes evaluated; no live orders. Validation remains exploratory and repeatedly used in project development.",
        "- All margins/cost variants were predeclared; no validation-based best-variant selection."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output, epochs, source=None):
    root, output = Path.cwd().resolve(), output.resolve()
    if root not in output.parents or root / "data" in output.parents or output.exists():
        raise ValueError("Use a new artifact directory outside data/")
    source = (Path(source) if source is not None else Path(json.loads(
        (root / "artifacts/model_lab_latest.json").read_text())["run_dir"])).resolve()
    output.mkdir(parents=True)
    (output / "models").mkdir()
    (output / "trades").mkdir()
    save_json(output / "process.json", {"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()})

    def progress(stage, **extra):
        state = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
        save_json(output / "status.json", state)
        print(json.dumps(state), flush=True)

    protocol = {"source": str(source), "dataset_policy": "past_only_v1", "horizons": HORIZONS,
        "primary_cost_buffer_cents": COST, "primary_expected_net_margin_cents": 1.,
        "margins_cents": MARGINS, "cost_buffers_cents": [0, 2, 4, 5], "epochs": epochs, "seed": 42,
        "target": "Expected conservative net scenario return on one contract, including uncertain-outcome bounds",
        "entry": "Nearest quote to signal+1 second, tolerance 0.5 seconds; unknown if absent",
        "exit": "Nearest quote to signal+horizon, tolerance 0.5 seconds; unknown if absent/insufficient top depth",
        "eligibility": "Valid 60-observation history only; horizon-specific fixed split/time protection; no future coverage filter",
        "calibration": "Global mean residual correction per side inside original training; no validation tuning",
        "missing_outcome_policy": "Known entry: contract value bound [0,1]; unknown entry: net bound [-100-cost,100-cost] cents",
        "source_manifest_sha256": digest(source / "input_manifest.json"),
        "future_session_policy": "Same session only; missing future records remain unknown, never stitched",
        "limitations": "Cost scenarios and hypothetical quote/depth fills; conservative mark-down targets are not actual realized profits"}
    save_json(output / "protocol.json", protocol)
    try:
        meta, histories, future, clocks, available, x = prepare_causal(source, output, progress)
        protocol["sample_metadata_sha256"] = digest(output / "sample_metadata.parquet")
        save_json(output / "protocol.json", protocol)
        progress("verifying_sources")
        verify_derived_inputs(output)
        fit_base, cal_base, val_base = inner_split(meta)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        rows, metrics = [], {}
        for horizon in HORIZONS:
            eligible = meta[f"eligible_{horizon}"].to_numpy()
            fit, cal, val = (ids[eligible[ids]] for ids in (fit_base, cal_base, val_base))
            for name, ids in (("fit", fit), ("calibration", cal), ("validation", val)):
                meta.iloc[ids].to_parquet(output / f"{name}_{horizon}.parquet", index=False)
            target, upper, status = return_bounds(future, available, horizon)
            np.savez_compressed(output / f"targets_{horizon}.npz", lower=target, upper=upper, status=status)
            combined = np.r_[cal, val]
            progress("monte_carlo", horizon=horizon, fit=len(fit), calibration=len(cal), validation=len(val))
            mc = mc_expected(histories[combined], meta.iloc[combined].reset_index(drop=True))[horizon]
            for name in [*regressors(), "gru", "monte_carlo", "no_trade"]:
                progress("fitting", model=name, horizon=horizon)
                training_warnings = []
                if name == "no_trade":
                    prediction = np.full((len(val), 2), -1000.)
                else:
                    if name == "gru":
                        raw_cal, raw_val = train_gru(histories, target, fit, cal, val, output, horizon, progress, epochs)
                    elif name == "monte_carlo":
                        raw_cal, raw_val = mc[:len(cal)], mc[len(cal):]
                    else:
                        model = regressors()[name]
                        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
                            warnings.simplefilter("always")
                            model.fit(x[fit], target[fit])
                            raw_cal, raw_val = model.predict(x[cal]), model.predict(x[val])
                        training_warnings = [str(item.message) for item in captured]
                        joblib.dump(model, output / "models" / f"{name}_{horizon}.joblib")
                    raw_cal, raw_val = np.clip(raw_cal, -104, 96), np.clip(raw_val, -104, 96)
                    correction = (target[cal] - raw_cal).mean(axis=0)
                    prediction = np.clip(raw_val + correction, -104, 96)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Non-finite net-return prediction")
                    np.savez_compressed(output / f"{name}_{horizon}_predictions.npz",
                        raw_calibration=raw_cal, raw_validation=raw_val, correction=correction, validation=prediction)
                    metrics[f"{name}_{horizon}"] = {"fit": len(fit), "calibration": len(cal), "validation": len(val),
                        "mae_lower_cents": float(mean_absolute_error(target[val], prediction)),
                        "mse_lower": float(mean_squared_error(target[val], prediction)),
                        "calibration_correction_cents": correction.tolist(), "training_warnings": training_warnings}
                for margin in MARGINS:
                    for cost in (0., 2., 4., 5.):
                        trades, result = evaluate(name, horizon, prediction, histories[val], meta.iloc[val].reset_index(drop=True),
                            future[val], available[val], margin, cost)
                        rows.append(result)
                        if cost == COST:
                            trades.to_parquet(output / "trades" / f"{name}_{horizon}_margin{margin:g}.parquet", index=False)
                report(output, rows, metrics)
                progress("model_complete", model=name, horizon=horizon)
        report(output, rows, metrics)
        progress("auditing")
        from src.audit_swing_returns import audit
        audit(output)
        save_json(root / "artifacts/swing_return_latest.json", {"run": str(output), "report": str(output / "REPORT.md")})
        progress("complete", report=str(output / "REPORT.md"))
    except BaseException:
        progress("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/swing_returns_20261106"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--source", type=Path, help="Explicit immutable development source")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.output, args.epochs, args.source)
