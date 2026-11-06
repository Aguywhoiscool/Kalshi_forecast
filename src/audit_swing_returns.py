"""Source, split, expected-return and outcome-bound checks for causal runs."""

from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd

from src.v3_research import overlaps, save_json
from src.verify_swing_inputs import equal, verify_derived_inputs


def independent_bounds(quotes, known, horizon, cost=4.):
    result = []
    for side in (0, 1):
        lower, upper, states = [], [], []
        for q, valid in zip(quotes, known):
            if not valid[1]:
                lo, hi, state = -100-cost, 100-cost, "unknown_entry"
            else:
                price = q[1, 1] if side == 0 else 1-q[1, 0]
                depth = q[1, 3] if side == 0 else q[1, 2]
                exit_depth = q[horizon, 2] if side == 0 else q[horizon, 3]
                if depth < 1:
                    lo, hi, state = 0., 0., "rejected_entry"
                elif not valid[horizon] or exit_depth < 1:
                    lo, hi = -100*price-cost, 100*(1-price)-cost
                    state = "missing_exit_quote" if not valid[horizon] else "insufficient_exit_depth"
                else:
                    liquidation = q[horizon, 0] if side == 0 else 1-q[horizon, 1]
                    lo = hi = 100*(liquidation-price)-cost
                    state = "closed"
            lower.append(lo); upper.append(hi); states.append(state)
        result.append((lower, upper, states))
    return (np.array([r[0] for r in result]).T,
            np.array([r[1] for r in result]).T,
            np.array([r[2] for r in result]).T)


def audit(run):
    source_checks = verify_derived_inputs(run)
    protocol = json.loads((run / "protocol.json").read_text())
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    from pathlib import Path
    source = Path(protocol["source"])
    original = pd.read_parquet(source / "windows.parquet")
    protected = json.loads((source / "protected_v2_period.json").read_text())
    if meta.session_id.isin(set(protected["sessions"]) | set(original.loc[original.split == "holdout", "session_id"])).any():
        raise ValueError("Reserved session")
    if overlaps(meta.input_start.to_numpy(), meta.target_end.to_numpy(), protected).any():
        raise ValueError("Reserved time")
    for split, part in meta.groupby("split"):
        bounds = original.loc[original.split == split]
        if bounds.empty or split not in ("train", "val"):
            raise ValueError("Unknown development split")
        if (part.input_start < bounds.input_start.min()).any() or (part.target_end > bounds.target_end.max()).any():
            raise ValueError("Development boundary violation")
    hist = np.load(run / "history.npy", mmap_mode="r")
    future = np.load(run / "future_quotes.npy", mmap_mode="r")
    known = np.load(run / "future_valid.npy", mmap_mode="r")
    x = np.load(run / "summary_features.npy", mmap_mode="r")
    lookup = pd.MultiIndex.from_frame(meta[["session_id", "ticker", "input_end"]])
    if not lookup.is_unique:
        raise ValueError("Duplicate entries")
    metrics = json.loads((run / "metrics.json").read_text())
    summaries = pd.read_csv(run / "strategy_results.csv")
    audited = 0
    for horizon in (30, 60):
        positions = {}
        for name in ("fit", "calibration", "validation"):
            part = pd.read_parquet(run / f"{name}_{horizon}.parquet")
            ids = lookup.get_indexer(pd.MultiIndex.from_frame(part[["session_id", "ticker", "input_end"]]))
            if (ids < 0).any() or len(np.unique(ids)) != len(ids):
                raise ValueError("Invalid split identities")
            positions[name] = ids
        fit, cal, val = (positions[name] for name in ("fit", "calibration", "validation"))
        expected_val = np.flatnonzero((meta.split == "val") & meta[f"eligible_{horizon}"])
        equal(val, expected_val, "complete past-only validation population")
        if not meta.iloc[fit].split.eq("train").all() or not meta.iloc[cal].split.eq("train").all():
            raise ValueError("Model fitting outside training")
        if set(meta.iloc[fit].session_id) & set(meta.iloc[cal].session_id):
            raise ValueError("Fit/calibration session overlap")
        if meta.iloc[cal].input_start.min() < meta.iloc[fit].target_end.max() + 70*10**9:
            raise ValueError("Fit/calibration temporal overlap")
        lower, upper, states = independent_bounds(future, known, horizon)
        with np.load(run / f"targets_{horizon}.npz") as target:
            np.testing.assert_allclose(lower, target["lower"], atol=1e-9)
            np.testing.assert_allclose(upper, target["upper"], atol=1e-9)
        for model in ("mean", "ridge", "sgd", "decision_tree", "random_forest", "gradient_boosting", "gru", "monte_carlo", "no_trade"):
            if model == "no_trade":
                prediction = np.full((len(val), 2), -1000.)
            else:
                with np.load(run / f"{model}_{horizon}_predictions.npz") as saved:
                    raw_cal, raw_val, correction, prediction = (saved[key] for key in
                        ("raw_calibration", "raw_validation", "correction", "validation"))
                np.testing.assert_allclose(correction, (lower[cal]-raw_cal).mean(axis=0), atol=1e-10)
                np.testing.assert_allclose(prediction, np.clip(raw_val+correction, -104, 96), atol=1e-10)
                path = run / "models" / f"{model}_{horizon}.joblib"
                if path.exists():
                    fitted = joblib.load(path)
                    np.testing.assert_allclose(raw_val, np.clip(fitted.predict(x[val]), -104, 96), atol=1e-10)
                    np.testing.assert_allclose(raw_cal, np.clip(fitted.predict(x[cal]), -104, 96), atol=1e-10)
                    if model in ("ridge", "sgd"):
                        scaler = fitted.named_steps["standardscaler"]
                        if scaler.n_samples_seen_ != len(fit):
                            raise ValueError("Scaler count is not fit-only")
                        np.testing.assert_allclose(scaler.mean_, x[fit].mean(axis=0, dtype=np.float64), atol=1e-9)
                details = metrics[f"{model}_{horizon}"]
                np.testing.assert_allclose(details["mae_lower_cents"], np.abs(lower[val]-prediction).mean(), atol=1e-9)
            values = prediction.copy()
            values[hist[val, -1, 5] < 1, 0] = -np.inf
            values[hist[val, -1, 4] < 1, 1] = -np.inf
            sides = values.argmax(axis=1)
            scores = values[np.arange(len(val)), sides]
            chosen = (scores >= 1.) & (values[:, 0] != values[:, 1])
            positions_chosen = np.flatnonzero(chosen)
            trades = pd.read_parquet(run / "trades" / f"{model}_{horizon}_margin1.parquet")
            expected_keys = meta.iloc[val[positions_chosen]][["session_id", "ticker", "input_end"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(trades[["session_id", "ticker", "input_end"]].reset_index(drop=True), expected_keys)
            blocked, busy = set(), {}
            for local, trade in zip(positions_chosen, trades.itertuples()):
                index, side = val[local], int(sides[local])
                row = meta.iloc[index]
                cannot_enter = row.ticker in blocked or row.input_end <= busy.get(row.ticker, -1)
                state = "blocked_uncertain_or_pending" if cannot_enter else states[index, side]
                if not cannot_enter:
                    busy[row.ticker] = int(row.input_end) + horizon*10**9 + 500_000_000
                    if state not in ("closed", "rejected_entry"):
                        blocked.add(row.ticker)
                if trade.status != state or trade.side != side+1:
                    raise ValueError("Causal position state or direction mismatch")
                if trade.planned_exit_time_ns != row.input_end+horizon*10**9:
                    raise ValueError("Wrong scheduled exit timestamp")
                expected = [0., 0.] if cannot_enter else [lower[index, side], upper[index, side]]
                np.testing.assert_allclose([trade.net_lower_cents, trade.net_upper_cents], expected, atol=1e-9)
                audited += 1
            summary = summaries.loc[(summaries.model == model) & (summaries.horizon == horizon)
                & (summaries.margin_cents == 1) & (summaries.cost_buffer_cents == 4)].iloc[0]
            np.testing.assert_allclose(summary.net_lower_total_cents, trades.net_lower_cents.sum(), atol=1e-9)
            np.testing.assert_allclose(summary.net_upper_total_cents, trades.net_upper_cents.sum(), atol=1e-9)
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
        **source_checks, "primary_signal_records_verified": audited,
        "checks": ["Derived arrays reconstructed from frozen CSVs", "Past-only validation cohort retained, including missing future quotes",
            "Reserved sessions/time ranges and development boundaries", "Chronological fit/calibration separation",
            "Independent interval-valued net-return targets", "Calibration-only bias correction and saved tabular predictions",
            "Expected-return entry gate, uncertain-position blocking, scheduled exit clocks and P&L bounds"],
        "scope": "Primary 1c margin / 4c cost; source arrays all examples. GRU/MC prediction internals are covered by source/scaler tests, not independent refitting."}
    save_json(run / "audit.json", result)
    with (run / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Audit\n\nPassed source reconstruction, missing-outcome retention, split, prediction-correction and primary execution-bound checks. See audit.json.\n")
    print(json.dumps(result, indent=2), flush=True)
