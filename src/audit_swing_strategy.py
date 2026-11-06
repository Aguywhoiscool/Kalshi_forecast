"""Independently verify saved swing signals, labels, exits and scenario totals."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.v3_research import digest, overlaps, save_json
from src.verify_swing_inputs import verify_derived_inputs, verify_exit_times


def audit(run):
    input_verification = verify_derived_inputs(run)
    protocol = json.loads((run / "protocol.json").read_text())
    source = Path(protocol["source"])
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    if digest(run / "sample_metadata.parquet") != protocol["sample_metadata_sha256"]:
        raise ValueError("Sample identity hash changed")
    hist = np.load(run / "history.npy", mmap_mode="r")
    future = np.load(run / "future_quotes.npy", mmap_mode="r")
    future_times = np.load(run / "future_times.npy", mmap_mode="r")
    x = np.load(run / "summary_features.npy", mmap_mode="r")
    protected = json.loads((source / "protected_v2_period.json").read_text())
    full_meta = pd.read_parquet(source / "windows.parquet")
    holdout = set(full_meta.loc[full_meta.split == "holdout", "session_id"])
    if meta.session_id.isin(holdout | set(protected["sessions"])).any():
        raise ValueError("Reserved session")
    if overlaps(meta.input_start.to_numpy(), meta.target_end.to_numpy(), protected).any():
        raise ValueError("Reserved time overlap")
    keys = ["session_id", "ticker", "input_end"]
    lookup = pd.MultiIndex.from_frame(meta[keys])
    if not lookup.is_unique:
        raise ValueError("Non-unique sample identities")
    positions = {}
    for name in ("fit", "calibration", "validation"):
        selected = pd.read_parquet(run / f"{name}_indices.parquet")
        rows = lookup.get_indexer(pd.MultiIndex.from_frame(selected[keys]))
        if (rows < 0).any() or len(np.unique(rows)) != len(rows):
            raise ValueError("Invalid split identities")
        positions[name] = rows
    fit, cal, val = (positions[name] for name in ("fit", "calibration", "validation"))
    if set(meta.iloc[fit].session_id) & set(meta.iloc[cal].session_id):
        raise ValueError("Fit/calibration session leakage")
    if meta.iloc[cal].input_start.min() < meta.iloc[fit].target_end.max() + 70 * 10**9:
        raise ValueError("Calibration purge violation")
    for name in ("fit", "calibration"):
        if not meta.iloc[positions[name]].split.eq("train").all():
            raise ValueError("Fitting outside original training split")
    if not meta.iloc[val].split.eq("val").all():
        raise ValueError("Invalid validation split")
    for item in json.loads((run / "source_files.json").read_text()):
        if digest(source / "snapshot" / item["file"]) != item["sha256"]:
            raise ValueError("Source snapshot changed")
    metrics = json.loads((run / "classification_metrics.json").read_text())
    probabilities = {}
    warnings = []
    for horizon in (30, 60):
        entry_yes, entry_no = future[:, 1, 1], 1 - future[:, 1, 0]
        yes_profit = (future[:, horizon, 0] - entry_yes) * 100
        no_profit = (1 - future[:, horizon, 1] - entry_no) * 100
        true_yes = (yes_profit >= 5 - 1e-9) & (future[:, 1, 3] >= 1) & (future[:, horizon, 2] >= 1)
        true_no = (no_profit >= 5 - 1e-9) & (future[:, 1, 2] >= 1) & (future[:, horizon, 3] >= 1)
        if (true_yes & true_no).any():
            raise ValueError("Impossible complementary outcomes")
        labels = np.where(true_yes, 1, np.where(true_no, 2, 0))
        np.testing.assert_array_equal(labels, np.load(run / f"labels_{horizon}.npy"))
        for model in ("prior", "logistic", "sgd", "decision_tree", "random_forest", "gradient_boosting", "gru", "monte_carlo"):
            with np.load(run / f"{model}_{horizon}_probabilities.npz") as saved:
                p_cal, p_val = saved["calibration"], saved["validation"]
            if p_cal.shape != (len(cal), 3) or p_val.shape != (len(val), 3):
                raise ValueError("Probability row alignment")
            np.testing.assert_allclose(p_val.sum(axis=1), 1, atol=1e-5)
            score_cal = p_cal[:, 1:].copy()
            score_cal[:, 0][hist[cal, -1, 5] < 1] = -1
            score_cal[:, 1][hist[cal, -1, 4] < 1] = -1
            scores = score_cal.max(axis=1)
            scores[score_cal[:, 0] == score_cal[:, 1]] = -1
            eligible = scores[scores >= 0]
            threshold = float(np.quantile(eligible, .99)) if len(eligible) else 1.
            np.testing.assert_allclose(threshold, metrics[f"{model}_{horizon}"]["calibration_threshold"], atol=1e-12)
            for side, name in ((1, "yes"), (2, "no")):
                observed = labels[val] == side
                if observed.any():
                    ap = average_precision_score(observed, p_val[:, side])
                    np.testing.assert_allclose(ap, metrics[f"{model}_{horizon}"]["event_metrics"][name]["average_precision"], atol=1e-12)
            probabilities[model, horizon] = p_val
            model_path = run / "models" / f"{model}_{horizon}.joblib"
            if model_path.exists():
                fitted = joblib.load(model_path)
                actual = np.zeros((len(val), 3))
                actual[:, fitted.classes_.astype(int)] = fitted.predict_proba(x[val])
                np.testing.assert_allclose(actual, p_val, rtol=0, atol=1e-12)
                if model in ("logistic", "sgd"):
                    scaler = fitted.named_steps["standardscaler"]
                    assert scaler.n_samples_seen_ == len(fit)
                    np.testing.assert_allclose(scaler.mean_, x[fit].mean(axis=0, dtype=np.float64), atol=1e-9)
                if model == "sgd" and fitted[-1].n_iter_ >= fitted[-1].max_iter:
                    warnings.append(f"SGD {horizon}s reached its iteration limit; treat it as a budget-limited baseline, not a converged optimum.")
        scaler = joblib.load(run / f"gru_{horizon}s/scaler.joblib")
        assert scaler.n_samples_seen_ == len(fit) * 60
        np.testing.assert_allclose(scaler.mean_, hist[fit].reshape(-1, 7).mean(axis=0), rtol=1e-10, atol=1e-9)
    results = pd.read_csv(run / "strategy_results.csv")
    primary = results.loc[(results.entry_delay_observations == 1) & (results.cost_buffer_cents == 4)]
    audited = 0
    val_lookup = pd.MultiIndex.from_frame(meta.iloc[val][keys])
    for row in primary.itertuples():
        trades = pd.read_parquet(run / "trades" / f"{row.model}_{row.horizon_seconds}_{row.entry_rule}_{row.exit_policy}.parquet")
        if row.model == "no_trade":
            assert trades.empty and row.trades == 0 and row.total_net_scenario_cents == 0
            continue
        if row.model == "volatility_momentum":
            threshold = np.quantile(x[cal, 49], .99)
            momentum = hist[val, -1, 3] - hist[val, -11, 3]
            sides = np.where(momentum > 0, 1, 2)
            scores = np.asarray(x[val, 49]).copy()
            scores[momentum == 0] = -1
            mask = scores > threshold
        else:
            p = probabilities[row.model, row.horizon_seconds]
            scores2 = p[:, 1:].copy()
            scores2[:, 0][hist[val, -1, 5] < 1] = -1
            scores2[:, 1][hist[val, -1, 4] < 1] = -1
            sides = np.where(scores2[:, 0] > scores2[:, 1], 1, 2)
            scores = scores2.max(axis=1)
            scores[scores2[:, 0] == scores2[:, 1]] = -1
            threshold = metrics[f"{row.model}_{row.horizon_seconds}"]["calibration_threshold"]
            mask = scores >= .5 if row.entry_rule == "p50" else scores > threshold
        selected = np.flatnonzero(mask)
        stored = val_lookup.get_indexer(pd.MultiIndex.from_frame(trades[keys]))
        np.testing.assert_array_equal(stored, selected)
        verify_exit_times(trades, future_times, val[stored])
        np.testing.assert_array_equal(trades.side, sides[selected])
        net, filled = [], []
        for local_index, trade in zip(selected, trades.itertuples()):
            index = val[local_index]
            quotes = future[index]
            yes = trade.side == 1
            entry = quotes[1, 1] if yes else 1 - quotes[1, 0]
            entry_depth = quotes[1, 3] if yes else quotes[1, 2]
            liquidation = quotes[:, 0] if yes else 1 - quotes[:, 1]
            path = (liquidation - entry) * 100
            exit_index = row.horizon_seconds
            if row.exit_policy == "bracket":
                hits = np.flatnonzero((path[2:exit_index + 1] >= 10 - 1e-9) | (path[2:exit_index + 1] <= -5 + 1e-9))
                if len(hits):
                    exit_index = 2 + int(hits[0])
            assert trade.exit_index == exit_index
            exit_depth = quotes[exit_index, 2] if yes else quotes[exit_index, 3]
            assert trade.unresolved_exit == (exit_depth < 1)
            expected = (-entry * 100 if exit_depth < 1 else path[exit_index]) - 4 if entry_depth >= 1 else 0.
            np.testing.assert_allclose(trade.net_conservative_cents, expected, atol=1e-9)
            net.append(expected)
            filled.append(entry_depth >= 1)
            audited += 1
        filled = np.asarray(filled, dtype=bool)
        net = np.asarray(net)[filled]
        assert row.trades == len(net)
        np.testing.assert_allclose(row.total_net_scenario_cents, net.sum(), atol=1e-9)
        if len(net):
            np.testing.assert_allclose(row.mean_net_scenario_cents, net.mean(), atol=1e-9)
    record = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "validated_signal_trade_records": audited, **input_verification, "checks": [
            "All histories, summary features, future quotes and timestamps reconstructed from source CSVs",
            "Saved trade exit timestamps match their source observations",
            "Source hashes, sample identities, reserved exclusions, chronological calibration purge",
            "Both fixed-horizon labels independently reproduced from delayed entry/exit quotes and depth",
            "Saved tabular predictions reproduced; scaler statistics verified against fit only",
            "Class AP and calibration-only thresholds reproduced",
            "Primary cost/delay signal selection, sides, first-crossing exits and aggregate P&L independently reproduced"],
        "warnings": warnings, "scope": "4c cost / one-observation delay for both predeclared entry and exit rules; other sensitivities covered by unit tests"}
    save_json(run / "audit.json", record)
    with (run / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Independent audit\n\nPassed saved-data, prediction, selection, and primary-scenario execution checks.\n")
        for warning in warnings:
            stream.write("- " + warning + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/swing_strategy_20261104"))
    args = parser.parse_args()
    audit(args.run_dir)
