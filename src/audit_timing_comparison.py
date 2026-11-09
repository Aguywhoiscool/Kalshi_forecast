"""Independent source/identity/prediction checks for paired timing variants."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.audit_swing_returns import independent_bounds
from src.verify_swing_inputs import verify_derived_inputs
from src.v3_research import digest, save_json


def audit(base, exclude_near=False):
    base = base.resolve()
    output = base / "timing_comparison"
    source_checks = verify_derived_inputs(base)
    parent = json.loads((base / "protocol.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    if digest(output / "identities.parquet") != protocol["identities_sha256"]:
        raise ValueError("Paired identities changed")
    source = Path(parent["source"])
    all_meta = pd.read_parquet(base / "sample_metadata.parquet")
    meta = pd.read_parquet(output / "identities.parquet")
    ids = meta.original_row.to_numpy(dtype=int)
    if (ids < 0).any() or (ids >= len(all_meta)).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid or duplicated original row identities")
    pd.testing.assert_frame_equal(meta[["session_id", "ticker", "input_end"]],
        all_meta.iloc[ids][["session_id", "ticker", "input_end"]].reset_index(drop=True))
    if not (all_meta.iloc[ids].eligible_30 & all_meta.iloc[ids].eligible_60).all():
        raise ValueError("Common horizon eligibility violated")
    # Reconstruct timing independently of the materializer and reject knowledge
    # arriving later than each individual source observation.
    timing = np.empty((len(meta), 60))
    for session, selected in meta.groupby("session_id"):
        frame = pd.read_csv(source / "snapshot" / f"features_{session}.csv")
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
        for ticker, rows in selected.groupby("ticker"):
            group = frame.loc[frame.ticker == ticker].sort_values("timestamp").reset_index(drop=True)
            times = group.timestamp.array.as_unit("ns").asi8
            ends = np.searchsorted(times, rows.input_end.to_numpy())
            indices = ends[:, None]+np.arange(-59, 1)
            observations = pd.to_datetime(group.close_time_observed_at, utc=True, errors="coerce", format="ISO8601")
            closes = pd.to_datetime(group.scheduled_close_time, utc=True, errors="coerce", format="ISO8601")
            seconds = pd.to_numeric(group.seconds_to_scheduled_close, errors="coerce").to_numpy()
            valid = observations.notna() & closes.notna() & (observations <= group.timestamp)
            expected = (closes-group.timestamp).dt.total_seconds().to_numpy()
            if not valid.to_numpy()[indices].all() or not np.isfinite(seconds[indices]).all():
                raise ValueError("Unverified timing observation")
            np.testing.assert_allclose(seconds[indices], expected[indices], atol=.001, rtol=0)
            timing[rows.index] = seconds[indices]
    from src.swing_strategy_experiment import inner_split
    fit, cal, val = inner_split(meta)
    for left, right in ((fit, cal), (fit, val), (cal, val)):
        if set(meta.iloc[left].session_id) & set(meta.iloc[right].session_id):
            raise ValueError("Paired session leakage")
    train = meta.loc[meta.split == "train"]
    validation = meta.loc[meta.split == "val"]
    if validation.input_start.min() < train.target_end.max()+70*10**9:
        raise ValueError("Paired validation embargo violation")
    hist = np.load(base / "history.npy", mmap_mode="r")[ids]
    future = np.load(base / "future_quotes.npy", mmap_mode="r")[ids]
    known = np.load(base / "future_valid.npy", mmap_mode="r")[ids]
    from src.model_benchmark import summarize
    from src.swing_return_experiment import evaluate
    base_x = summarize(hist)
    features8 = np.column_stack([base_x, timing[:, -1], timing.mean(axis=1), timing.std(axis=1),
        timing.min(axis=1), timing.max(axis=1), timing[:, -1]-timing[:, 0]]).astype(np.float32)
    checked = 0
    variants = (("exclude_near", base_x),) if exclude_near else (("seven", base_x), ("eight", features8))
    for variant, features in variants:
        folder = output / variant
        fit_used = fit
        if exclude_near:
            near = (timing[fit, -1] > 600) & (timing[fit, -1] <= 3600)
            fit_used = fit[~near]
            for name, expected in (("fit", fit_used), ("calibration", cal), ("validation", val)):
                np.testing.assert_array_equal(np.load(folder / f"{name}_rows.npy"), expected)
            supplement = json.loads((folder / "protocol.json").read_text())
            if supplement["identities_sha256"] != protocol["identities_sha256"]:
                raise ValueError("Supplemental parent identity mismatch")
            if supplement["counts"]["fit"] != len(fit_used) or len(fit_used) == 0:
                raise ValueError("Supplemental fit exclusion/count mismatch")
        sequences = hist if variant != "eight" else np.concatenate([hist, timing[:, :, None]], axis=2)
        for horizon in (30, 60):
            lower, upper, states = independent_bounds(future, known, horizon)
            if len(pd.read_parquet(folder / "trades" / f"no_trade_{horizon}.parquet")):
                raise ValueError("No-trade baseline unexpectedly entered a trade")
            names = ("mean", "ridge", "sgd", "decision_tree", "random_forest", "gradient_boosting", "gru", "monte_carlo")
            for model_name in names:
                path = folder / f"{model_name}_{horizon}_predictions.npz"
                if not path.is_file():
                    raise ValueError("Missing paired prediction file: "+str(path))
                with np.load(path) as saved:
                    correction, rawcal, rawval, prediction = (saved[k] for k in (
                        "correction", "raw_calibration", "raw_validation", "validation"))
                np.testing.assert_allclose(correction, (lower[cal]-rawcal).mean(axis=0), atol=1e-9)
                np.testing.assert_allclose(prediction, np.clip(rawval+correction, -104, 96), atol=1e-9)
                model_path = folder / "models" / f"{model_name}_{horizon}.joblib"
                if model_path.exists():
                    model = joblib.load(model_path)
                    np.testing.assert_allclose(rawval, np.clip(model.predict(features[val]), -104, 96), atol=1e-9)
                    np.testing.assert_allclose(rawcal, np.clip(model.predict(features[cal]), -104, 96), atol=1e-9)
                    if model_name in ("ridge", "sgd"):
                        scaler = model.named_steps["standardscaler"]
                        if scaler.n_samples_seen_ != len(fit_used):
                            raise ValueError("Paired scaler training count mismatch")
                        np.testing.assert_allclose(scaler.mean_, features[fit_used].mean(axis=0, dtype=np.float64), atol=1e-9)
                elif model_name == "gru":
                    scaler = joblib.load(folder / f"gru_{horizon}/scaler.joblib")
                    if scaler.n_samples_seen_ != 60*len(fit_used) or scaler.n_features_in_ != sequences.shape[2]:
                        raise ValueError("GRU scaler identity/count mismatch")
                    np.testing.assert_allclose(scaler.mean_, sequences[fit_used].reshape(-1, sequences.shape[2]).mean(axis=0), atol=1e-8)
                    import tensorflow as tf
                    model = tf.keras.models.load_model(folder / f"gru_{horizon}/best.keras", compile=False)
                    for indices, raw in ((cal[:3], rawcal[:3]), (val[:3], rawval[:3])):
                        batch = sequences[indices]
                        scaled = scaler.transform(batch.reshape(-1, batch.shape[2])).reshape(batch.shape).astype(np.float32)
                        expected = np.clip(model(scaled, training=False).numpy()*100, -104, 96)
                        np.testing.assert_allclose(expected, raw, atol=1e-4)
                elif model_name == "monte_carlo":
                    from src.swing_return_experiment import mc_expected
                    for indices, raw in ((cal[:3], rawcal[:3]), (val[:3], rawval[:3])):
                        expected = np.clip(mc_expected(hist[indices], meta.iloc[indices].reset_index(drop=True))[horizon], -104, 96)
                        np.testing.assert_allclose(expected, raw, atol=1e-9)
                else:
                    raise ValueError("Missing saved fitted model")
                recomputed, result = evaluate(model_name, horizon, prediction, hist[val],
                    meta.iloc[val].reset_index(drop=True), future[val], known[val], 1., 4.)
                saved_trades = pd.read_parquet(folder / "trades" / f"{model_name}_{horizon}.parquet")
                pd.testing.assert_frame_equal(saved_trades, recomputed)
                checked += 1
    result = {"status": "passed", **source_checks, "prospective_histories_reconstructed": len(meta),
        "paired_predictions_verified": checked, "counts": {"fit": len(fit_used) if exclude_near else len(fit), "calibration": len(cal), "validation": len(val)},
        "scope": "Independent source timing reconstruction, shared chronological identities, independent targets/calibration and tabular predictions. GRU/MC internals separately tested; execution is replayed through shared policy."}
    save_json((output / "exclude_near" if exclude_near else output) / "audit.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--exclude-near", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.base, args.exclude_near), indent=2))
