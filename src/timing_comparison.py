"""Paired seven/eight-input experiment on verified prospective histories only."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
from src.model_benchmark import summarize
from src.swing_return_experiment import regressors, train_gru, return_bounds, evaluate, mc_expected
from src.swing_strategy_experiment import inner_split
from src.v3_research import digest, save_json


def timing_history(group, ends):
    fields = {"scheduled_close_time", "close_time_observed_at", "seconds_to_scheduled_close"}
    if not fields <= set(group.columns):
        return np.full((len(ends), 60), np.nan)
    stamp = pd.to_datetime(group.timestamp, utc=True, format="ISO8601")
    close = pd.to_datetime(group.scheduled_close_time, utc=True, errors="coerce", format="ISO8601")
    observed = pd.to_datetime(group.close_time_observed_at, utc=True, errors="coerce", format="ISO8601")
    seconds = pd.to_numeric(group.seconds_to_scheduled_close, errors="coerce")
    valid = close.notna() & observed.notna() & (observed <= stamp) & np.isfinite(seconds)
    valid &= ((close-stamp).dt.total_seconds()-seconds).abs() <= .001
    values = np.where(valid, seconds, np.nan)
    return values[ends[:, None]+np.arange(-59, 1)]


def materialize(run):
    protocol = json.loads((run / "protocol.json").read_text())
    source = Path(protocol["source"])
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    timing = np.full((len(meta), 60), np.nan)
    for session, selected in meta.groupby("session_id", sort=False):
        frame = pd.read_csv(source / "snapshot" / f"features_{session}.csv")
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
        for ticker, rows in selected.groupby("ticker", sort=False):
            group = frame.loc[frame.ticker == ticker].sort_values("timestamp").reset_index(drop=True)
            times = group.timestamp.array.as_unit("ns").asi8
            ends = np.searchsorted(times, rows.input_end.to_numpy())
            if not np.array_equal(times[ends], rows.input_end.to_numpy()):
                raise ValueError("Timing history identity mismatch")
            timing[rows.index] = timing_history(group, ends)
    return meta, timing


def summarize_schema(histories):
    base = summarize(histories[:, :, :7])
    if histories.shape[2] == 7:
        return base
    values = histories[:, :, 7]
    return np.column_stack([base, values[:, -1], values.mean(axis=1), values.std(axis=1),
        values.min(axis=1), values.max(axis=1), values[:, -1]-values[:, 0]]).astype(np.float32)


def non_near_fit(fit, remaining):
    """Apply only the predeclared >10 through <=60 minute fit exclusion."""
    fit = np.asarray(fit, dtype=int)
    values = np.asarray(remaining, dtype=float)[fit]
    if not np.isfinite(values).all():
        raise ValueError("Exclusion requires verified timing")
    return fit[~((values > 600) & (values <= 3600))]


def run_exclusion(base, epochs=8):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    base = base.resolve()
    parent = base / "timing_comparison"
    if json.loads((parent / "audit.json").read_text())["status"] != "passed":
        raise ValueError("Existing paired comparison must pass its audit first")
    folder = parent / "exclude_near"
    if folder.exists():
        raise ValueError("Supplement already exists; preserve artifacts")
    meta = pd.read_parquet(parent / "identities.parquet")
    base_fit, cal, val = inner_split(meta)
    original_ids = meta.original_row.to_numpy(dtype=int)
    print(json.dumps({"stage": "materializing_verified_timing", "variant": "exclude_near",
                      "cohort_rows": len(meta)}), flush=True)
    _, timing = materialize(base)
    fit = non_near_fit(base_fit, timing[original_ids, -1])
    if len(fit) == 0:
        raise ValueError("No fit examples remain after exclusion")
    (folder / "models").mkdir(parents=True)
    (folder / "traces").mkdir()
    (folder / "trades").mkdir()
    for label, rows in (("fit", fit), ("calibration", cal), ("validation", val)):
        np.save(folder / f"{label}_rows.npy", rows)
    save_json(folder / "protocol.json", {"parent": str(parent),
        "identities_sha256": digest(parent / "identities.parquet"),
        "policy": "Seven-input supplemental fit-only exclusion of >600 and <=3600 seconds to close; verified prospective timing",
        "counts": {"original_fit": len(base_fit), "fit": len(fit), "excluded_fit": len(base_fit)-len(fit),
                   "calibration": len(cal), "validation": len(val)},
        "comparison": "Same calibration/validation identities and settings as paired seven-input baseline; fewer fit examples is a confound"})
    histories = np.load(base / "history.npy", mmap_mode="r")[original_ids]
    future = np.load(base / "future_quotes.npy", mmap_mode="r")[original_ids]
    available = np.load(base / "future_valid.npy", mmap_mode="r")[original_ids]
    features = summarize(histories)
    metrics, results = {}, []
    for horizon in (30, 60):
        target, _, _ = return_bounds(future, available, horizon)
        for name, model in [*regressors().items(), ("gru", None), ("monte_carlo", None), ("no_trade", None)]:
            print(json.dumps({"variant": "exclude_near", "horizon": horizon, "model": name,
                              "fit": len(fit), "calibration": len(cal), "validation": len(val)}), flush=True)
            if name == "no_trade":
                prediction = np.full((len(val), 2), -1000.)
            else:
                if name == "gru":
                    raw_cal, raw_val = train_gru(histories, target, fit, cal, val, folder, horizon,
                        lambda stage, **kw: print(stage, kw, flush=True), epochs)
                elif name == "monte_carlo":
                    # No fit occurs: reuse the identical deterministic baseline,
                    # then independently replay it during the supplemental audit.
                    with np.load(parent / "seven" / f"monte_carlo_{horizon}_predictions.npz") as saved:
                        raw_cal, raw_val = saved["raw_calibration"], saved["raw_validation"]
                else:
                    with threadpool_limits(2):
                        model.fit(features[fit], target[fit])
                        raw_cal, raw_val = model.predict(features[cal]), model.predict(features[val])
                    joblib.dump(model, folder / "models" / f"{name}_{horizon}.joblib")
                    if name == "sgd":
                        save_json(folder / "traces" / f"{name}_{horizon}.json", model.steps[-1][1].trace_)
                raw_cal, raw_val = np.clip(raw_cal, -104, 96), np.clip(raw_val, -104, 96)
                correction = (target[cal]-raw_cal).mean(axis=0)
                prediction = np.clip(raw_val+correction, -104, 96)
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite supplemental predictions")
                np.savez_compressed(folder / f"{name}_{horizon}_predictions.npz", raw_calibration=raw_cal,
                    raw_validation=raw_val, correction=correction, validation=prediction)
                metrics[f"exclude_near_{name}_{horizon}"] = {"mae_conservative_target_cents":
                    float(mean_absolute_error(target[val], prediction)), "calibration_correction_cents": correction.tolist()}
            trades, result = evaluate(name, horizon, prediction, histories[val], meta.iloc[val].reset_index(drop=True),
                future[val], available[val], 1., 4.)
            trades.to_parquet(folder / "trades" / f"{name}_{horizon}.parquet", index=False)
            result["variant"] = "exclude_near"
            results.append(result)
            save_json(folder / "metrics.json", metrics)
            pd.DataFrame(results).to_csv(folder / "results.csv", index=False)
    from src.audit_timing_comparison import audit
    audit(base, exclude_near=True)
    save_json(folder / "status.json", {"status": "complete", "variant": "exclude_near"})
    counts = json.loads((folder / "protocol.json").read_text())["counts"]
    lines = ["# Seven-input fit exclusion comparison", "", f"Fit: {counts['fit']} (excluded {counts['excluded_fit']} of {counts['original_fit']}); calibration {len(cal)}; validation {len(val)}.",
        "Same calibration/validation identities as the paired seven/eight comparison. Only fit rows >10 and <=60 minutes before their known scheduled close were removed.",
        "", "| Model | Horizon | MAE cents | Attempts | Net lower cents | Net upper cents |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        item = metrics.get(f"exclude_near_{row['model']}_{row['horizon']}")
        mae = f"{item['mae_conservative_target_cents']:.3f}" if item else "—"
        lines.append(f"| {row['model']} | {row['horizon']}s | {mae} | {row['attempts']} | {row['net_lower_total_cents']:+.3f} | {row['net_upper_total_cents']:+.3f} |")
    lines += ["", "Exploratory comparison: fewer fit samples confound distribution effects. Fixed 1c entry margin and 4c hypothetical cost buffer; uncertain outcomes remain bounded. No verified profitable strategy or held-out outcome claim. Monte Carlo does not fit and is unchanged."]
    (folder / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def write_report(base):
    output = base / "timing_comparison"
    protocol = json.loads((output / "protocol.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    rows = pd.read_csv(output / "results.csv")
    audited = json.loads((output / "audit.json").read_text())["status"] == "passed"
    lines = ["# Paired prospective timing comparison", "", f"Audit: {'passed' if audited else 'not passed'}.",
        f"Shared cohort: {protocol['counts']['fit']:,} fit, {protocol['counts']['calibration']:,} calibration, {protocol['counts']['validation']:,} validation observations.",
        "", "Seven book inputs versus the same inputs plus verified seconds-to-close. Both variants use identical chronological session identities and 70-second embargoes.",
        "", "| Variant | Model | Horizon | MAE lower-target cents | Attempts | Closed | Net lower cents | Net upper cents |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows.itertuples():
        item = metrics.get(f"{row.variant}_{row.model}_{row.horizon}")
        mae = f"{item['mae_conservative_target_cents']:.3f}" if item else "—"
        lines.append(f"| {row.variant} | {row.model} | {row.horizon}s | {mae} | {row.attempts} | {row.closed} | {row.net_lower_total_cents:+.3f} | {row.net_upper_total_cents:+.3f} |")
    lines += ["", "## Interpretation boundaries", "",
        "- Same paired validation data; full-history baseline has a separate cohort and is not a controlled feature comparison.",
        "- Targets include conservative missing-outcome bounds. Lower MAE does not establish profitable execution.",
        "- Fixed 1c expected-return gate, 4c additional cost scenario and approximate 1-second entry; fees/fills remain hypothetical.",
        "- Monte Carlo uses the same book-only simulation in both variants. It is not trained or made time-aware by the eight-input comparison.",
        "- Missing future quotes were retained. Entire input timing histories were reconstructed from source knowledge timestamps.",
        "- Reserved v2/v3 outcomes were not evaluated. This is exploratory development validation, not final held-out proof."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run_comparison(base, epochs=8):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    base = base.resolve()
    output = base / "timing_comparison"
    if output.exists():
        raise ValueError("Comparison already exists; preserve previous artifacts")
    output.mkdir()
    meta, timing = materialize(base)
    valid = np.isfinite(timing).all(axis=1) & meta.eligible_30.to_numpy() & meta.eligible_60.to_numpy()
    original_ids = np.flatnonzero(valid)
    paired = meta.loc[valid].reset_index(drop=True)
    sessions = paired.groupby("session_id").input_start.min().sort_values()
    cut = int(len(sessions)*.8)
    if cut < 2 or cut >= len(sessions):
        raise ValueError("Insufficient prospective sessions for paired comparison")
    train_sessions = set(sessions.index[:cut])
    paired["split"] = np.where(paired.session_id.isin(train_sessions), "train", "val")
    boundary = paired.loc[paired.split == "train", "target_end"].max()+70*10**9
    keep = paired.split.eq("train") | paired.input_start.ge(boundary)
    original_ids, paired = original_ids[keep], paired.loc[keep].reset_index(drop=True)
    fit, cal, val = inner_split(paired)
    paired["original_row"] = original_ids
    paired.to_parquet(output / "identities.parquet", index=False)
    save_json(output / "protocol.json", {"parent": str(base), "identities_sha256": digest(output / "identities.parquet"),
        "policy": "Paired prospective-only seven/eight inputs; own chronological session split, same identities for both variants",
        "timing_feature": "Verified seconds_to_scheduled_close at every input observation; no retrospective schedules",
        "eligibility": "Common cohort administratively eligible for both horizons; no observed future-coverage filter",
        "counts": {"fit": len(fit), "calibration": len(cal), "validation": len(val)},
        "limitations": "Separate cohort from full-history baseline; future reserved v2/v3 outcomes remain excluded"})
    histories = np.load(base / "history.npy", mmap_mode="r")[original_ids]
    future = np.load(base / "future_quotes.npy", mmap_mode="r")[original_ids]
    available = np.load(base / "future_valid.npy", mmap_mode="r")[original_ids]
    all_metrics, all_results = {}, []
    for variant, dimensions in (("seven", 7), ("eight", 8)):
        folder = output / variant
        (folder / "models").mkdir(parents=True)
        (folder / "traces").mkdir()
        (folder / "trades").mkdir()
        history = histories if dimensions == 7 else np.concatenate([histories, timing[original_ids, :, None]], axis=2)
        features = summarize_schema(history)
        save_json(folder / "feature_schema.json", {"dimensions": dimensions,
            "sequence": ["bid", "ask", "spread", "midpoint", "bid_depth", "ask_depth", "imbalance"]+
                (["seconds_to_scheduled_close"] if dimensions == 8 else []),
            "summary_dimensions": features.shape[1], "fit_rows": len(fit)})
        for horizon in (30, 60):
            target, upper, status = return_bounds(future, available, horizon)
            for name, model in [*regressors().items(), ("gru", None), ("monte_carlo", None), ("no_trade", None)]:
                print(json.dumps({"variant": variant, "horizon": horizon, "model": name}), flush=True)
                if name == "no_trade":
                    prediction = np.full((len(val), 2), -1000.)
                else:
                    if name == "gru":
                        raw_cal, raw_val = train_gru(history, target, fit, cal, val, folder, horizon,
                            lambda stage, **kw: print(stage, kw, flush=True), epochs)
                    elif name == "monte_carlo":
                        combined = np.r_[cal, val]
                        mc = mc_expected(histories[combined], paired.iloc[combined].reset_index(drop=True))[horizon]
                        raw_cal, raw_val = mc[:len(cal)], mc[len(cal):]
                    else:
                        with threadpool_limits(2):
                            model.fit(features[fit], target[fit])
                            raw_cal, raw_val = model.predict(features[cal]), model.predict(features[val])
                        joblib.dump(model, folder / "models" / f"{name}_{horizon}.joblib")
                        if name == "sgd":
                            save_json(folder / "traces" / f"{name}_{horizon}.json", model.steps[-1][1].trace_)
                    raw_cal, raw_val = np.clip(raw_cal, -104, 96), np.clip(raw_val, -104, 96)
                    correction = (target[cal]-raw_cal).mean(axis=0)
                    prediction = np.clip(raw_val+correction, -104, 96)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Non-finite paired predictions")
                    np.savez_compressed(folder / f"{name}_{horizon}_predictions.npz", raw_calibration=raw_cal,
                        raw_validation=raw_val, correction=correction, validation=prediction)
                    all_metrics[f"{variant}_{name}_{horizon}"] = {"mae_conservative_target_cents":
                        float(mean_absolute_error(target[val], prediction)), "calibration_correction_cents": correction.tolist()}
                trades, summary = evaluate(name, horizon, prediction, histories[val], paired.iloc[val].reset_index(drop=True),
                    future[val], available[val], 1., 4.)
                trades.to_parquet(folder / "trades" / f"{name}_{horizon}.parquet", index=False)
                summary["variant"] = variant
                all_results.append(summary)
                save_json(output / "metrics.json", all_metrics)
                pd.DataFrame(all_results).to_csv(output / "results.csv", index=False)
    from src.audit_timing_comparison import audit
    audit(base)
    write_report(base)
    save_json(output / "status.json", {"status": "complete", "variants": ["seven", "eight"],
        "note": "Monte Carlo uses book-only paths in both variants; no time-aware simulation claim"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--exclude-near", action="store_true", help="Add fit-only exclusion supplement without changing paired artifacts")
    args = parser.parse_args()
    (run_exclusion if args.exclude_near else run_comparison)(args.base, args.epochs)
