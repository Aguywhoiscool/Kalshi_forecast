"""Audit a completed legacy benchmark without fitting or reading holdout outcomes."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_benchmark import (IDENTITY, classification_metrics, execution_rows,
    FEATURE_NAMES, quote_frame, select_samples, simulate_validation, summarize, validate_splits)
from src.movement_labels import LABEL_POLICY, meaningful_movement
from src.v3_research import ShardCache, digest, protected_period, save_json, targets, valid_rows


def decimal_changes(quotes):
    return np.array([float((Decimal(str(eb)) + Decimal(str(ea))
                          - Decimal(str(ib)) - Decimal(str(ia))) * 50)
                     for ib, ia, eb, ea in quotes], dtype=float)


def audit(benchmark, output):
    if output.exists():
        raise FileExistsError("Preserve existing audits: use a new directory")
    protocol = json.loads((benchmark / "protocol.json").read_text())
    current_policy = protocol.get("label_policy") == LABEL_POLICY
    source = Path(protocol["source_run"])
    for name, expected in protocol["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError("Source hash mismatch: " + name)
    for name, expected in protocol["sample_hashes"].items():
        if digest(benchmark / name) != expected:
            raise ValueError("Sample hash mismatch: " + name)
    protected = json.loads((source / "protected_v2_period.json").read_text())
    if protected != protected_period(Path.cwd()):
        raise ValueError("Reserved period protection differs from original v2 metadata")
    meta = pd.read_parquet(source / "windows.parquet")
    validate_splits(meta, protected)
    training, validation = select_samples(meta, protocol["source_config"])
    for name, selected in (("training", training), ("validation", validation)):
        pd.testing.assert_frame_equal(selected[IDENTITY],
            pd.read_parquet(benchmark / f"{name}_indices.parquet"), check_dtype=False)
    saved = pd.read_parquet(benchmark / "validation_predictions.parquet")
    pd.testing.assert_frame_equal(validation[IDENTITY], saved[IDENTITY], check_dtype=False)
    for _, group in validation.groupby(["session_id", "ticker"]):
        group = group.sort_values("input_start")
        if (group.input_start.to_numpy()[1:] <= group.target_end.to_numpy()[:-1]).any():
            raise ValueError("Overlapping validation windows")
    cache = ShardCache(source)
    train_change, val_change = targets(training, cache), targets(validation, cache)
    np.testing.assert_array_equal(val_change, saved.change_cents)
    quotes = quote_frame(validation, cache)
    exact_change = decimal_changes(quotes)
    corrected_val, corrected_train = meaningful_movement(val_change), meaningful_movement(train_change)
    np.testing.assert_array_equal(corrected_val, np.abs(exact_change) >= .5)
    # Independently check YES and complementary NO P&L using Decimal arithmetic.
    yes_pnl = np.array([float((Decimal(str(eb)) - Decimal(str(ia))) * 100)
                        for ib, ia, eb, ea in quotes])
    no_pnl = np.array([float(((1 - Decimal(str(ea))) - (1 - Decimal(str(ib)))) * 100)
                       for ib, ia, eb, ea in quotes])
    manifest = json.loads((source / "input_manifest.json").read_text())
    checked_rows = 0
    feature_matrix = np.load(benchmark / "validation_features.npy", mmap_mode="r")
    # Reconstruct every validation window directly from hashed source CSVs,
    # independently of the shard's row offsets. No holdout labels are touched.
    for shard, positions in validation.groupby("shard", sort=False).indices.items():
        item = manifest["files"][int(shard)]
        path = source / "snapshot" / item["name"]
        if digest(path) != item["sha256"]:
            raise ValueError("Snapshot hash mismatch: " + path.name)
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        if protocol["feature_names"] != FEATURE_NAMES:
            raise ValueError("Feature schema differs from benchmark")
        frame = frame.loc[valid_rows(frame)]
        frame = frame.loc[frame.spread <= .10 + 1e-6].drop_duplicates(["session_id", "ticker", "timestamp"])
        for (session, ticker), selected in validation.iloc[positions].groupby(["session_id", "ticker"]):
            group = frame.loc[(frame.session_id.astype(str) == str(session)) & (frame.ticker == ticker)].sort_values("timestamp")
            times = group.timestamp.array.as_unit("ns").asi8
            ends = np.searchsorted(times, selected.input_end.to_numpy())
            np.testing.assert_array_equal(times[ends], selected.input_end)
            np.testing.assert_array_equal(times[ends - 59], selected.input_start)
            np.testing.assert_array_equal(times[ends + 10], selected.target_end)
            raw = group[["bid", "ask", "spread", "midpoint", "bid_depth", "ask_depth", "imbalance"]].to_numpy()
            history = raw[ends[:, None] + np.arange(-59, 1)]
            np.testing.assert_allclose(summarize(history), feature_matrix[selected.index], rtol=0, atol=0)
            np.testing.assert_array_equal(raw[ends, :2], quotes[selected.index, :2])
            np.testing.assert_array_equal(raw[ends + 10, :2], quotes[selected.index, 2:])
            checked_rows += len(selected)
    old_metrics = json.loads((benchmark / "metrics.json").read_text())
    train_matrix = np.load(benchmark / "train_features.npy", mmap_mode="r")
    for task in ("movement", "direction"):
        train_mask = corrected_train if current_policy else np.abs(train_change) >= .5
        selected_train = train_matrix[train_mask] if task == "direction" else train_matrix
        expected_mean = selected_train.mean(axis=0, dtype=np.float64)
        for name in ("logistic", "sgd"):
            pipeline = joblib.load(benchmark / "models" / f"{name}_{task}.joblib")
            scaler = pipeline.named_steps["standardscaler"]
            if scaler.n_samples_seen_ != len(selected_train):
                raise ValueError("Scaler includes a different population: " + name + task)
            np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-12, atol=1e-9)
    diagnostics, executions = {}, []
    for model in old_metrics:
        diagnostics[model] = {}
        for task in ("movement", "direction"):
            recorded_moving = corrected_val if current_policy else np.abs(val_change) >= .5
            mask_old = recorded_moving if task == "direction" else np.ones(len(validation), bool)
            y_old = val_change[mask_old] > 0 if task == "direction" else recorded_moving
            prob = saved[f"{model}_{task}"].to_numpy()
            if model not in ("gru", "monte_carlo"):
                fitted = joblib.load(benchmark / "models" / f"{model}_{task}.joblib")
                recomputed = fitted.predict_proba(feature_matrix)[:, list(fitted.classes_).index(1)]
                np.testing.assert_allclose(recomputed, prob, rtol=0, atol=1e-12)
            reproduced = classification_metrics(y_old, prob[mask_old])
            for metric in ("average_precision", "roc_auc", "accuracy", "balanced_accuracy", "brier_score", "log_loss"):
                np.testing.assert_allclose(reproduced[metric], old_metrics[model][task][metric], rtol=0, atol=1e-12)
            mask_new = corrected_val if task == "direction" else np.ones(len(validation), bool)
            y_new = val_change[mask_new] > 0 if task == "direction" else corrected_val
            diagnostics[model][task] = classification_metrics(y_new, prob[mask_new])
        rows = execution_rows(model, saved[f"{model}_movement"], saved[f"{model}_direction"], quotes)
        up = saved[f"{model}_direction"].to_numpy() >= .5
        np.testing.assert_allclose(rows[-1]["mean_gross_cents"], np.where(up, yes_pnl, no_pnl).mean(), atol=1e-12)
        executions.extend(rows)
    # Corrected Monte Carlo probabilities are a diagnostic for the corrected
    # simulation policy; supervised models above remain frozen legacy models.
    simulation = simulate_validation(validation, cache)
    for task in ("movement", "direction"):
        mask = corrected_val if task == "direction" else np.ones(len(validation), bool)
        y = val_change[mask] > 0 if task == "direction" else corrected_val
        diagnostics["monte_carlo"][task] = classification_metrics(y, simulation[task].to_numpy()[mask])
    executions = [row for row in executions if row["model"] != "monte_carlo"] + execution_rows(
        "monte_carlo", simulation.movement, simulation.direction, quotes)
    summary = {"benchmark": str(benchmark.resolve()), "label_policy": LABEL_POLICY,
        "training_examples": len(training), "validation_examples": len(validation),
        "legacy_moving_training": int((np.abs(train_change) >= .5).sum()),
        "corrected_moving_training": int(corrected_train.sum()),
        "legacy_moving_validation": int((np.abs(val_change) >= .5).sum()),
        "corrected_moving_validation": int(corrected_val.sum()),
        "source_reconstructed_validation_windows": checked_rows,
        "checks": ["Source/sample SHA256 provenance matches", "Original v2 protection reproduced from timestamps/indices only",
            "Chronological 70-second embargo and disjoint sessions", "No shared input/target timestamps within validation market/session",
            "All validation histories/quotes reconstructed from hashed source CSVs",
            "Saved tabular models reproduce predictions; scaler counts/means match training only",
            "All saved classifier metrics reproduced", "Corrected labels match independent Decimal quotes",
            "Complementary YES/NO gross P&L matches independent Decimal arithmetic"],
        "findings": (["Current label policy verified; historical threshold errors no longer determine labels",
            "Corrected cutoff ties use uniform expected selection; fractions remain retrospective",
            "GRU and tabular models still use different history representations and validation-selection budgets"] if current_policy else
            ["Exact 0.5-cent moves were excluded due to floating-point threshold error",
            "Stable-sort cutoff ties selected arbitrary early timestamps, notably for constant-prior baseline",
            "Top fractions are retrospective full-validation ranks, not prospective deployable thresholds",
            "GRU and tabular models use different history representations and validation-selection budgets"]),
        "status": ("Corrected-label benchmark audit passed; results remain exploratory development validation" if current_policy else
                   "Audit completed; legacy models require retraining under corrected labels before claiming an updated fair comparison"),
        "diagnostic_scope": ("Corrected models re-scored; Monte Carlo reproduced under the same corrected policy" if current_policy else
                            "Frozen legacy-model re-scoring; Monte Carlo rerun with corrected threshold. Not a corrected training benchmark."),
        "holdout": "No reserved outcomes read; all diagnostics use existing development validation only"}
    output.mkdir(parents=True)
    save_json(output / "audit.json", summary)
    save_json(output / "diagnostic_metrics.json", diagnostics)
    pd.DataFrame(executions).to_csv(output / "diagnostic_execution.csv", index=False)
    report = ["# Model benchmark correctness audit", "", summary["status"], "",
        f"Moving training examples: {summary['legacy_moving_training']:,} -> {summary['corrected_moving_training']:,}.",
        f"Moving validation examples: {summary['legacy_moving_validation']:,} -> {summary['corrected_moving_validation']:,}.",
        "", "## Verified", *["- " + item for item in summary["checks"]], "",
        "## Findings and fixes", *["- " + item for item in summary["findings"]], "",
        "Shared threshold handling and regression tests now fix numerical boundary behavior. Old GRU runs are rejected",
        "by the benchmark and trainer rather than silently reused with changed labels. Score-cutoff ties now use",
        "uniform expected selection. Original checkpoints, predictions, metrics and split definitions were preserved.",
        "", summary["diagnostic_scope"], summary["holdout"], "",
        "## Next required experiment", ("Corrected retraining is complete. Replicate across development periods and seeds before generalization claims." if current_policy else
        "Retrain all models, including GRU, in a NEW directory using the same frozen dataset/splits and corrected labels; do not move holdout boundaries."),
        "Then rerun the benchmark. Multiple development periods/seeds remain necessary before generalization claims.",
        "No live collection changes, trades, Git commits or pushes were performed."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/model_benchmark_v3_monte_carlo"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_correctness_audit"))
    args = parser.parse_args()
    audit(args.benchmark, args.output)
