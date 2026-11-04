"""Offline, resumable v3 research. Never imports a client or submits orders.

Run: python -m src.v3_research --run-dir artifacts/v3_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

from src.movement_labels import LABEL_POLICY, meaningful_movement

FEATURES = ["bid", "ask", "spread", "midpoint", "bid_depth", "ask_depth", "imbalance"]
WINDOW, HORIZON, EMBARGO = 60, 10, 70
NS = 1_000_000_000


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def status(run, stage, **extra):
    entry = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    save_json(run / "status.json", entry)
    print(json.dumps(entry, default=str), flush=True)


def protected_period(root):
    """Read timestamp metadata only; never open v2 outcomes or model predictions."""
    base = root / "data/processed/datasets/v2_phase7j"
    indices = np.load(base / "splits/test_indices.npy", allow_pickle=False)
    meta = pd.read_csv(base / "gru_dataset_combined.metadata.csv",
                       usecols=["input_start", "target_end", "session_id"])
    reserved = meta.iloc[indices]
    return {"start": pd.to_datetime(reserved.input_start, utc=True).min().value,
            "end": pd.to_datetime(reserved.target_end, utc=True).max().value,
            "sessions": sorted(set(reserved.session_id.astype(str))),
            "indices_sha256": digest(base / "splits/test_indices.npy"),
            "policy": "Exclude entire reserved time range plus every reserved session globally."}


def overlaps(starts, ends, protected):
    return (starts <= protected["end"]) & (ends >= protected["start"])


def candidate_ends(times):
    """Require all 69 intervals, elapsed history and elapsed forecast to agree."""
    if len(times) < WINDOW + HORIZON:
        return np.empty(0, dtype=np.int64)
    ends = np.arange(WINDOW - 1, len(times) - HORIZON)
    starts, futures = ends - WINDOW + 1, ends + HORIZON
    gaps = np.diff(times) / NS
    bad = np.r_[0, np.cumsum((gaps < 0.5) | (gaps > 1.5))]
    good = ((bad[futures] == bad[starts])
            & (np.abs((times[futures] - times[ends]) / NS - HORIZON) <= 0.5)
            & (np.abs((times[ends] - times[starts]) / NS - (WINDOW - 1)) <= 1.5))
    return ends[good]


def valid_rows(frame):
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    bid, ask, spread, mid, bd, ad, imb = values.T
    depth = bd + ad
    expected = np.divide(bd - ad, depth, out=np.zeros_like(depth), where=depth > 0)
    return (np.isfinite(values).all(axis=1) & frame.timestamp.notna().to_numpy()
            & frame.session_id.notna().to_numpy() & frame.ticker.notna().to_numpy()
            & (bid >= 0) & (bid < ask) & (ask <= 1)
            & (bd >= 0) & (ad >= 0) & (np.abs(imb) <= 1 + 1e-6)
            & (np.abs(imb - expected) <= 1e-5)
            & (spread > 0) & (np.abs(spread - (ask - bid)) <= 1e-6)
            & (np.abs(mid - (bid + ask) / 2) <= 1e-6))


def snapshot(root, run):
    manifest_path = run / "input_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            if digest(run / "snapshot" / item["name"]) != item["sha256"]:
                raise ValueError("Snapshot hash mismatch: " + item["name"])
        return manifest
    source = root / "data/processed"
    paths = sorted(source.glob("features_*.csv"))
    # The newest session might still be active even when its mtime is old.
    newest = paths[-1] if paths else None
    cutoff = time.time() - 30 * 60
    selected, excluded = [], []
    destination = run / "snapshot"
    destination.mkdir(exist_ok=True)
    for i, path in enumerate(paths):
        before = path.stat()
        if path == newest or before.st_mtime >= cutoff or "sync-conflict" in path.name:
            excluded.append(path.name)
            continue
        target = destination / path.name
        shutil.copyfile(path, target)
        fingerprint = digest(target)
        current_hash = digest(path)
        after = path.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or fingerprint != current_hash):
            target.unlink()
            excluded.append(path.name)
            continue
        selected.append({"name": path.name, "size": before.st_size,
                         "mtime_ns": before.st_mtime_ns, "sha256": fingerprint})
        if i % 200 == 0:
            status(run, "snapshot", copied=len(selected), inspected=i + 1)
    manifest = {"files": selected, "excluded": excluded,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "syncthing_status": "Not inspected; source-completion cannot be guaranteed remotely.",
                "policy": "Exclude newest and <30-minute-mtime files; immutable copies verified against source hashes and size/mtime."}
    save_json(manifest_path, manifest)
    return manifest


def build(root, run, manifest, protected):
    complete = run / "dataset_complete.json"
    if complete.exists():
        return json.loads(complete.read_text())
    shards = run / "shards"
    shards.mkdir(exist_ok=True)
    pieces, audits = [], []
    reserved_sessions = set(protected["sessions"])
    for number, item in enumerate(manifest["files"]):
        path = run / "snapshot" / item["name"]
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
        for col in FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        validity = valid_rows(frame)
        audit = {"file": path.name, "rows": len(frame), "invalid": int((~validity).sum())}
        frame = frame.loc[validity].copy()
        audit["wide"] = int((frame.spread > .10 + 1e-6).sum())
        frame = frame.loc[frame.spread <= .10 + 1e-6]
        pre = len(frame)
        frame = frame.drop_duplicates(["session_id", "ticker", "timestamp"])
        audit["duplicates"] = pre - len(frame)
        arrays, metadata, offset = [], [], 0
        for (session, ticker), group in frame.groupby(["session_id", "ticker"], sort=False):
            if str(session) in reserved_sessions:
                continue
            group = group.sort_values("timestamp").reset_index(drop=True)
            times = group.timestamp.array.as_unit("ns").asi8
            ends = candidate_ends(times)
            if not len(ends):
                continue
            ends = ends[~overlaps(times[ends - 59], times[ends + 10], protected)]
            if not len(ends):
                continue
            arrays.append(group[FEATURES].to_numpy(dtype=np.float64))
            metadata.append(pd.DataFrame({"session_id": str(session), "ticker": str(ticker),
                "shard": number, "end": ends + offset,
                "input_start": times[ends - 59], "input_end": times[ends],
                "target_end": times[ends + 10]}))
            offset += len(group)
        audits.append(audit)
        if arrays:
            np.save(shards / f"{number:05d}.npy", np.concatenate(arrays), allow_pickle=False)
            part = pd.concat(metadata, ignore_index=True)
            part.to_parquet(shards / f"{number:05d}.parquet", index=False)
            pieces.append(part)
        if number % 100 == 0:
            status(run, "build", files=number + 1, windows=sum(len(p) for p in pieces))
    pd.DataFrame(audits).to_csv(run / "input_audit.csv", index=False)
    if not pieces:
        raise ValueError("No eligible development windows outside reserved period.")
    meta = pd.concat(pieces, ignore_index=True)
    sessions = meta.groupby("session_id").agg(start=("input_start", "min"),
                    end=("target_end", "max"), count=("end", "size")).sort_values("start")
    cumulative = sessions["count"].cumsum().to_numpy()
    first = int(np.searchsorted(cumulative, len(meta) * .70)) + 1
    second = int(np.searchsorted(cumulative, len(meta) * .85)) + 1
    if not (0 < first < second < len(sessions)):
        raise ValueError("Insufficient sessions for splits")
    names = sessions.index.to_numpy()
    assignments = {s: "train" if i < first else "val" if i < second else "holdout"
                   for i, s in enumerate(names)}
    meta["split"] = meta.session_id.map(assignments)
    train_end = meta.loc[meta.split == "train", "target_end"].max()
    val_end = meta.loc[meta.split == "val", "target_end"].max()
    keep = ((meta.split == "train")
            | ((meta.split == "val") & (meta.input_start >= train_end + EMBARGO * NS))
            | ((meta.split == "holdout") & (meta.input_start >= val_end + EMBARGO * NS)))
    meta = meta.loc[keep].reset_index(drop=True)
    for split in ("train", "val", "holdout"):
        if not (meta.split == split).any():
            raise ValueError("Empty split: " + split)
    # Persist holdout timing/index metadata only; never calculate its targets here.
    meta.to_parquet(run / "windows.parquet", index=False)
    summary = {"rows_audited": sum(a["rows"] for a in audits),
        "invalid_rows": sum(a["invalid"] for a in audits),
        "wide_rows": sum(a["wide"] for a in audits),
        "duplicate_rows": sum(a["duplicates"] for a in audits),
        "splits": {name: {"windows": len(group), "sessions": group.session_id.nunique(),
            "first_input": pd.Timestamp(group.input_start.min(), tz="UTC").isoformat(),
            "last_target": pd.Timestamp(group.target_end.max(), tz="UTC").isoformat()}
            for name, group in meta.groupby("split")},
        "embargo_seconds": EMBARGO,
        "holdout_policy": "Frozen; outcomes not calculated or evaluated by this run."}
    save_json(complete, summary)
    return summary


class ShardCache:
    def __init__(self, run):
        self.run, self.cache = run, OrderedDict()

    def get(self, number):
        number = int(number)
        if number not in self.cache:
            self.cache[number] = np.load(self.run / "shards" / f"{number:05d}.npy", mmap_mode="r")
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(number)
        return self.cache[number]


def targets(meta, cache):
    # Compute only train/validation outcomes. Float64 before thresholding.
    if not meta.split.isin(["train", "val"]).all():
        raise ValueError("Refusing to calculate holdout targets")
    changes = np.empty(len(meta), dtype=np.float64)
    for shard, positions in meta.groupby("shard", sort=False).indices.items():
        rows = meta.iloc[positions].end.to_numpy(dtype=np.int64)
        data = cache.get(shard)
        changes[positions] = (data[rows + HORIZON, 3] - data[rows, 3]) * 100
    return changes


def spaced(meta):
    selected = []
    for _, group in meta.groupby(["session_id", "ticker"], sort=False):
        next_time = -1
        for row in group.sort_values("input_start").itertuples():
            if row.input_start >= next_time:
                selected.append(row.Index)
                next_time = row.target_end + 1
    return meta.loc[selected].sort_values(["input_end", "ticker"]).reset_index(drop=True)


def train_evaluate(root, run, args):
    saved_config = json.loads((run / "config.json").read_text())
    if saved_config.get("label_policy") != LABEL_POLICY:
        raise ValueError("Legacy label policy: use a NEW run directory; do not resume old checkpoints")
    import joblib
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    all_meta = pd.read_parquet(run / "windows.parquet")
    cache = ShardCache(run)
    training = all_meta.loc[all_meta.split == "train"].reset_index(drop=True)
    # Explicit bounded compute policy, deterministic chronological thinning.
    training = training.iloc[::args.train_stride].reset_index(drop=True)
    validation = spaced(all_meta.loc[all_meta.split == "val"].reset_index(drop=True))
    if len(training) > args.max_train_windows:
        training = training.iloc[np.linspace(0, len(training) - 1,
                                             args.max_train_windows, dtype=int)].reset_index(drop=True)
    fraction = getattr(args, "train_fraction", 1.0)
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1]")
    training = training.iloc[:max(1, int(len(training) * fraction))].reset_index(drop=True)
    train_change, val_change = targets(training, cache), targets(validation, cache)
    scaler_path = run / "feature_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        for start in range(0, len(training), 256):
            chunk = training.iloc[start:start + 256]
            batch = np.stack([cache.get(row.shard)[row.end-59:row.end+1]
                              for row in chunk.itertuples()])
            scaler.partial_fit(batch.reshape(-1, 7))
        joblib.dump(scaler, scaler_path)

    class Batches(tf.keras.utils.Sequence):
        def __init__(self, meta, labels=None, shuffle=False):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.meta, self.labels, self.shuffle = meta, labels, shuffle
            self.order = np.arange(len(meta))
            self.rng = np.random.default_rng(42)
            self.on_epoch_end()
        def __len__(self):
            return (len(self.order) + args.batch_size - 1) // args.batch_size
        def __getitem__(self, index):
            positions = self.order[index*args.batch_size:(index+1)*args.batch_size]
            chunk = self.meta.iloc[positions]
            raw = np.stack([cache.get(row.shard)[row.end-59:row.end+1]
                            for row in chunk.itertuples()])
            features = scaler.transform(raw.reshape(-1, 7)).reshape(raw.shape).astype(np.float32)
            return features if self.labels is None else (features, self.labels[positions])
        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)

    class Budget(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.started = time.monotonic()
        def on_train_batch_end(self, batch, logs=None):
            if time.monotonic() - self.started >= args.model_hours * 3600:
                self.model.stop_training = True

    metrics, predictions = {}, {}
    moving_train = meaningful_movement(train_change)
    moving_val = meaningful_movement(val_change)
    for task in ("direction", "movement"):
        output = run / task
        output.mkdir(exist_ok=True)
        model_path = output / "best.keras"
        predictions_path = output / "validation_probability.npy"
        if task == "direction":
            train_meta = training.loc[moving_train].reset_index(drop=True)
            train_y = (train_change[moving_train] > 0).astype(np.float32)
            val_meta = validation.loc[moving_val].reset_index(drop=True)
            val_y = (val_change[moving_val] > 0).astype(np.float32)
        else:
            train_meta, val_meta = training, validation
            train_y, val_y = moving_train.astype(np.float32), moving_val.astype(np.float32)
        if min(len(np.unique(train_y)), len(np.unique(val_y))) < 2:
            raise ValueError("Both classes needed for " + task)
        if not (output / "training_complete.json").exists():
            model = tf.keras.models.load_model(model_path) if model_path.exists() else tf.keras.Sequential([
                tf.keras.layers.Input((60, 7)),
                tf.keras.layers.GRU(64, return_sequences=True), tf.keras.layers.Dropout(.2),
                tf.keras.layers.GRU(32), tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")])
            model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="roc_auc"),
                         tf.keras.metrics.AUC(name="pr_auc", curve="PR")])
            monitor = "val_roc_auc" if task == "direction" else "val_pr_auc"
            status(run, "training_" + task, training_windows=len(train_meta), validation_windows=len(val_meta))
            history = model.fit(Batches(train_meta, train_y, True),
                validation_data=Batches(val_meta, val_y), epochs=args.epochs, verbose=2,
                callbacks=[tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor=monitor,
                    mode="max", save_best_only=True),
                    tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=3),
                    tf.keras.callbacks.CSVLogger(str(output / "epochs.csv"), append=True), Budget()])
            save_json(output / "training_complete.json", {"history": history.history})
            del model
        model = tf.keras.models.load_model(model_path)
        probability = model.predict(Batches(validation), verbose=0).reshape(-1)
        np.save(predictions_path, probability)
        predictions[task] = probability
        scores = probability[moving_val] if task == "direction" else probability
        metrics[task] = {"training_windows": len(train_meta), "validation_windows": len(val_meta),
            "roc_auc": roc_auc_score(val_y, scores),
            "average_precision": average_precision_score(val_y, scores),
            "accuracy": accuracy_score(val_y, scores >= .5)}
        save_json(output / "metrics.json", metrics[task])
        tf.keras.backend.clear_session()

    status(run, "offline_execution_evaluation")
    frame = validation.copy()
    frame["change_cents"] = val_change
    entry_bid, entry_ask, exit_bid, exit_ask, vol, momentum = ([] for _ in range(6))
    for row in validation.itertuples():
        data = cache.get(row.shard)
        entry_bid.append(data[row.end, 0]); entry_ask.append(data[row.end, 1])
        exit_bid.append(data[row.end+10, 0]); exit_ask.append(data[row.end+10, 1])
        vol.append(np.std(np.diff(data[row.end-59:row.end+1, 3])))
        momentum.append(data[row.end, 3] - data[row.end-10, 3])
    frame["volatility"] = vol
    frame["momentum10"] = momentum
    metrics["baseline"] = {"movement_prevalence": float(moving_val.mean()),
        "volatility_average_precision": average_precision_score(moving_val, vol),
        "momentum_direction_accuracy": accuracy_score(val_change[moving_val] > 0,
                                                      np.asarray(momentum)[moving_val] >= 0)}
    from src.model_benchmark import execution_rows

    rows = []
    for method, score, up in [
        ("v3_gru", predictions["movement"], predictions["direction"] >= .5),
        ("volatility_momentum", np.asarray(vol), np.asarray(momentum) >= 0)]:
        for result in execution_rows(method, score, np.asarray(up, dtype=float),
                                     np.column_stack([entry_bid, entry_ask, exit_bid, exit_ask])):
            result["method"] = result.pop("model")
            result.update(fees_and_slippage="excluded",
                          fill_assumption="hypothetical taker at sampled quotes")
            rows.append(result)
    pd.DataFrame(rows).to_csv(run / "gross_execution.csv", index=False)
    frame["p_movement"] = predictions["movement"]
    frame["p_up"] = predictions["direction"]
    frame.to_parquet(run / "validation_predictions.parquet", index=False)
    metrics["v2_comparison"] = {"status": "unavailable",
        "reason": "Canonical frozen v2_phase7j/k models and scalers absent locally; older v1 artifacts are not substitutes."}
    save_json(run / "metrics.json", metrics)
    return metrics


def write_report(run, args, summary, metrics=None, error=None):
    resume = f'python -m src.v3_research --run-dir "{run}" --epochs {args.epochs} --model-hours {args.model_hours}'
    text = ["# Offline v3 research run", "", "No orders, account calls, collector changes, Git commits or pushes are performed.",
        "", "## Dataset", "```json", json.dumps(summary, indent=2, default=str), "```",
        "", "## Metrics", "```json", json.dumps(metrics or {}, indent=2, default=str), "```",
        "", "## Limitations",
        "- Syncthing remote completion/status is unverified. Inputs are immutable, hash-verified local copies; newest and recently modified files excluded.",
        "- Reserved v2 test timestamp range and sessions excluded globally. No reserved labels or predictions loaded.",
        "- v3 holdout outcomes are never evaluated. Validation was used for checkpoint selection and is not an untouched test.",
        "- Training is deterministically thinned/capped for CPU compute; architecture matches phase7j/k but this is not an exact full-data rerun.",
        "- Validation windows do not overlap within each market/session. Related markets and events can still be correlated.",
        "- Gross execution assumes available sampled bid/ask quotes; fees, latency, capacity and slippage excluded. No maker fills inferred.",
        "- Canonical v2 models/scaler are absent locally, so no frozen v2 model comparison is claimed.",
        "", "## Resume", "```powershell", resume, "```", "",
        "## Collection checkpoints",
        "Keep the independent collector running. After seven additional calendar days, freeze a new immutable manifest and evaluate a fixed model on new dates only. Do not tune on that period and call it untouched.",
        "After two to four weeks, compare weekly results, market groups, dataset-size learning curves and gross execution economics. Trade-event collection requires a separate reviewed collector change; order-book deltas do not identify executions.",
        "Future snapshots must retain this run's split boundaries and must not reuse its holdout for training while it is reserved."]
    if error:
        text += ["", "## Error", "```", error, "```"]
    (run / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--model-hours", type=float, default=2.5)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    run = args.run_dir.resolve()
    if root not in run.parents or (root / "data") in run.parents:
        raise ValueError("Output must be inside workspace and outside synced data folder")
    run.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["python"] = sys.version
    config["label_policy"] = LABEL_POLICY
    config["disk_free_bytes"] = shutil.disk_usage(root).free
    config_path = run / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("label_policy") != LABEL_POLICY:
            raise ValueError("Legacy label policy: preserve this run and use a NEW run directory")
        for key in ("train_stride", "max_train_windows", "batch_size"):
            if config[key] != previous[key]:
                raise ValueError("Cannot change frozen configuration: " + key)
    else:
        save_json(config_path, config)
    save_json(run / "process.json", {"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()})
    summary = {}
    write_report(run, args, summary)
    try:
        protection_path = run / "protected_v2_period.json"
        if protection_path.exists():
            protection = json.loads(protection_path.read_text())
        else:
            protection = protected_period(root)
            save_json(protection_path, protection)
        status(run, "snapshot_start")
        manifest = snapshot(root, run)
        summary = build(root, run, manifest, protection)
        write_report(run, args, summary)
        status(run, "dataset_ready", summary=summary)
        metrics = train_evaluate(root, run, args)
        write_report(run, args, summary, metrics)
        status(run, "complete")
    except BaseException:
        error = traceback.format_exc()
        write_report(run, args, summary, error=error)
        status(run, "failed", error=error)
        raise


if __name__ == "__main__":
    main()
