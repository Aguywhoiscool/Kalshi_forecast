"""Retrain corrected models against an immutable copy of the original dataset.

Run: python -u -m src.rerun_model_lab --source artifacts/v3_20261104_overnight
No client imports, order submission, live collection, or holdout evaluation.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import pandas as pd

from src.model_benchmark import run_benchmark, validate_splits
from src.movement_labels import LABEL_POLICY
from src.v3_research import digest, save_json, train_evaluate, write_report


def clone_dataset(source, destination):
    """Copy only data/metadata, never models, scalers, predictions or checkpoints."""
    for name in ("snapshot", "shards", "windows.parquet", "input_manifest.json",
                 "protected_v2_period.json", "dataset_complete.json"):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    protected = json.loads((source / "protected_v2_period.json").read_text())
    validate_splits(pd.read_parquet(source / "windows.parquet"), protected)
    expected = []
    for directory in ("snapshot", "shards"):
        expected.extend(path for path in sorted((source / directory).iterdir()) if path.is_file())
    expected.extend(source / name for name in (
        "windows.parquet", "input_manifest.json", "protected_v2_period.json", "dataset_complete.json"))
    hashes = []
    for path in expected:
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            raise FileExistsError("Cannot overwrite copied data: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        fingerprint = digest(path)
        shutil.copyfile(path, target)
        after = path.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or digest(target) != fingerprint or digest(path) != fingerprint):
            raise ValueError("Dataset changed while copying: " + str(path))
        hashes.append({"path": relative.as_posix(), "sha256": fingerprint, "bytes": before.st_size})
    save_json(destination / "frozen_dataset_provenance.json", {
        "source": str(source), "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": hashes, "policy": "Independent verified copies; exact old windows/splits; no resplitting or new source ingestion"})
    manifest = json.loads((destination / "input_manifest.json").read_text())
    for item in manifest["files"]:
        if digest(destination / "snapshot" / item["name"]) != item["sha256"]:
            raise ValueError("Original snapshot manifest mismatch: " + item["name"])


def workflow(source, destination, epochs=8, prepare_only=False):
    root = Path.cwd().resolve()
    source, destination = source.resolve(), destination.resolve()
    if (root not in destination.parents or root / "data" in destination.parents
            or source in destination.parents or source == destination):
        raise ValueError("Use a separate run directory inside project and outside data/")
    if destination.exists():
        raise FileExistsError("A new directory is required; existing runs are preserved")
    destination.mkdir(parents=True)
    save_json(destination / "process.json", {"pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat()})

    def progress(stage, **extra):
        state = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
        save_json(destination / "status.json", state)
        print(json.dumps(state), flush=True)

    try:
        progress("copying_frozen_dataset")
        clone_dataset(source, destination)
        old = json.loads((source / "config.json").read_text())
        options = argparse.Namespace(run_dir=str(destination), epochs=epochs,
            model_hours=2.5, train_stride=old["train_stride"],
            max_train_windows=old["max_train_windows"], batch_size=old["batch_size"],
            train_fraction=old.get("train_fraction", 1.))
        config = {**vars(options), "label_policy": LABEL_POLICY,
            "python": sys.version, "frozen_dataset_source": str(source)}
        save_json(destination / "config.json", config)
        summary = json.loads((destination / "dataset_complete.json").read_text())
        write_report(destination, options, summary)
        if prepare_only:
            progress("prepared")
            return
        progress("dataset_ready", summary=summary)
        started = time.monotonic()
        metrics = train_evaluate(root, destination, options)
        write_report(destination, options, summary, metrics)
        progress("benchmarking", training_seconds=time.monotonic() - started)
        benchmark = destination.with_name(destination.name + "_benchmark")
        run_benchmark(destination, benchmark)
        progress("auditing_benchmark")
        from src.audit_model_benchmark import audit
        audit(benchmark, destination / "audit")
        # Updated default pointers are written only after the whole workflow succeeds.
        save_json(root / "artifacts/model_lab_latest.json", {
            "run_dir": str(destination), "benchmark_dir": str(benchmark),
            "report": str(benchmark / "index.html"), "label_policy": LABEL_POLICY})
        progress("complete", report=str(benchmark / "index.html"))
    except BaseException:
        progress("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/v3_20261104_overnight"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts") / (
        "model_lab_corrected_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    workflow(args.source, args.run_dir, args.epochs, args.prepare_only)
