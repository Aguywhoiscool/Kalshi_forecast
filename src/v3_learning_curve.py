"""Sequential offline 25%-versus-100% training-data comparison on fixed validation.

Waits for the primary run; never evaluates either holdout or v2 test outcomes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import pandas as pd

from src.v3_research import digest, save_json, status, train_evaluate, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    args = parser.parse_args()
    primary = args.primary.resolve()
    root = Path.cwd().resolve()
    if root not in primary.parents or (root / "data") in primary.parents:
        raise ValueError("Primary run must be outside synced data and inside workspace")
    run = primary / "learning_curve_25pct"
    run.mkdir(exist_ok=True)
    save_json(run / "process.json", {"pid": os.getpid(),
        "started": datetime.now(timezone.utc).isoformat()})
    status(run, "waiting_for_primary")
    timeout = time.monotonic() + 6 * 3600
    while True:
        state = json.loads((primary / "status.json").read_text())
        if state["stage"] == "failed":
            raise RuntimeError("Primary run failed; learning curve not started")
        if state["stage"] == "complete":
            break
        if time.monotonic() >= timeout:
            raise TimeoutError("Primary run has not completed within six hours")
        time.sleep(30)
    try:
        provenance = []
        for source in sorted((primary / "shards").glob("*.npy")):
            target = run / "shards" / source.name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if digest(source) != digest(target):
                raise ValueError("Learning-curve shard copy mismatch")
            provenance.append({"name": source.name, "sha256": digest(target)})
        for name in ("windows.parquet", "dataset_complete.json", "protected_v2_period.json"):
            shutil.copyfile(primary / name, run / name)
        save_json(run / "prepared_data_provenance.json", provenance)
        config = json.loads((primary / "config.json").read_text())
        config.update(train_fraction=.25, run_dir=str(run))
        save_json(run / "config.json", config)
        options = argparse.Namespace(**config)
        summary = json.loads((run / "dataset_complete.json").read_text())
        status(run, "prepared")
        metrics = train_evaluate(root, run, options)
        write_report(run, options, summary, metrics)
        report = run / "REPORT.md"
        report_text = report.read_text(encoding="utf-8")
        report_text = report_text.replace(
            f'python -m src.v3_research --run-dir "{run}" --epochs {options.epochs} --model-hours {options.model_hours}',
            f'python -m src.v3_learning_curve --primary "{primary}"')
        report.write_text(report_text + "\nTraining fraction: earliest 25% of primary sampled training windows. Prepared observation shards were hash-verified copies from the primary frozen run.\n", encoding="utf-8")
        primary_metrics = json.loads((primary / "metrics.json").read_text())
        comparison = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "Same fixed non-overlapping validation; 25% is earliest chronological subset of primary sampled training windows. Both fit own train-only scaler. Single seed, no significance claim.",
            "quarter_data": metrics, "full_sampled_data": primary_metrics}
        save_json(primary / "learning_curve_comparison.json", comparison)
        gross = pd.concat([pd.read_csv(run / "gross_execution.csv").assign(training_fraction=.25),
                           pd.read_csv(primary / "gross_execution.csv").assign(training_fraction=1.)])
        gross.to_csv(primary / "learning_curve_execution.csv", index=False)
        status(run, "complete")
    except BaseException:
        status(run, "failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
