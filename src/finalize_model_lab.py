"""Wait for a background Model Lab run, audit it, and publish a local pointer."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from src.audit_model_benchmark import audit
from src.movement_labels import LABEL_POLICY
from src.v3_research import save_json


def finalize(run, timeout_hours=6):
    run = run.resolve()
    root = Path.cwd().resolve()
    if root not in run.parents or root / "data" in run.parents:
        raise ValueError("Run must be inside project and outside synced data/")
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() < deadline:
        try:
            state = json.loads((run / "status.json").read_text())
        except (OSError, ValueError):
            time.sleep(5)
            continue
        if state["stage"] == "failed":
            raise RuntimeError("Research run failed; inspect its status and stderr")
        if state["stage"] == "complete":
            break
        time.sleep(5)
    else:
        raise TimeoutError("Research did not complete within the monitoring budget")
    benchmark = run.with_name(run.name + "_benchmark")
    audit_dir = run / "audit"
    try:
        if not (audit_dir / "audit.json").exists():
            audit(benchmark, audit_dir)
        result = json.loads((audit_dir / "audit.json").read_text())
        if result["label_policy"] != LABEL_POLICY or not result["status"].startswith("Corrected-label benchmark audit passed"):
            raise ValueError("Corrected benchmark audit did not pass")
        pointer = {"run_dir": str(run), "benchmark_dir": str(benchmark),
            "report": str(benchmark / "index.html"), "audit": str(audit_dir / "REPORT.md"),
            "label_policy": LABEL_POLICY, "verified_utc": datetime.now(timezone.utc).isoformat()}
        save_json(root / "artifacts/model_lab_latest.json", pointer)
        save_json(run / "verified_complete.json", pointer)
        print(json.dumps(pointer, indent=2), flush=True)
    except BaseException as error:
        save_json(run / "verification_failed.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run_dir)
