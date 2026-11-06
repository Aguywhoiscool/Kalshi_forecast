"""Freeze all completed feature files for development, excluding reserved periods."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import pandas as pd

from src.causal_swing_data import NS, validate_development_splits
from src.v3_research import digest, overlaps, save_json


def split_sessions(sessions):
    """Fixed chronological 80/20 session split, with a 70-second embargo."""
    sessions = sessions.sort_values(["input_start", "session_id"]).reset_index(drop=True)
    cut = int(len(sessions) * .8)
    if not 0 < cut < len(sessions):
        raise ValueError("Too few sessions")
    boundary = int(sessions.iloc[cut].input_start)
    train = sessions.loc[sessions.target_end <= boundary - 70 * NS].copy()
    val = sessions.iloc[cut:].copy()
    train["split"], val["split"] = "train", "val"
    return pd.concat([train, val], ignore_index=True)


def build(output, baseline):
    root, output, baseline = Path.cwd().resolve(), output.resolve(), baseline.resolve()
    if output.exists() or root not in output.parents or root / "data" in output.parents:
        raise ValueError("Use a new artifact directory outside data/")
    output.mkdir(parents=True)
    (output / "snapshot").mkdir()
    original = pd.read_parquet(baseline / "windows.parquet")
    holdout = original.loc[original.split == "holdout"]
    if holdout.empty:
        raise ValueError("Missing frozen v3 holdout identities")
    protected = json.loads((baseline / "protected_v2_period.json").read_text())
    protected["additional_periods"] = [*protected.get("additional_periods", []), {
        "start": int(holdout.input_start.min()), "end": int(holdout.target_end.max()),
        "name": "Frozen v3 holdout; timestamps only, no outcomes accessed"}]
    protected["sessions"] = sorted(set(protected["sessions"]) | set(holdout.session_id.astype(str)))
    save_json(output / "protected_v2_period.json", protected)
    paths = sorted((root / "data/processed").glob("features_*.csv"))
    excluded, files, rows = [], [], []
    cutoff = time.time() - 30 * 60
    for number, path in enumerate(paths):
        session = path.stem.removeprefix("features_")
        before = path.stat()
        reason = None
        if session in protected["sessions"]:
            reason = "reserved_session"
        elif path == paths[-1] or before.st_mtime >= cutoff or "sync-conflict" in path.name:
            reason = "active_recent_or_sync_conflict"
        else:
            # Read clocks only to protect time ranges before accessing any features.
            clock = pd.read_csv(path, usecols=["timestamp"])
            times = pd.to_datetime(clock.timestamp, utc=True, errors="coerce", format="ISO8601").dropna()
            if len(times) < 60:
                reason = "insufficient_timestamp_history"
            else:
                start, end = int(times.min().value), int(times.max().value)
                if overlaps(start, end, protected):
                    reason = "reserved_time_range"
        if reason is None:
            target = output / "snapshot" / path.name
            shutil.copyfile(path, target)
            fingerprint = digest(target)
            after_hash = digest(path)
            after = path.stat()
            if fingerprint != after_hash or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                target.unlink()
                reason = "changed_during_snapshot"
            else:
                files.append({"name": path.name, "size": before.st_size, "mtime_ns": before.st_mtime_ns,
                              "sha256": fingerprint})
                rows.append({"session_id": session, "input_start": start, "input_end": start,
                             "target_end": end, "source_rows": len(clock)})
        if reason:
            excluded.append({"name": path.name, "reason": reason})
        if number % 200 == 0:
            print(json.dumps({"stage": "expanded_snapshot", "inspected": number+1,
                              "copied": len(files), "excluded": len(excluded)}), flush=True)
    meta = split_sessions(pd.DataFrame(rows))
    validate_development_splits(meta, protected)
    meta.to_parquet(output / "windows.parquet", index=False)
    save_json(output / "input_manifest.json", {"files": files, "excluded": excluded,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "All stable completed files; frozen v2/v3 sessions and periods excluded; newest/recent files excluded",
        "syncthing_status": "Remote completion not confirmed; copied bytes checked against source SHA256 and size/mtime"})
    summary = {"baseline": str(baseline), "available_files": len(paths), "snapshot_files": len(files),
        "snapshot_rows": sum(row["source_rows"] for row in rows), "eligible_source_files": len(meta),
        "split": "Chronological 80/20 sessions with 70-second embargo; calibration inside training",
        "comparability": "New validation period; comparisons with prior runs are exploratory, not a controlled data-volume test",
        "counts": meta.groupby("split").size().to_dict(),
        "embargo_excluded_files": len(rows)-len(meta)}
    save_json(output / "development_source.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/model_lab_corrected_20261104"))
    args = parser.parse_args()
    build(args.output, args.baseline)
