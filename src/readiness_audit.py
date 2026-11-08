"""Incremental, offline schedule-readiness report. Never trains or submits orders."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from src.causal_swing_data import history_ends
from src.v3_research import FEATURES, overlaps, save_json, valid_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START = "2026-11-07T04:01:14Z"
VERSION = 1


@contextmanager
def single_run(path):
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno in (11, 13):
                    raise BlockingIOError("Audit already running") from error
                raise
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def scan_frame(frame, start, protected):
    required = ["timestamp", "session_id", "ticker", *FEATURES,
                "scheduled_close_time", "close_time_observed_at", "seconds_to_scheduled_close"]
    missing = sorted(set(required)-set(frame.columns))
    if missing:
        return {"schema_missing": missing, "rows": 0, "windows": []}
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
    clocks = frame.timestamp.array.as_unit("ns").asi8
    frame = frame.loc[frame.timestamp.notna() & (frame.timestamp >= start)
        & ~frame.session_id.isin(protected["sessions"]) & ~overlaps(clocks, clocks, protected)]
    duplicates = int(frame.duplicated(["session_id", "ticker", "timestamp"]).sum())
    frame = frame.drop_duplicates(["session_id", "ticker", "timestamp"]).copy()
    close = pd.to_datetime(frame.scheduled_close_time, utc=True, errors="coerce", format="ISO8601")
    observed = pd.to_datetime(frame.close_time_observed_at, utc=True, errors="coerce", format="ISO8601")
    seconds = pd.to_numeric(frame.seconds_to_scheduled_close, errors="coerce")
    present = frame[["scheduled_close_time", "close_time_observed_at", "seconds_to_scheduled_close"]].notna().any(axis=1)
    complete = close.notna() & observed.notna() & np.isfinite(seconds)
    future = complete & (observed > frame.timestamp)
    wrong = complete & (((close-frame.timestamp).dt.total_seconds()-seconds).abs() > .001)
    verified = complete & ~future & ~wrong
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    quote_valid = valid_rows(frame)
    frame["verified"] = verified
    frame["seconds"] = seconds
    frame["quote_valid"] = quote_valid
    windows = []
    for (session, ticker), group in frame.groupby(["session_id", "ticker"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        times = group.timestamp.array.as_unit("ns").asi8
        ends = history_ends(times, (group.verified & group.quote_valid).to_numpy())
        next_signal = -1
        for end in ends:
            if times[end] < next_signal:
                continue
            next_signal = int(times[end])+120*10**9
            remaining = float(group.seconds.iloc[end])
            if not 600 < remaining <= 3600:
                continue
            windows.append({"session": str(session), "ticker": str(ticker),
                "event_proxy": str(ticker).rsplit("-", 1)[0], "time": int(times[end]),
                "date": group.timestamp.iloc[end].date().isoformat()})
    return {"rows": len(frame), "duplicates": duplicates, "schema_missing": [],
        "verified_rows": int(verified.sum()), "blank_metadata_rows": int((~present).sum()),
        "invalid_metadata_rows": int((present & ~verified).sum()),
        "future_knowledge_rows": int(future.sum()), "countdown_errors": int(wrong.sum()),
        "invalid_quote_rows": int((~quote_valid).sum()),
        "near_close_rows": int((verified & (seconds > 600) & (seconds <= 3600)).sum()),
        "latest_sample_ns": int(frame.timestamp.max().value) if len(frame) else None,
        "windows": windows}


def assess(windows, invalid, errors, latest_age, min_windows=1000, min_sessions=50, min_events=20, min_days=3):
    counts = {"near_close_windows": len(windows), "sessions": len({w["session"] for w in windows}),
              "event_proxies": len({w["event_proxy"] for w in windows}),
              "utc_days": len({w["date"] for w in windows})}
    targets = dict(zip(counts, [min_windows, min_sessions, min_events, min_days]))
    reasons = [f"{key}: {value}/{targets[key]}" for key, value in counts.items() if value < targets[key]]
    if invalid:
        reasons.append("Invalid schedule metadata needs review")
    if errors:
        reasons.append("File read/schema errors need review")
    if latest_age is None or latest_age > 3600:
        reasons.append("No recently completed synced observations within one hour")
    elif latest_age < -2:
        reasons.append("Future-dated observations need a clock review")
    return {"status": "collecting" if reasons else "ready_for_pilot_review", "counts": counts,
        "screening_targets": targets, "reasons": reasons,
        "meaning": "Heuristic pilot-data screen, not proof of statistical power, model quality or profitability"}


def audit(directory, output, protected_path, start, stable_seconds=120):
    output.mkdir(parents=True, exist_ok=True)
    protected = json.loads(protected_path.read_text())
    config = {"version": VERSION, "directory": str(directory.resolve()), "start": start.isoformat(),
              "protected_sha256": hashlib.sha256(protected_path.read_bytes()).hexdigest()}
    cache_path = output / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    old = cache.get("files", {}) if cache.get("config") == config else {}
    files = sorted(directory.glob("features_*.csv"))
    current, skipped, errors = {}, [], []
    now = time.time()
    earliest_name = "features_"+start.strftime("%Y-%m-%d_%H-%M")
    for path in files:
        if path.name < earliest_name or "sync-conflict" in path.name:
            continue
        if path.stem.removeprefix("features_") in protected["sessions"]:
            continue
        try:
            before = path.stat()
            if path == files[-1] or now-before.st_mtime < stable_seconds:
                skipped.append(path.name)
                continue
            fingerprint = [before.st_size, before.st_mtime_ns]
            if old.get(path.name, {}).get("fingerprint") == fingerprint:
                current[path.name] = old[path.name]
                continue
            frame = pd.read_csv(path)
            if "session_id" in frame and not frame.session_id.eq(path.stem.removeprefix("features_")).all():
                raise ValueError("Filename/session mismatch")
            result = scan_frame(frame, start, protected)
            after = path.stat()
            if [after.st_size, after.st_mtime_ns] != fingerprint:
                skipped.append(path.name)
                continue
            current[path.name] = {"fingerprint": fingerprint, "result": result}
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            errors.append({"file": path.name, "error": type(error).__name__})
    totals, windows, latest = {}, [], None
    for name, item in current.items():
        result = item["result"]
        if result["schema_missing"]:
            errors.append({"file": name, "error": "Missing metadata columns", "columns": result["schema_missing"]})
        for key, value in result.items():
            if isinstance(value, int) and key != "latest_sample_ns":
                totals[key] = totals.get(key, 0)+value
        windows.extend(result["windows"])
        if result.get("latest_sample_ns") is not None:
            latest = max(latest or 0, result["latest_sample_ns"])
    age = now-latest/1e9 if latest else None
    screen = assess(windows, totals.get("invalid_metadata_rows", 0), errors, age)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "files_scanned": len(current), "files_reused": sum(old.get(k) == v for k, v in current.items()),
        "active_recent_or_changing_files_skipped": len(skipped), "totals": totals, "readiness": screen,
        "latest_completed_sample_utc": pd.Timestamp(latest, tz="UTC").isoformat() if latest else None,
        "latest_sample_age_seconds": age, "errors": errors,
        "limitations": ["Metadata-only readiness; no future return targets calculated",
            "Requires schedule metadata known at each of all 60 history observations",
            "Signals spaced 120 seconds per market/session; related markets may remain correlated",
            "Event grouping is a ticker-prefix proxy, not verified exchange event metadata",
            "Pilot screen does not establish a chronological validation split or statistical power",
            "Newest/recent/changing files excluded; Syncthing remote completion is not independently verified",
            "No training, trades, network requests or collector actions"]}
    save_json(cache_path, {"config": config, "files": current})
    save_json(output / "status.json", summary)
    lines = ["# Near-close data readiness", "", f"Updated: {summary['generated_utc']}",
        f"Status: **{screen['status']}**", "", "## Pilot-data screen", "",
        "| Measure | Observed | Screening target |", "|---|---:|---:|"]
    lines += [f"| {key} | {value:,} | {screen['screening_targets'][key]:,} |" for key, value in screen["counts"].items()]
    lines += ["", f"Verified rows: {totals.get('verified_rows', 0):,}; near-close rows: {totals.get('near_close_rows', 0):,}.",
        f"Invalid metadata rows: {totals.get('invalid_metadata_rows', 0):,}; file errors: {len(errors)}.",
        f"Stable files: {len(current)}; active/recent/changing skipped: {len(skipped)}.", "",
        "## Remaining checks", "", *["- "+reason for reason in screen["reasons"]],
        "", "A ready screen means review the dataset next; it does not trigger training or prove profitability.",
        "", "## Limits", "", *["- "+line for line in summary["limitations"]]]
    target = output / "LATEST.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(target)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/readiness")
    parser.add_argument("--protected", type=Path, default=ROOT / "artifacts/swing_source_all_20261106/protected_v2_period.json")
    parser.add_argument("--since", default=DEFAULT_START)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or ROOT / "data" in output.parents:
        parser.error("Output must be inside workspace and outside data/")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with single_run(output / "audit.lock"):
            try:
                result = audit(args.data_dir, output, args.protected, pd.Timestamp(args.since))
                save_json(output / "last_run.json", {"status": "success", "utc": result["generated_utc"]})
            except Exception as error:
                save_json(output / "last_run.json", {"status": "failed", "utc": datetime.now(timezone.utc).isoformat(),
                    "error": type(error).__name__, "message": str(error)[:300]})
                raise
    except BlockingIOError:
        return


if __name__ == "__main__":
    main()
