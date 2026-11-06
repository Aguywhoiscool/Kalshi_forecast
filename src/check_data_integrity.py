"""Read-only feature integrity, frozen hashes and sampled raw replay audit.

Run: python -u -m src.check_data_integrity
No account calls, data repairs, model fitting, or reserved target evaluation.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.feature_engineering import extract_features
from src.orderbook import OrderBookManager
from src.v3_research import FEATURES, digest, save_json


REQUIRED = ["timestamp", "session_id", "ticker", *FEATURES]
ROLLOUT = pd.Timestamp("2026-11-04T16:14:27Z")


def row_checks(frame):
    """Preserve original ordering; never silently discard invalid observations."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    frame = frame.copy()
    original_time = frame.timestamp.astype(str)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    bid, ask, spread, mid, bd, ad, imbalance = values.T
    depth = bd + ad
    expected = np.divide(bd - ad, depth, out=np.zeros_like(depth), where=depth > 0)
    checks = {
        "nonfinite_features": ~np.isfinite(values).all(axis=1),
        "invalid_timestamp": frame.timestamp.isna().to_numpy(),
        "timestamp_missing_timezone": ~original_time.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).to_numpy(),
        "missing_identity": (frame.session_id.isna() | frame.ticker.isna()
            | frame.session_id.astype(str).str.strip().eq("")
            | frame.ticker.astype(str).str.strip().eq("")).to_numpy(dtype=bool, na_value=True),
        "invalid_quote": ~((bid >= 0) & (bid < ask) & (ask <= 1)),
        "invalid_depth": ~((bd >= 0) & (ad >= 0)),
        "spread_inconsistent": ~(np.isfinite(spread) & (spread > 0) & (np.abs(spread - (ask - bid)) <= 1e-6)),
        "midpoint_inconsistent": ~(np.isfinite(mid) & (np.abs(mid - (bid + ask) / 2) <= 1e-6)),
        "imbalance_inconsistent": ~(np.isfinite(imbalance) & (np.abs(imbalance) <= 1 + 1e-6)
            & (np.abs(imbalance - expected) <= 1e-5)),
    }
    valid = ~np.logical_or.reduce(list(checks.values()))
    counts = {key: int(value.sum()) for key, value in checks.items()}
    keys = ["session_id", "ticker", "timestamp"]
    counts["duplicate_rows"] = int(frame.duplicated(keys).sum())
    counts["conflicting_duplicate_keys"] = int(frame.drop_duplicates(keys + FEATURES).duplicated(keys).sum())
    counts["invalid_rows"] = int((~valid).sum())
    counts["usable_rows"] = int((valid & (spread <= .10 + 1e-6)).sum())
    counts["wide_rows"] = int((valid & (spread > .10 + 1e-6)).sum())
    examples = frame.loc[~valid, keys].head(5).astype(str).to_dict("records")
    return frame, counts, examples


def timing_counts(frame):
    counts = Counter(backward_market_timestamps=0, market_intervals_over_1_5s=0)
    largest = 0.
    for _, group in frame.dropna(subset=["timestamp", "session_id", "ticker"]).groupby(["session_id", "ticker"], sort=False):
        differences = group.timestamp.diff().dt.total_seconds()
        counts["backward_market_timestamps"] += int((differences < 0).sum())
        long = differences > 1.5
        counts["market_intervals_over_1_5s"] += int(long.sum())
        if differences.notna().any():
            largest = max(largest, float(differences.max()))
    counts["largest_market_gap_seconds"] = largest
    return dict(counts)


def feature_scan(root, output, progress):
    files = sorted((root / "data/processed").glob("features_*.csv"))
    newest = files[-1] if files else None
    cutoff = time.time() - 15 * 60
    results, inventory = [], []
    hashes, session_files = defaultdict(list), defaultdict(list)
    for number, path in enumerate(files, 1):
        before = path.stat()
        provisional = path == newest or before.st_mtime >= cutoff
        result = {"file": path.name, "bytes": before.st_size,
                  "provisional": provisional, "sync_conflict_name": "sync-conflict" in path.name.lower()}
        try:
            data = path.read_bytes()
            fingerprint = hashlib.sha256(data).hexdigest()
            after = path.stat()
            changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            result.update(sha256=fingerprint, changed_during_read=changed,
                provisional=provisional or changed, missing_final_newline=bool(data) and not data.endswith(b"\n"))
            frame = pd.read_csv(io.BytesIO(data), dtype={"session_id": "string", "ticker": "string"})
            frame, checks, examples = row_checks(frame)
            result.update(rows=len(frame), **checks, invalid_examples=examples,
                sessions=frame.session_id.dropna().astype(str).unique().tolist(),
                tickers=frame.ticker.dropna().astype(str).unique().tolist(),
                first_timestamp=str(frame.timestamp.min()), last_timestamp=str(frame.timestamp.max()),
                empty_recording=frame.empty)
            result.update(timing_counts(frame))
            post = frame.loc[frame.timestamp >= ROLLOUT]
            result["post_rollout_rows"] = len(post)
            result["post_rollout_timing"] = timing_counts(post)
            for session in result["sessions"]:
                session_files[session].append(path.name)
            hashes[fingerprint].append(path.name)
        except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as error:
            result["read_error"] = str(error)
        inventory.append({"path": path, "size": before.st_size, "mtime_ns": before.st_mtime_ns})
        results.append(result)
        if number % 100 == 0:
            progress("feature_scan", files=number, total=len(files))
    # Re-check metadata after the complete pass: a file replaced later in the
    # scan cannot be called a stable observation of the synchronized folder.
    for result, item in zip(results, inventory):
        try:
            after = item["path"].stat()
            result["changed_by_scan_end"] = (after.st_size, after.st_mtime_ns) != (item["size"], item["mtime_ns"])
        except OSError:
            result["changed_by_scan_end"] = True
        result["provisional"] |= result["changed_by_scan_end"]
    cross_file = []
    for session, names in session_files.items():
        if len(names) <= 1:
            continue
        pieces = []
        for name in names:
            try:
                frame = pd.read_csv(root / "data/processed" / name)
                frame = frame.loc[frame.session_id.astype(str) == session].copy()
                frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
                frame["source_file"] = name
                pieces.append(frame)
            except (OSError, ValueError):
                pass
        if pieces:
            joined = pd.concat(pieces, ignore_index=True)
            keys = ["session_id", "ticker", "timestamp"]
            repeated = joined.loc[joined.duplicated(keys, keep=False)]
            if not repeated.empty:
                cross_file.append({"session_id": session, "files": names,
                    "duplicate_key_rows": len(repeated),
                    "conflicting_keys": int(repeated.drop_duplicates(keys + FEATURES).duplicated(keys).sum())})
    save_json(output / "feature_files.json", results)
    save_json(output / "cross_file_duplicates.json", cross_file)
    save_json(output / "identical_content_files.json", [names for names in hashes.values() if len(names) > 1])
    return results, cross_file


def sampled_lines(path, head_limit=5000, tail_limit=1000):
    """Offsets prevent double-counting small files; tail reads at most 8 MiB."""
    with path.open("rb") as stream:
        for _ in range(head_limit):
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            yield "head", offset, line
        head_end = stream.tell()
        size = path.stat().st_size
        start = max(0, size - 8 * 1024 * 1024)
        stream.seek(start)
        if start:
            stream.readline()  # A partial first line is not a malformed record.
        candidates = []
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if offset >= head_end:
                candidates.append((offset, line))
        for offset, line in candidates[-tail_limit:]:
            yield "tail", offset, line


def check_raw_message(message):
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ValueError("Invalid message envelope")
    if message["type"] not in ("orderbook_snapshot", "orderbook_delta"):
        return
    if not isinstance(message.get("seq"), int) or message["seq"] < 1 or "sid" not in message:
        raise ValueError("Missing sequence/subscription")
    payload = message.get("msg")
    if not isinstance(payload, dict) or not payload.get("market_ticker"):
        raise ValueError("Missing book payload/ticker")
    if message["type"] == "orderbook_delta":
        price = Decimal(str(payload["price_dollars"]))
        delta = Decimal(str(payload.get("delta_fp", payload.get("delta"))))
        if not (price.is_finite() and 0 <= price <= 1 and delta.is_finite()
                and payload.get("side") in ("yes", "no")):
            raise ValueError("Invalid delta values")
    else:
        for side in ("yes", "no"):
            levels = payload.get(side + "_dollars_fp", payload.get(side + "_dollars", []))
            if levels is None:
                levels = []
            for price, quantity in levels:
                price, quantity = Decimal(str(price)), Decimal(str(quantity))
                if not (price.is_finite() and 0 <= price <= 1 and quantity.is_finite() and quantity >= 0):
                    raise ValueError("Invalid snapshot level")


def replay_comparison(records, root, features):
    if not records:
        return {"samples": 0}, []
    start, end = records[0][0], records[-1][0]
    if any(records[i][0] < records[i-1][0] for i in range(1, len(records))):
        return {"samples": 0, "skipped_nonmonotonic_raw_head": 1}, []
    parts = []
    for info in features:
        if info.get("provisional") or info.get("read_error") or not info.get("rows"):
            continue
        if pd.Timestamp(info["first_timestamp"]) > end or pd.Timestamp(info["last_timestamp"]) < start:
            continue
        frame = pd.read_csv(root / "data/processed" / info["file"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, format="ISO8601")
        frame = frame.loc[frame.timestamp.between(start, end)]
        if len(frame) > 300:
            frame = frame.iloc[np.linspace(0, len(frame) - 1, 300, dtype=int)]
        if len(frame):
            frame = frame.copy()
            frame["source_file"] = info["file"]
            parts.append(frame)
    if not parts:
        return {"samples": 0}, []
    samples = pd.concat(parts).sort_values("timestamp", kind="stable")
    manager, position = OrderBookManager(), 0
    counts, errors = Counter(), []

    def apply(message):
        nonlocal manager
        if message["type"] == "subscribed":
            manager = OrderBookManager()  # sid values are reused across lifetimes.
        elif message["type"] in ("orderbook_snapshot", "orderbook_delta"):
            try:
                manager.process(message)
            except (ValueError, KeyError, TypeError, InvalidOperation):
                counts["replay_book_errors"] += 1

    def differences(actual, sample):
        if actual is None:
            return list(FEATURES)
        return [field for field in FEATURES
                if not np.isclose(float(actual[field]), float(getattr(sample, field)),
                    rtol=0, atol=1e-6 if "depth" in field else 1e-9)]

    for timestamp, group in samples.groupby("timestamp", sort=True):
        while position < len(records) and records[position][0] < timestamp:
            apply(records[position][1])
            position += 1
        # Equal wall-clock timestamps do not establish which came first. Keep
        # the pre-event state and each possible prefix of this timestamp bucket.
        candidates = {ticker: [extract_features(manager.get_prices(ticker))] for ticker in group.ticker.unique()}
        tied = False
        while position < len(records) and records[position][0] == timestamp:
            tied = True
            apply(records[position][1])
            position += 1
            for ticker in candidates:
                candidates[ticker].append(extract_features(manager.get_prices(ticker)))
        for sample in group.itertuples():
            counts["samples"] += 1
            actual = extract_features(manager.get_prices(sample.ticker))
            delta = differences(actual, sample)
            if not delta:
                counts["matched_samples"] += 1
            elif tied and any(not differences(candidate, sample) for candidate in candidates[sample.ticker]):
                counts["ambiguous_same_timestamp_samples"] += 1
            elif actual is None:
                counts["unavailable_reconstructed_quote"] += 1
            else:
                counts["mismatched_samples"] += 1
                if len(errors) < 20:
                    errors.append({"file": sample.source_file, "timestamp": str(sample.timestamp),
                        "ticker": sample.ticker, "fields": delta,
                        "processed": {key: float(getattr(sample, key)) for key in delta},
                        "raw_replay": {key: float(actual[key]) for key in delta}})
    return dict(counts), errors


def raw_scan(root, features, output, progress):
    results, discrepancies = [], []
    files = sorted((root / "data/raw").glob("*.jsonl"))
    for number, path in enumerate(files, 1):
        before = path.stat()
        result = {"file": path.name, "bytes": before.st_size,
                  "provisional": path == files[-1] or before.st_mtime > time.time() - 900}
        counts, examples, records = Counter(), [], []
        sequences, region, last_time = {}, None, None
        for segment, offset, line in sampled_lines(path):
            counts["sampled_lines"] += 1
            if segment != region:
                sequences, last_time, region = {}, None, segment
            if not line.endswith(b"\n"):
                counts["unterminated_sampled_lines"] += 1
                if result["provisional"]:
                    counts["deferred_in_progress_tail"] += 1
                    continue
            try:
                record = json.loads(line)
                message = record["message"]
                check_raw_message(message)
                received = pd.Timestamp(record["received_at"])
                if received.tzinfo is None:
                    raise ValueError("Naive raw receive timestamp")
                if last_time is not None and received < last_time:
                    counts["backward_receive_timestamps"] += 1
                last_time = received
                kind = message["type"]
                counts[kind] += 1
                if kind == "subscribed":
                    sequences.pop((message.get("msg") or {}).get("sid"), None)
                    counts["subscription_lifetime_resets"] += 1
                elif kind in ("orderbook_snapshot", "orderbook_delta"):
                    sid, seq = message["sid"], message["seq"]
                    if sid in sequences and seq != sequences[sid] + 1:
                        counts["forward_sequence_gaps" if seq > sequences[sid] + 1 else "backward_or_duplicate_sequences"] += 1
                    sequences[sid] = seq
                if segment == "head":
                    records.append((received, message))
            except (ValueError, KeyError, TypeError, UnicodeError, InvalidOperation) as error:
                counts["invalid_sampled_records"] += 1
                if len(examples) < 5:
                    examples.append({"byte_offset": offset, "error": str(error)})
        after = path.stat()
        result["changed_during_scan"] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        result["provisional"] |= result["changed_during_scan"]
        comparison, errors = replay_comparison(records, root, features)
        discrepancies.extend({"raw_file": path.name, **error} for error in errors)
        result.update(counts=dict(counts), examples=examples, replay=comparison)
        results.append(result)
        progress("raw_sample", files=number, total=len(files))
    save_json(output / "raw_files.json", results)
    save_json(output / "raw_feature_discrepancies.json", discrepancies)
    return results


def frozen_checks(root, features, output, progress):
    pointer = root / "artifacts/model_lab_latest.json"
    if not pointer.exists():
        return {"status": "no_frozen_run_pointer"}
    run = Path(json.loads(pointer.read_text())["run_dir"])
    provenance = json.loads((run / "frozen_dataset_provenance.json").read_text())
    mismatches, missing = [], []
    for number, item in enumerate(provenance["files"], 1):
        path = run / item["path"]
        if not path.exists():
            missing.append(item["path"])
        elif digest(path) != item["sha256"]:
            mismatches.append(item["path"])
        if number % 500 == 0:
            progress("frozen_hashes", files=number, total=len(provenance["files"]))
    live = {info["file"]: info for info in features}
    manifest = json.loads((run / "input_manifest.json").read_text())
    live_changed, live_missing = [], []
    for item in manifest["files"]:
        if item["name"] not in live:
            live_missing.append(item["name"])
        elif live[item["name"]].get("sha256") != item["sha256"]:
            live_changed.append(item["name"])
    result = {"run": str(run), "files_verified": len(provenance["files"]),
        "missing_frozen_files": missing, "mismatched_frozen_files": mismatches,
        "live_files_differing_from_frozen_manifest": live_changed,
        "live_files_missing_from_frozen_manifest": live_missing,
        "status": "passed" if not missing and not mismatches else "failed"}
    save_json(output / "frozen_hash_verification.json", result)
    return result


def run(output):
    root = Path.cwd().resolve()
    output = output.resolve()
    if root not in output.parents or root / "data" in output.parents:
        raise ValueError("Reports must be inside project and outside data/")
    if output.exists():
        raise FileExistsError("Use a new report directory")
    output.mkdir(parents=True)

    def progress(stage, **values):
        state = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **values}
        save_json(output / "status.json", state)
        print(json.dumps(state), flush=True)

    progress("started")
    features, cross = feature_scan(root, output, progress)
    raw = raw_scan(root, features, output, progress)
    frozen = frozen_checks(root, features, output, progress)
    stable = [row for row in features if not row["provisional"]]
    readable = [row for row in stable if not row.get("read_error")]
    raw_counts, replay = Counter(), Counter()
    for row in raw:
        raw_counts.update(row["counts"])
        replay.update(row["replay"])
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_files_found": len(features), "stable_feature_files": len(stable),
        "provisional_feature_files": len(features) - len(stable),
        "stable_read_errors": [row for row in stable if row.get("read_error")],
        "stable_rows": sum(row["rows"] for row in readable),
        "invalid_rows": sum(row["invalid_rows"] for row in readable),
        "usable_rows": sum(row["usable_rows"] for row in readable),
        "wide_rows": sum(row["wide_rows"] for row in readable),
        "duplicate_rows": sum(row["duplicate_rows"] for row in readable),
        "conflicting_duplicate_keys": sum(row["conflicting_duplicate_keys"] for row in readable),
        "cross_file_duplicate_groups": len(cross),
        "backward_market_timestamps": sum(row["backward_market_timestamps"] for row in readable),
        "market_intervals_over_1_5s": sum(row["market_intervals_over_1_5s"] for row in readable),
        "largest_market_gap_seconds": max((row["largest_market_gap_seconds"] for row in readable), default=0),
        "post_rollout_rows": sum(row["post_rollout_rows"] for row in readable),
        "post_rollout_market_gaps_over_1_5s": sum(row["post_rollout_timing"].get("market_intervals_over_1_5s", 0) for row in readable),
        "post_rollout_largest_market_gap_seconds": max((row["post_rollout_timing"].get("largest_market_gap_seconds", 0) for row in readable), default=0),
        "markets": len(set(ticker for row in readable for ticker in row["tickers"])),
        "sessions": len(set(session for row in readable for session in row["sessions"])),
        "first_timestamp": min((row["first_timestamp"] for row in readable if row["rows"]), default=None),
        "last_timestamp": max((row["last_timestamp"] for row in readable if row["rows"]), default=None),
        "raw_files": len(raw), "raw_inventory_bytes": sum(row["bytes"] for row in raw),
        "raw_sample_counts": dict(raw_counts), "raw_feature_replay": dict(replay),
        "frozen_dataset": frozen,
        "limitations": ["Local synchronized copy only; remote Syncthing Up-to-Date state not inspected",
            "Recent/newest/changing feature files are provisional and excluded from stable totals",
            "Raw scan samples first 5,000 and last 1,000 messages per file, without overlap; no full raw scan",
            "Raw-feature replay covers sampled beginnings only; no quote-age rejection",
            "Recorded within-market gaps can reflect unavailable quotes; they are not proven collector pauses",
            "Post-rollout totals are descriptive; no causal before/after claim",
            "No data repair, deletion, source edits, trades or reserved target evaluation"]}
    save_json(output / "summary.json", result)
    report = ["# Data integrity audit", "", f"Stable rows: {result['stable_rows']:,}",
        f"Invalid rows: {result['invalid_rows']:,}; duplicates: {result['duplicate_rows']:,}; cross-file duplicate groups: {len(cross)}.",
        f"Usable at spread <=10 cents: {result['usable_rows']:,}; wide spread: {result['wide_rows']:,}.",
        f"Stable files: {len(stable)}; provisional files: {len(features)-len(stable)}; stable read errors: {len(result['stable_read_errors'])}.",
        f"Within-market gaps >1.5s: {result['market_intervals_over_1_5s']:,}; largest {result['largest_market_gap_seconds']:.3f}s.",
        f"Post-rollout rows: {result['post_rollout_rows']:,}; recorded market gaps >1.5s: {result['post_rollout_market_gaps_over_1_5s']:,}.",
        "", "## Sampled raw records", "```json", json.dumps(dict(raw_counts), indent=2), "```",
        "", "## Sampled raw/feature replay", "```json", json.dumps(dict(replay), indent=2), "```",
        "", "## Frozen dataset hashes", "```json", json.dumps(frozen, indent=2), "```",
        "", "## Limits", *["- " + item for item in result["limitations"]], "",
        "See feature_files.json for individual file checks, raw_feature_discrepancies.json for mismatches,",
        "and cross_file_duplicates.json for duplicate-key findings. No automatic PASS is inferred from partial raw coverage."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    progress("complete", report=str(output / "REPORT.md"))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / (
        "data_integrity_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    run(args.output)
