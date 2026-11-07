"""Resumable, rate-limited schedule audit of frozen development markets.

Never trains on retrospective schedules or reads reserved outcomes.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests

from src.backfill_close_metadata import SCHEDULE_FIELDS, annotate, fetch_schedule
from src.v3_research import digest, overlaps, save_json


BIN_NAMES = ("after_close", "under_10m", "10_to_30m", "30_to_60m", "over_60m", "unknown")


def classify(seconds):
    values = np.asarray(seconds, float)
    labels = np.full(values.shape, "unknown", dtype=object)
    labels[np.isfinite(values) & (values < 0)] = "after_close"
    labels[(values >= 0) & (values < 600)] = "under_10m"
    labels[(values >= 600) & (values < 1800)] = "10_to_30m"
    labels[(values >= 1800) & (values < 3600)] = "30_to_60m"
    labels[values >= 3600] = "over_60m"
    return labels


def load_cache(path):
    """Recover a final interrupted JSONL write; reject corruption elsewhere."""
    if not path.exists():
        return {}
    result, valid_bytes = {}, 0
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError("Schedule cache corruption")
            break
        result[record["ticker"]] = record
        valid_bytes += len(line)
    if valid_bytes != len(raw):
        with path.open("r+b") as stream:
            stream.truncate(valid_bytes)
    elif raw and not raw.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return result


class RateLimit:
    def __init__(self, per_second):
        self.interval, self.next_start, self.lock = 1/per_second, 0., threading.Lock()

    def wait(self):
        with self.lock:
            target = max(time.monotonic(), self.next_start)
            self.next_start = target+self.interval
        time.sleep(max(0, target-time.monotonic()))


def inventory(source, output):
    original = pd.read_parquet(source / "windows.parquet")
    allowed = set(original.loc[original.split.isin(["train", "val"]), "session_id"])
    protected = json.loads((source / "protected_v2_period.json").read_text())
    parts, files = [], []
    manifest = json.loads((source / "input_manifest.json").read_text())
    for number, item in enumerate(manifest["files"]):
        session = item["name"].removeprefix("features_").removesuffix(".csv")
        if session not in allowed or session in protected["sessions"]:
            continue
        path = source / "snapshot" / item["name"]
        if digest(path) != item["sha256"]:
            raise ValueError("Source hash mismatch: "+path.name)
        frame = pd.read_csv(path, usecols=["timestamp", "ticker"])
        stamps = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        keep = stamps.notna() & ~overlaps(stamps.array.as_unit("ns").asi8, stamps.array.as_unit("ns").asi8, protected)
        frame = frame.loc[keep].copy()
        frame["timestamp"] = stamps.loc[keep]
        parts.append(frame.groupby("ticker").timestamp.agg(["min", "max", "size"]).reset_index())
        files.append(item["name"])
        if number % 300 == 0:
            print(json.dumps({"stage": "inventory", "files": len(files)}), flush=True)
    result = pd.concat(parts).groupby("ticker").agg(first=("min", "min"), last=("max", "max"), rows=("size", "sum"))
    result = result.sort_values("last", ascending=False).reset_index()
    result.to_csv(output / "inventory.csv", index=False)
    return result, files, protected


def download(tickers, output, seed, progress, workers=4, rate=6, offline=False):
    path = output / "schedules.jsonl"
    records = load_cache(path)
    # Reuse known schedules, not previous failures. Never import extra fields.
    if seed:
        allowed = set(SCHEDULE_FIELDS) | {"retrieved_at", "endpoint", "asof_verified", "warning"}
        for ticker, record in load_cache(seed).items():
            if ticker in tickers and ticker not in records and record.get("close_time"):
                records[ticker] = {k: v for k, v in record.items() if k in allowed}
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records.values():
            stream.write(json.dumps(record)+"\n")
    temporary.replace(path)
    limiter, local = RateLimit(rate), threading.local()
    from src.market_ranker import BASE_URL

    class LimitedSession:
        def __init__(self):
            self.session = requests.Session()
        def get(self, *args, **kwargs):
            limiter.wait()
            return self.session.get(*args, **kwargs)

    def fetch(ticker):
        if not hasattr(local, "session"):
            local.session = LimitedSession()
        for attempt in range(3):
            try:
                return fetch_schedule(local.session, ticker, BASE_URL)
            except requests.RequestException as error:
                retry = error.response is None or error.response.status_code in (429, 500, 502, 503, 504)
                if attempt == 2 or not retry:
                    return {"ticker": ticker, "asof_verified": False, "error": type(error).__name__,
                            "retrieved_at": datetime.now(timezone.utc).isoformat()}
                time.sleep(min(15, 2**attempt))

    missing = [t for t in tickers if not records.get(t, {}).get("close_time")]
    if offline and missing:
        raise ValueError("Offline audit requires cached schedules for every market")
    progress("metadata", completed=len(tickers)-len(missing), total=len(tickers), reused=len(tickers)-len(missing))
    with path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        iterator, pending = iter(missing), {}
        def fill():
            while len(pending) < workers:
                ticker = next(iterator, None)
                if ticker is None:
                    break
                pending[pool.submit(fetch, ticker)] = ticker
        fill()
        completed = len(tickers)-len(missing)
        while pending:
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not done:
                progress("metadata", completed=completed, total=len(tickers), waiting=True)
            for future in done:
                pending.pop(future)
                record = future.result()
                records[record["ticker"]] = record
                stream.write(json.dumps(record)+"\n")
                stream.flush()
                completed += 1
                if completed % 20 == 0 or completed == len(tickers):
                    progress("metadata", completed=completed, total=len(tickers),
                             found=sum(bool(r.get("close_time")) for r in records.values()))
            fill()
    return {t: records[t] for t in tickers}


def summarize_rows(source, files, protected, schedules, output, progress):
    parts = []
    for number, name in enumerate(files):
        frame = pd.read_csv(source / "snapshot" / name, usecols=["timestamp", "ticker", "session_id"])
        stamp = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        frame = frame.loc[stamp.notna() & ~overlaps(stamp.array.as_unit("ns").asi8, stamp.array.as_unit("ns").asi8, protected)]
        frame = annotate(frame, schedules)
        frame["time_bin"] = classify(frame.retrospective_seconds_to_close)
        grouped = frame.groupby(["ticker", "session_id", "time_bin"]).size().rename("rows").reset_index()
        parts.append(grouped)
        if number % 300 == 0:
            progress("counting_rows", files=number+1, total=len(files))
    detail = pd.concat(parts, ignore_index=True)
    detail.to_parquet(output / "session_coverage.parquet", index=False)
    summary = detail.groupby("time_bin").agg(rows=("rows", "sum"), sessions=("session_id", "nunique"), markets=("ticker", "nunique"))
    summary.to_csv(output / "row_coverage.csv")
    return summary.reset_index().to_dict("records")


def summarize_windows(run, source, schedules, output):
    protocol = json.loads((run / "protocol.json").read_text())
    if Path(protocol["source"]).resolve() != source.resolve():
        raise ValueError("Window source does not match audited snapshot")
    audit = json.loads((run / "audit.json").read_text())
    if audit["status"] != "passed" or digest(source / "input_manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("Window audit/source provenance mismatch")
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    if digest(run / "sample_metadata.parquet") != protocol["sample_metadata_sha256"]:
        raise ValueError("Window metadata changed")
    meta["timestamp"] = pd.to_datetime(meta.input_end, utc=True)
    meta = annotate(meta, schedules)
    meta["time_bin"] = classify(meta.retrospective_seconds_to_close)
    known = np.load(run / "future_valid.npy", mmap_mode="r")
    future = np.load(run / "future_quotes.npy", mmap_mode="r")
    history = np.load(run / "history.npy", mmap_mode="r")
    for horizon in (30, 60):
        meta[f"known_entry_exit_{horizon}"] = known[:, 1] & known[:, horizon]
        mid = (future[:, horizon, 0]+future[:, horizon, 1])/2
        meta[f"moving_{horizon}"] = known[:, horizon] & (np.abs(mid-history[:, -1, 3])*100 >= .5-1e-9)
    for (_, _), group in meta.groupby(["session_id", "ticker"]):
        if (np.diff(group.input_end) < 120*10**9).any():
            raise ValueError("Expected spaced, independent-in-time windows")
    meta.to_parquet(output / "window_schedule_diagnostics.parquet", index=False)
    records = []
    for (split, time_bin), group in meta.groupby(["split", "time_bin"]):
        records.append({"split": split, "time_bin": time_bin, "windows": len(group),
            "sessions": group.session_id.nunique(), "markets": group.ticker.nunique(),
            **{f"known_entry_exit_{h}": int(group[f"known_entry_exit_{h}"].sum()) for h in (30, 60)},
            **{f"moving_{h}": int(group[f"moving_{h}"].sum()) for h in (30, 60)}})
    pd.DataFrame(records).to_csv(output / "window_coverage.csv", index=False)
    return records


def summarize_predictions(run, output):
    """Post-hoc subgroup diagnostics; preserve the original global trade policy."""
    meta = pd.read_parquet(output / "window_schedule_diagnostics.parquet")
    keys = ["session_id", "ticker", "input_end"]
    lookup = pd.MultiIndex.from_frame(meta[keys])
    records = []
    models = ("mean", "ridge", "sgd", "decision_tree", "random_forest", "gradient_boosting", "gru", "monte_carlo", "no_trade")
    for horizon in (30, 60):
        val = pd.read_parquet(run / f"validation_{horizon}.parquet")
        indices = lookup.get_indexer(pd.MultiIndex.from_frame(val[keys]))
        if (indices < 0).any() or not meta.iloc[indices].split.eq("val").all():
            raise ValueError("Unexpected validation identities")
        bins = meta.iloc[indices].time_bin.to_numpy()
        with np.load(run / f"targets_{horizon}.npz") as target:
            lower = target["lower"][indices]
        for model in models:
            prediction = None
            if model != "no_trade":
                with np.load(run / f"{model}_{horizon}_predictions.npz") as saved:
                    prediction = saved["validation"]
                if prediction.shape != lower.shape:
                    raise ValueError("Saved validation prediction shape mismatch")
            trades = pd.read_parquet(run / "trades" / f"{model}_{horizon}_margin1.parquet")
            if len(trades):
                selected = lookup.get_indexer(pd.MultiIndex.from_frame(trades[keys]))
                if (selected < 0).any() or not meta.iloc[selected].split.eq("val").all():
                    raise ValueError("Unexpected trade identities")
                trades["time_bin"] = meta.iloc[selected].time_bin.to_numpy()
            else:
                trades["time_bin"] = pd.Series(dtype="str")
            for time_bin in BIN_NAMES:
                keep = bins == time_bin
                if not keep.any():
                    continue
                cohort = trades.loc[trades.time_bin == time_bin]
                active = cohort.loc[cohort.status != "blocked_uncertain_or_pending"]
                closed = active.loc[active.status == "closed"]
                records.append({"model": model, "horizon": horizon, "time_bin": time_bin,
                    "validation_windows": int(keep.sum()), "attempts": len(active), "closed": len(closed),
                    "unknown_entries": int(active.status.eq("unknown_entry").sum()),
                    "unresolved_exits": int(active.status.isin(["missing_exit_quote", "insufficient_exit_depth"]).sum()),
                    "mae_conservative_target_cents": float(np.abs(prediction[keep]-lower[keep]).mean()) if prediction is not None else None,
                    "net_lower_total_cents": float(active.net_lower_cents.sum()),
                    "net_upper_total_cents": float(active.net_upper_cents.sum())})
    pd.DataFrame(records).to_csv(output / "model_subgroup_diagnostics.csv", index=False)
    return records


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    root = Path.cwd().resolve()
    if root not in output.parents or root / "data" in output.parents:
        raise ValueError("Use an artifact directory outside data/")
    output.mkdir(parents=True, exist_ok=True)
    config = {"source": str(source), "source_manifest_sha256": digest(source / "input_manifest.json"),
              "window_run": str(args.windows.resolve()), "asof_verified": False}
    config_path = output / "protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration differs")
    save_json(config_path, config)
    def progress(stage, **extra):
        state = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
        save_json(output / "status.json", state)
        print(json.dumps(state), flush=True)
    try:
        progress("inventory")
        markets, files, protected = inventory(source, output)
        schedules = download(markets.ticker.tolist(), output, args.seed, progress,
                             args.workers, args.requests_per_second, args.offline)
        rows = summarize_rows(source, files, protected, schedules, output, progress)
        progress("counting_windows")
        windows = summarize_windows(args.windows, source, schedules, output)
        diagnostics = summarize_predictions(args.windows, output)
        updated = pd.to_datetime(markets.ticker.map({t: r.get("updated_time") for t, r in schedules.items()}),
                                 utc=True, errors="coerce", format="ISO8601")
        last_observed = pd.to_datetime(markets["last"], utc=True)
        schedule_quality = {
            "markets_allowing_early_close": sum(r.get("can_close_early") is True for r in schedules.values()),
            "market_updated_after_last_observation": int((updated > last_observed).sum()),
            "market_update_timestamp_unknown": int(updated.isna().sum()),
            "note": "A later market update does not prove its close time changed; neither field verifies historical schedule knowledge"}
        result = {**config, "markets": len(markets), "files": len(files), "rows": int(markets.rows.sum()),
            "schedules_found": sum(bool(r.get("close_time")) for r in schedules.values()),
            "errors": sum(bool(r.get("error")) for r in schedules.values()),
            "schedule_metadata_quality": schedule_quality,
            "row_coverage": rows, "window_coverage": windows, "model_subgroup_diagnostics": diagnostics,
            "training_allowed": False, "verified_causal_schedule_windows": 0,
            "limitations": ["Schedules fetched later are not proven known at observation time",
                "120-second spacing reduces temporal overlap; markets/events can remain correlated",
                "Movement counts use observed midpoint changes, not executable returns",
                "Model subgroup diagnostics are post-hoc and use retrospectively identified time bins; not a controlled training experiment",
                "No collector restart or model training performed by this audit"]}
        save_json(output / "summary.json", result)
        lines = ["# Near-close coverage audit", "", f"Frozen development data: {len(markets):,} markets, {len(files):,} files, {int(markets.rows.sum()):,} rows.",
            f"Schedules recovered: {result['schedules_found']:,}; failures: {result['errors']:,}.",
            "", "Retrospective schedule metadata is not as-of verified. No causal training inputs approved.", "",
            "| Split | Time remaining | Spaced windows | Sessions | Markets | Known entry/30s exit | Known entry/60s exit |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for row in windows:
            lines.append(f"| {row['split']} | {row['time_bin']} | {row['windows']} | {row['sessions']} | {row['markets']} | {row['known_entry_exit_30']} | {row['known_entry_exit_60']} |")
        lines += ["", "## Existing validation forecasts by retrospective time bin", "",
            "No models were retrained. Original global trade blocking and the fixed 1c margin / 4c cost policy are preserved. These are descriptive subgroups, not evidence of a validated near-close strategy.", "",
            "| Model | Horizon | Time remaining | Validation windows | Attempts | Net lower cents | Net upper cents |",
            "|---|---:|---|---:|---:|---:|---:|"]
        for row in diagnostics:
            if row["attempts"] or row["model"] in ("gru", "no_trade"):
                lines.append(f"| {row['model']} | {row['horizon']}s | {row['time_bin']} | {row['validation_windows']} | {row['attempts']} | {row['net_lower_total_cents']:+.3f} | {row['net_upper_total_cents']:+.3f} |")
        lines += ["", "## Next action", "Deploy the prepared point-in-time metadata recording on the collection computer, then collect separate future near-close validation sessions. Preserve existing data; use retrospective coverage for descriptive exploration only.",
                  "", *["- "+line for line in result["limitations"]]]
        (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        progress("complete", report=str(output / "REPORT.md"), schedules_found=result["schedules_found"])
    except BaseException as error:
        progress("failed", error=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/swing_source_all_20261106"))
    parser.add_argument("--windows", type=Path, default=Path("artifacts/swing_returns_all_20261106"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=Path, default=Path("artifacts/near_close_backfill_20261106/schedules.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=6)
    parser.add_argument("--offline", action="store_true", help="Require a complete metadata cache; never fetch")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 0 < args.requests_per_second <= 6:
        parser.error("Use 1-4 workers and at most six requests per second")
    run(args)
