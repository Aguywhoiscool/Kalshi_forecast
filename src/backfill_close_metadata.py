"""Retrospective schedule audit; fetched close times are NOT as-of verified."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import quote

import pandas as pd
import requests

from src.v3_research import overlaps, save_json


SCHEDULE_FIELDS = ("ticker", "close_time", "open_time", "updated_time", "can_close_early",
                   "expected_expiration_time", "expiration_time", "latest_expiration_time")


def fetch_schedule(session, ticker, base):
    for prefix in ("markets", "historical/markets"):
        endpoint = f"{base}/{prefix}/{quote(ticker, safe='')}"
        response = session.get(endpoint, timeout=10)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        market = response.json()["market"]
        return {**{key: market.get(key) for key in SCHEDULE_FIELDS},
            "ticker": ticker, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint, "asof_verified": False,
            "warning": "Retrospective schedule may differ from the schedule known at observation; diagnostics only"}
    return {"ticker": ticker, "asof_verified": False, "error": "not_found"}


def annotate(frame, schedules):
    """Keep retrospective schedule fields separate from causal model features."""
    result = frame.copy()
    close = result.ticker.map({key: value.get("close_time") for key, value in schedules.items()})
    close = pd.to_datetime(close, utc=True, errors="coerce", format="ISO8601")
    stamp = pd.to_datetime(result.timestamp, utc=True, errors="coerce", format="ISO8601")
    result["retrospective_seconds_to_close"] = (close-stamp).dt.total_seconds()
    result["asof_verified"] = False
    return result


def run(source, output, limit):
    root, output = Path.cwd().resolve(), output.resolve()
    if output.exists() or root not in output.parents or root / "data" in output.parents:
        raise ValueError("Use a new artifact directory outside data/")
    output.mkdir(parents=True)
    meta = pd.read_parquet(source / "windows.parquet")
    allowed = set(meta.loc[meta.split.isin(["train", "val"]), "session_id"])
    protected = json.loads((source / "protected_v2_period.json").read_text())
    manifests = json.loads((source / "input_manifest.json").read_text())["files"]
    parts, eligible = [], []
    for item in manifests:
        session = item["name"].removeprefix("features_").removesuffix(".csv")
        if session not in allowed or session in protected["sessions"]:
            continue
        path = source / "snapshot" / item["name"]
        frame = pd.read_csv(path, usecols=["timestamp", "ticker"])
        stamps = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        frame = frame.loc[stamps.notna() & ~overlaps(stamps.array.as_unit("ns").asi8,
                          stamps.array.as_unit("ns").asi8, protected)].copy()
        frame["timestamp"] = stamps.loc[frame.index]
        summary = frame.groupby("ticker").timestamp.agg(["min", "max", "size"]).reset_index()
        parts.append(summary)
        eligible.append(path)
    inventory = pd.concat(parts).groupby("ticker").agg(first=("min", "min"), last=("max", "max"), rows=("size", "sum"))
    inventory = inventory.sort_values("last", ascending=False).reset_index()
    inventory.to_csv(output / "inventory.csv", index=False)
    print(json.dumps({"stage": "inventory", "markets": len(inventory), "rows": int(inventory.rows.sum())}), flush=True)
    schedules = {}
    from src.market_ranker import BASE_URL
    with requests.Session() as session, (output / "schedules.jsonl").open("w", encoding="utf-8") as stream:
        for ticker in inventory.ticker.head(limit):
            try:
                record = fetch_schedule(session, ticker, BASE_URL)
            except requests.RequestException as error:
                record = {"ticker": ticker, "asof_verified": False, "error": type(error).__name__}
            schedules[ticker] = record
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            print(json.dumps({"stage": "metadata", "completed": len(schedules), "limit": limit,
                              "found": bool(record.get("close_time"))}), flush=True)
            time.sleep(.2)
    counts = Counter()
    for path in eligible:
        frame = pd.read_csv(path, usecols=["timestamp", "ticker"])
        frame = frame.loc[frame.ticker.isin(schedules)]
        if frame.empty:
            continue
        stamps = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        frame = frame.loc[stamps.notna() & ~overlaps(stamps.array.as_unit("ns").asi8,
                          stamps.array.as_unit("ns").asi8, protected)]
        seconds = annotate(frame, schedules).retrospective_seconds_to_close
        counts["matched_rows"] += int(seconds.notna().sum())
        counts["rows_10_to_60_minutes"] += int(((seconds >= 600) & (seconds < 3600)).sum())
        counts["rows_0_to_10_minutes"] += int(((seconds >= 0) & (seconds < 600)).sum())
        counts["rows_after_reported_close"] += int((seconds < 0).sum())
    result = {"utc": datetime.now(timezone.utc).isoformat(), "source": str(source.resolve()),
        "development_markets": len(inventory), "development_rows": int(inventory.rows.sum()),
        "requested_metadata": len(schedules), "schedules_found": sum(bool(r.get("close_time")) for r in schedules.values()),
        **counts, "scope": "Most recently observed markets first; bounded metadata sample, not exchange-wide coverage",
        "asof_verified": False, "training_allowed": False,
        "conclusion": "Existing book rows are reusable. Retrospective close times support coverage diagnostics, not verified causal features. Missing book history cannot be recovered from schedules."}
    save_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/swing_source_all_20261106"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-markets", type=int, default=40)
    args = parser.parse_args()
    if args.max_markets < 1:
        parser.error("max-markets must be positive")
    run(args.source, args.output, args.max_markets)
