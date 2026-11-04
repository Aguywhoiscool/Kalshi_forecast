"""Offline attribution of collection window losses; never stitches production data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v3_research import FEATURES, NS, candidate_ends, valid_rows


def update(run, stage, **extra):
    value = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = run / "status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(run / "status.json")
    with (run / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def timing_attribution(times):
    """Disjoint priority: interval violations, then horizon, then history span."""
    if len(times) < 70:
        return Counter()
    ends = np.arange(59, len(times) - 10)
    starts, future = ends - 59, ends + 10
    gap = np.diff(times) / NS
    bad = np.r_[0, np.cumsum((gap < .5) | (gap > 1.5))]
    interval_ok = bad[future] == bad[starts]
    horizon_ok = np.abs((times[future] - times[ends]) / NS - 10) <= .5
    history_ok = np.abs((times[ends] - times[starts]) / NS - 59) <= 1.5
    return Counter(interval_rejected=int((~interval_ok).sum()),
                   horizon_only_rejected=int((interval_ok & ~horizon_ok).sum()),
                   history_only_rejected=int((interval_ok & horizon_ok & ~history_ok).sum()),
                   timing_valid=int((interval_ok & horizon_ok & history_ok).sum()))


def boundary_saving(previous_length, current_length):
    """Upper bound if two observed segments could be collected uninterrupted."""
    return min(previous_length, 69) + min(current_length, 69) - min(previous_length + current_length, 69)


def interval_causes(times, full_times):
    if len(times) < 70:
        return Counter()
    gaps = np.diff(times) / NS
    skipped = np.diff(np.searchsorted(full_times, times)) > 1
    bad = (gaps < .5) | (gaps > 1.5)
    # Categorize bad intervals by whether filtering removed intervening rows.
    filtered = bad & skipped
    recorded = bad & ~skipped
    ends = np.arange(59, len(times) - 10)
    starts, futures = ends - 59, ends + 10
    a = np.r_[0, np.cumsum(filtered)]
    b = np.r_[0, np.cumsum(recorded)]
    has_a, has_b = a[futures] > a[starts], b[futures] > b[starts]
    return Counter(interval_rejected_wide_filter_only=int((has_a & ~has_b).sum()),
                   interval_rejected_recorded_gap_only=int((has_b & ~has_a).sum()),
                   interval_rejected_both=int((has_a & has_b).sum()))


def crossing_candidates(previous, current):
    """Timestamp-only counterfactual count; does not establish book continuity."""
    previous, current = previous[-69:], current[:69]
    combined = np.r_[previous, current]
    ends = candidate_ends(combined)
    return int(((ends - 59 < len(previous)) & (ends + 10 >= len(previous))).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-run", type=Path, default=Path("artifacts/v3_20261104_overnight"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/collection_efficiency_20261104_overnight"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.snapshot_run / "input_manifest.json").read_text())
    totals, rows, gaps, sessions = Counter(), [], [], []
    previous_groups, seen = {}, set()
    previous_session = None
    for index, item in enumerate(manifest["files"]):
        frame = pd.read_csv(args.snapshot_run / "snapshot" / item["name"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        totals["source_rows"] += len(frame)
        validity = valid_rows(frame)
        totals["invalid_rows"] += int((~validity).sum())
        frame = frame.loc[validity].drop_duplicates(["session_id", "ticker", "timestamp"])
        totals["wide_rows"] += int((frame.spread > .10 + 1e-6).sum())
        # A file normally represents exactly one connection; retain true IDs.
        for session, session_frame in frame.groupby("session_id", sort=False):
            current_groups = {}
            session_first = session_frame.timestamp.min().value
            session_tickers = set(session_frame.ticker)
            prior_tickers = set(previous_groups)
            sessions.append({"session_id": session, "source_file": item["name"],
                "start_ns": session_first, "markets": len(session_tickers),
                "shared_with_previous": len(session_tickers & prior_tickers),
                "added": len(session_tickers - prior_tickers),
                "removed": len(prior_tickers - session_tickers)})
            for ticker, full in session_frame.groupby("ticker", sort=False):
                full = full.sort_values("timestamp")
                full_times = full.timestamp.array.as_unit("ns").asi8
                usable = full.loc[full.spread <= .10 + 1e-6]
                times = usable.timestamp.array.as_unit("ns").asi8
                n = len(times)
                totals["usable_rows"] += n
                totals["boundary_loss"] += min(n, 69)
                totals["potential_after_boundaries"] += max(0, n - 69)
                attribution = timing_attribution(times)
                totals.update(attribution)
                totals.update(interval_causes(times, full_times))
                previous = previous_groups.get(ticker)
                classification = ("retained_adjacent_session" if previous is not None else
                                  "return_after_absence" if ticker in seen else "first_observation")
                saving, actual_cross, gap_seconds = 0, 0, None
                if previous is not None and n and previous["length"]:
                    saving = boundary_saving(previous["run_length"], n)
                    actual_cross = crossing_candidates(previous["tail"], times)
                    gap_seconds = float((times[0] - previous["tail"][-1]) / NS)
                    totals["retained_boundary_saving_upper_bound"] += saving
                    totals["timestamp_compatible_cross_boundary_candidates"] += actual_cross
                totals[classification + "_groups"] += 1
                totals[classification + "_boundary_loss"] += min(n, 69)
                current_groups[ticker] = {"length": n, "tail": times[-69:],
                    "run_length": n + (previous["run_length"] if previous is not None else 0)}
                rows.append({"session_id": session, "ticker": ticker, "source_file": item["name"],
                    "rows": len(full), "usable_rows": n, "boundary_loss": min(n, 69),
                    "classification": classification, "prior_session": previous_session,
                    "gap_from_prior_seconds": gap_seconds, "saving_upper_bound": saving,
                    "timestamp_compatible_cross_boundary_candidates": actual_cross,
                    **dict(attribution)})
                for j in np.flatnonzero(np.diff(times) / NS > 1.5):
                    left, right = times[j:j + 2]
                    skipped = int(np.searchsorted(full_times, right) - np.searchsorted(full_times, left) - 1)
                    phase = float((left - session_first) / NS)
                    gaps.append({"session_id": session, "ticker": ticker,
                        "left_ns": int(left), "right_ns": int(right),
                        "seconds": float((right - left) / NS),
                        "wide_rows_between": skipped,
                        "kind": "wide_quote_filter" if skipped else "recorded_sampling_gap",
                        "seconds_since_first_recorded_sample": phase,
                        "distance_to_60second_multiple": float(min(phase % 60, 60 - phase % 60))})
            previous_groups = current_groups
            seen.update(session_tickers)
            previous_session = session
        if index % 100 == 0:
            update(run, "analyzing_snapshot", files=index + 1, total=len(manifest["files"]))
    groups = pd.DataFrame(rows)
    gap_frame = pd.DataFrame(gaps)
    session_frame = pd.DataFrame(sessions)
    groups.to_csv(run / "market_segments.csv", index=False)
    gap_frame.to_csv(run / "usable_interval_gaps.csv", index=False)
    session_frame.to_csv(run / "session_transitions.csv", index=False)
    counts = dict(totals)
    counts["snapshot_files"] = len(manifest["files"])
    counts["market_segments"] = len(groups)
    counts["sessions"] = len(sessions)
    counts["segments_shorter_than_70_usable_rows"] = int((groups.usable_rows < 70).sum())
    counts["short_segment_usable_rows"] = int(groups.loc[groups.usable_rows < 70, "usable_rows"].sum())
    counts["adjacent_sessions_identical_observed_ticker_sets"] = int(((session_frame.added == 0) & (session_frame.removed == 0)).sum())
    counts["adjacent_sessions_partially_retained"] = int(((session_frame.shared_with_previous > 0) & ((session_frame.added > 0) | (session_frame.removed > 0))).sum())
    for kind, group in gap_frame.groupby("kind"):
        counts[kind + "_intervals"] = len(group)
        counts[kind + "_median_seconds"] = float(group.seconds.median())
        eligible = group.seconds_since_first_recorded_sample >= 50
        counts[kind + "_near_60second_multiple_3s"] = int((eligible & (group.distance_to_60second_multiple <= 3)).sum())
        unique = group.drop_duplicates(["session_id", "left_ns", "right_ns"])
        counts[kind + "_unique_intervals_across_markets"] = len(unique)
        counts[kind + "_unique_near_60second_multiple_3s"] = int(((unique.seconds_since_first_recorded_sample >= 50) & (unique.distance_to_60second_multiple <= 3)).sum())
    kept = groups.loc[groups.classification == "retained_adjacent_session", "gap_from_prior_seconds"].dropna()
    counts["retained_gap_median_seconds"] = float(kept.median())
    counts["retained_gap_p90_seconds"] = float(kept.quantile(.9))
    counts["retained_gaps_at_most_1_5seconds"] = int((kept <= 1.5).sum())
    (run / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    update(run, "analysis_complete", summary=counts)


if __name__ == "__main__":
    main()
