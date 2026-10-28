
"""Analyze timestamp continuity in Kalshi feature CSV sessions."""

import csv
from collections import defaultdict
from pathlib import Path
from datetime import datetime


def analyze_sampling_gaps(filepath):
    timestamps = defaultdict(list)

    with Path(filepath).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timestamps[row["ticker"]].append(
                datetime.fromisoformat(row["timestamp"])
            )

    expected = 0
    observed = 0
    gaps = 0
    largest_gap = 0.0

    for ticker, times in timestamps.items():
        times.sort()

        if not times:
            continue

        # Count unique sampling timestamps for each market.
        unique_times = sorted(set(times))
        observed += len(unique_times)

        duration = (
            unique_times[-1] - unique_times[0]
        ).total_seconds()

        # Each market is expected to be sampled about once
        # per second during its observed lifetime.
        expected += max(
            len(unique_times),
            round(duration) + 1,
        )

        for previous, current in zip(
            unique_times,
            unique_times[1:],
        ):
            delta = (current - previous).total_seconds()

            if delta > 1.5:
                gaps += 1
                largest_gap = max(largest_gap, delta)

    return {
        "expected": expected,
        "observed": observed,
        "coverage": observed / expected if expected else 0,
        "gaps": gaps,
        "largest_gap": largest_gap,
    }


def main():
    files = sorted(
        Path("data/processed").glob("features_*.csv")
    )

    total_expected = 0
    total_observed = 0
    total_gaps = 0
    largest_gap = 0.0

    print("\n=== SAMPLING GAP REPORT ===")

    for filepath in files:
        result = analyze_sampling_gaps(filepath)

        total_expected += result["expected"]
        total_observed += result["observed"]
        total_gaps += result["gaps"]
        largest_gap = max(
            largest_gap,
            result["largest_gap"],
        )

    coverage = (
        total_observed / total_expected
        if total_expected else 0
    )

    print(f"Sessions: {len(files)}")
    print(f"Expected market-seconds: {total_expected:,}")
    print(f"Observed market-seconds: {total_observed:,}")
    print(f"Within-market coverage: {coverage:.2%}")
    print(f"Gaps longer than 1.5 seconds: {total_gaps:,}")
    print(f"Largest gap: {largest_gap:.2f} seconds")


if __name__ == "__main__":
    main()
