
"""Analyze quality of recorded Kalshi feature CSV files."""

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


MAX_SPREAD = 0.10
MEANINGFUL_MOVE = 0.005


def analyze_file(filepath):
    filepath = Path(filepath)

    rows = 0
    usable_rows = 0
    wide_rows = 0
    invalid_rows = 0
    tickers = set()
    timestamps = set()
    midpoint_changes = 0
    meaningful_changes = 0
    meaningful_usable = 0
    last_midpoint = {}
    per_ticker = defaultdict(Counter)
    first_timestamp = None
    last_timestamp = None

    with filepath.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp", "ticker",
            "bid", "ask", "spread", "midpoint",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing required columns in {filepath}"
            )

        for row in reader:
            rows += 1
            ticker = row["ticker"]
            tickers.add(ticker)
            per_ticker[ticker]["rows"] += 1

            timestamp = datetime.fromisoformat(
                row["timestamp"]
            )
            timestamps.add(timestamp)

            if (
                first_timestamp is None
                or timestamp < first_timestamp
            ):
                first_timestamp = timestamp

            if (
                last_timestamp is None
                or timestamp > last_timestamp
            ):
                last_timestamp = timestamp

            try:
                bid = float(row["bid"])
                ask = float(row["ask"])
                spread = float(row["spread"])
                midpoint = float(row["midpoint"])
            except (ValueError, TypeError):
                invalid_rows += 1
                continue

            import math

            if not all(
                math.isfinite(v)
                for v in (bid, ask, spread, midpoint)
            ):
                invalid_rows += 1
                continue

            if not (
                0 <= bid < ask <= 1
                and spread > 0
                and abs(spread - (ask - bid)) < 0.000001
                and abs(midpoint - (bid + ask) / 2) < 0.000001
            ):
                invalid_rows += 1
                continue

            usable = spread <= MAX_SPREAD + 0.000001

            if usable:
                usable_rows += 1
                per_ticker[ticker]["usable"] += 1
            else:
                wide_rows += 1
                per_ticker[ticker]["wide"] += 1

           
            previous = last_midpoint.get(ticker)

            if previous is not None:
                previous_time, previous_mid, previous_usable = previous

                elapsed = (
                    timestamp - previous_time
                ).total_seconds()

                # Compare only consecutive, approximately
                # one-second observations from the same market.
                if 0 < elapsed <= 1.5:
                    movement = abs(midpoint - previous_mid)

                    if movement > 0.000001:
                        midpoint_changes += 1

                    if movement >= MEANINGFUL_MOVE - 0.000001:
                        meaningful_changes += 1

                        # BOTH endpoints must have acceptable spreads.
                        if previous_usable and usable:
                            meaningful_usable += 1

            last_midpoint[ticker] = (
                timestamp,
                midpoint,
                usable,
            )


    duration = (
        (last_timestamp - first_timestamp).total_seconds()
        if first_timestamp and last_timestamp
        else 0
    )

    return {
        "file": str(filepath),
        "rows": rows,
        "usable_rows": usable_rows,
        "wide_rows": wide_rows,
        "invalid_rows": invalid_rows,
        "tickers": len(tickers),
        "sampling_timestamps": len(timestamps),
        "duration_seconds": duration,
        "midpoint_changes": midpoint_changes,
        "meaningful_changes": meaningful_changes,
        "meaningful_usable": meaningful_usable,
        "per_ticker": dict(per_ticker),
    }


def main():
    files = sorted(
        Path("data/processed").glob("features_*.csv")
    )

    if not files:
        print("No feature CSV files found.")
        return

    totals = Counter()

    print("\n=== COLLECTION QUALITY REPORT ===")

    for filepath in files:
        result = analyze_file(filepath)

        for key in (
            "rows", "usable_rows", "wide_rows",
            "invalid_rows", "midpoint_changes",
            "meaningful_changes", "meaningful_usable",
        ):
            totals[key] += result[key]

        print(
            f"\n{filepath.name}"
            f"\n  Rows: {result['rows']:,}"
            f"\n  Usable (spread <= 10c): "
            f"{result['usable_rows']:,}"
            f"\n  Wide-spread rows: "
            f"{result['wide_rows']:,}"
            f"\n  Invalid rows: "
            f"{result['invalid_rows']:,}"
            f"\n  Meaningful changes: "
            f"{result['meaningful_changes']:,}"
            f"\n  Meaningful changes at usable spreads: "
            f"{result['meaningful_usable']:,}"
        )

    rows = totals["rows"]
    usable_pct = (
        100 * totals["usable_rows"] / rows
        if rows else 0
    )

    print("\n=== TOTALS ===")
    print(f"Files: {len(files)}")
    print(f"Recorded rows: {rows:,}")
    print(
        f"Usable spread rows: "
        f"{totals['usable_rows']:,} ({usable_pct:.1f}%)"
    )
    print(f"Wide-spread rows: {totals['wide_rows']:,}")
    print(f"Invalid rows: {totals['invalid_rows']:,}")
    print(
        f"Meaningful midpoint changes: "
        f"{totals['meaningful_changes']:,}"
    )
    print(
        f"Meaningful changes at usable spreads: "
        f"{totals['meaningful_usable']:,}"
    )


if __name__ == "__main__":
    main()
