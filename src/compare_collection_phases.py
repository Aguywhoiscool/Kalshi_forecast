
"""Compare Kalshi collection quality across development phases."""

from collections import Counter
from datetime import datetime
from pathlib import Path

from src.collection_quality_report import analyze_file


PHASES = [
    ("Original collector", None, "2026-10-28_17-02-29"),
    ("Shadow mode", "2026-10-28_17-02-29", "2026-10-28_17-21-28"),
    ("Activity-aware rotation", "2026-10-28_17-21-28", "2026-10-28_17-40-13"),
    ("Continuous observer", "2026-10-28_17-40-13", "2026-10-28_18-09-04"),
    ("Spread-aware", "2026-10-28_18-09-04", None),
]


def classify_phase(filename):
    stamp = filename.removeprefix("features_").removesuffix(".csv")

    for name, start, end in PHASES:
        if (start is None or stamp >= start) and (
            end is None or stamp < end
        ):
            return name

    return "Unknown"


def main():
    files = sorted(Path("data/processed").glob("features_*.csv"))

    summaries = {
        name: {
            "files": 0,
            "totals": Counter(),
            "duration_seconds": 0.0,
            "tickers": set(),
        }
        for name, _, _ in PHASES
    }

    for filepath in files:
        phase = classify_phase(filepath.name)
        result = analyze_file(filepath)
        summary = summaries[phase]

        summary["files"] += 1
        summary["duration_seconds"] += result["duration_seconds"]

        for key in (
            "rows", "usable_rows", "wide_rows",
            "invalid_rows", "midpoint_changes",
            "meaningful_changes", "meaningful_usable",
        ):
            summary["totals"][key] += result[key]

    print("\n=== COLLECTION PHASE COMPARISON ===")

    for name, summary in summaries.items():
        totals = summary["totals"]

        hours = summary["duration_seconds"] / 3600

        usable = totals["usable_rows"]
        rows = totals["rows"]
        meaningful = totals["meaningful_usable"]

        print(f"\n{name}")
        print(f"  Sessions: {summary['files']}")
        print(f"  Recorded hours: {hours:.2f}")
        print(f"  Rows: {rows:,}")
        print(
            f"  Usable spread: "
            f"{100 * usable / rows if rows else 0:.1f}%"
        )
        print(
            f"  Meaningful usable changes: {meaningful:,}"
        )
        print(
            f"  Meaningful changes / hour: "
            f"{meaningful / hours if hours else 0:.1f}"
        )
        print(
            f"  Rows / hour: "
            f"{rows / hours if hours else 0:.0f}"
        )


if __name__ == "__main__":
    main()
