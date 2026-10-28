
"""Analyze quality of Kalshi collector feature recordings.

Run:
    python -m src.analyze_collector_quality

Read-only. Does not connect to Kalshi or modify recordings.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports/collector_quality")

MOVEMENT_THRESHOLD_CENTS = 0.5
STALE_PROXY_SECONDS = 30
EXPECTED_SAMPLE_SECONDS = 1.0

REQUIRED_COLUMNS = {
    "timestamp",
    "session_id",
    "ticker",
    "bid",
    "ask",
    "spread",
    "midpoint",
    "bid_depth",
    "ask_depth",
    "imbalance",
}


def analyze_market_recording(group, source_file):
    """Analyze one ticker within one recording session."""

    group = group.sort_values("timestamp").copy()
    group = group.drop_duplicates(
        subset=["timestamp"], keep="last"
    )

    if len(group) < 2:
        return None

    elapsed = group["timestamp"].diff().dt.total_seconds()

    valid_elapsed = elapsed[
        elapsed.notna() & (elapsed > 0)
    ]

    duration_seconds = (
        group["timestamp"].iloc[-1]
        - group["timestamp"].iloc[0]
    ).total_seconds()

    if duration_seconds <= 0:
        return None

    midpoint = group["midpoint"].to_numpy(dtype=float)

    # Difference between consecutive sampled midpoints.
    movement_cents = np.diff(midpoint) * 100
    absolute_changes = np.abs(movement_cents)

    changed = absolute_changes > 1e-6
    meaningful = absolute_changes >= (
        MOVEMENT_THRESHOLD_CENTS - 1e-6
    )

    change_count = int(changed.sum())
    meaningful_count = int(meaningful.sum())

    duration_minutes = duration_seconds / 60.0

    # This measures how long an unchanged midpoint persists.
    # It is NOT the age of the most recent WebSocket update.
    change_start = 0
    longest_unchanged_seconds = 0.0
    unchanged_30s_intervals = 0

    timestamps = group["timestamp"].tolist()

    for i in range(1, len(group)):
        if changed[i - 1]:
            change_start = i
        else:
            elapsed_unchanged = (
                timestamps[i] - timestamps[change_start]
            ).total_seconds()

            longest_unchanged_seconds = max(
                longest_unchanged_seconds,
                elapsed_unchanged,
            )

            if elapsed_unchanged >= STALE_PROXY_SECONDS:
                unchanged_30s_intervals += 1

    spread_cents = (
        group["ask"].to_numpy(dtype=float)
        - group["bid"].to_numpy(dtype=float)
    ) * 100

    spread_valid = np.isfinite(spread_cents)

    source_session = str(group["session_id"].iloc[0])
    ticker = str(group["ticker"].iloc[0])

    return {
        "source_file": source_file,
        "session_id": source_session,
        "ticker": ticker,
        "series_proxy": ticker.split("-")[0],
        "first_sample": timestamps[0],
        "last_sample": timestamps[-1],
        "samples": len(group),
        "duration_minutes": duration_minutes,
        "midpoint_changes": change_count,
        "meaningful_changes": meaningful_count,
        "midpoint_changes_per_min": (
            change_count / duration_minutes
        ),
        "meaningful_changes_per_min": (
            meaningful_count / duration_minutes
        ),
        "pct_intervals_with_midpoint_change": (
            100 * changed.mean()
        ),
        "pct_intervals_with_meaningful_change": (
            100 * meaningful.mean()
        ),
        "median_change_cents_when_changed": (
            float(np.median(absolute_changes[changed]))
            if changed.any() else 0.0
        ),
        "max_change_cents": (
            float(absolute_changes.max())
            if len(absolute_changes) else 0.0
        ),
        "mean_spread_cents": (
            float(np.mean(spread_cents[spread_valid]))
            if spread_valid.any() else np.nan
        ),
        "median_spread_cents": (
            float(np.median(spread_cents[spread_valid]))
            if spread_valid.any() else np.nan
        ),
        "longest_unchanged_midpoint_seconds": (
            longest_unchanged_seconds
        ),
        "pct_samples_after_30s_unchanged_midpoint": (
            100 * unchanged_30s_intervals / max(len(group)-1, 1)
        ),
        "sampling_gaps_over_1_5s": int(
            (valid_elapsed > 1.5).sum()
        ),
        "largest_sampling_gap_seconds": (
            float(valid_elapsed.max())
            if len(valid_elapsed) else 0.0
        ),
        "avg_bid_depth": float(group["bid_depth"].mean()),
        "avg_ask_depth": float(group["ask_depth"].mean()),
    }


def load_file(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Cannot read {path.name}: {exc}")
        return None

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        print(
            f"Skipping {path.name}; missing: {sorted(missing)}"
        )
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    numeric_columns = [
        "bid", "ask", "spread", "midpoint",
        "bid_depth", "ask_depth", "imbalance",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)

    df = df.dropna(
        subset=[
            "timestamp", "session_id", "ticker",
            "bid", "ask", "midpoint",
            "bid_depth", "ask_depth",
        ]
    )

    df = df[
        (df["bid"] >= 0)
        & (df["ask"] <= 1)
        & (df["ask"] >= df["bid"])
        & (df["bid_depth"] >= 0)
        & (df["ask_depth"] >= 0)
    ]

    removed = before - len(df)

    if removed:
        print(f"{path.name}: excluded {removed} invalid rows")

    return df


def create_market_summary(recordings):
    """Combine recording summaries by ticker."""

    rows = []

    for ticker, group in recordings.groupby("ticker"):
        total_minutes = group["duration_minutes"].sum()
        total_changes = group["midpoint_changes"].sum()
        total_meaningful = group["meaningful_changes"].sum()

        rows.append({
            "ticker": ticker,
            "series_proxy": group["series_proxy"].iloc[0],
            "recording_segments": len(group),
            "sessions": group["session_id"].nunique(),
            "samples": int(group["samples"].sum()),
            "total_minutes": total_minutes,
            "midpoint_changes": int(total_changes),
            "meaningful_changes": int(total_meaningful),
            "changes_per_min": (
                total_changes / total_minutes
                if total_minutes else 0
            ),
            "meaningful_changes_per_min": (
                total_meaningful / total_minutes
                if total_minutes else 0
            ),
            "mean_spread_cents": float(
                np.average(
                    group["mean_spread_cents"],
                    weights=group["samples"],
                )
            ),
            "max_change_cents": float(
                group["max_change_cents"].max()
            ),
            "max_unchanged_midpoint_seconds": float(
                group[
                    "longest_unchanged_midpoint_seconds"
                ].max()
            ),
            "sampling_gaps_over_1_5s": int(
                group["sampling_gaps_over_1_5s"].sum()
            ),
        })

    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("features_*.csv"))

    print("\n=== COLLECTOR QUALITY ANALYSIS ===")
    print("Feature files found:", len(files))

    if not files:
        print("No feature recordings found.")
        return

    results = []

    for index, path in enumerate(files, start=1):
        print(f"Analyzing {index}/{len(files)}: {path.name}")

        df = load_file(path)

        if df is None or df.empty:
            continue

        for _, group in df.groupby(
            ["session_id", "ticker"], sort=False
        ):
            result = analyze_market_recording(
                group, path.name
            )

            if result is not None:
                results.append(result)

    if not results:
        print("No usable recording segments.")
        return

    recordings = pd.DataFrame(results)
    markets = create_market_summary(recordings)

    markets = markets.sort_values(
        "meaningful_changes_per_min",
        ascending=False,
    )

    series = markets.groupby("series_proxy").agg(
        markets=("ticker", "nunique"),
        samples=("samples", "sum"),
        total_minutes=("total_minutes", "sum"),
        meaningful_changes=("meaningful_changes", "sum"),
    ).reset_index()

    series["meaningful_changes_per_min"] = (
        series["meaningful_changes"]
        / series["total_minutes"].replace(0, np.nan)
    )

    print("\n=== OVERALL ===")
    print("Recording segments:", len(recordings))
    print("Unique markets:", markets["ticker"].nunique())
    print("Series proxies:", markets["series_proxy"].nunique())
    print("Total samples:", f"{markets['samples'].sum():,}")
    print(
        "Total market-minutes:",
        round(markets["total_minutes"].sum(), 1),
    )

    print(
        "Markets with zero midpoint changes:",
        int((markets["midpoint_changes"] == 0).sum()),
    )

    print(
        "Markets with zero >=0.5 cent changes:",
        int((markets["meaningful_changes"] == 0).sum()),
    )

    print(
        "Markets with >=1 meaningful change/min:",
        int(
            (markets["meaningful_changes_per_min"] >= 1)
            .sum()
        ),
    )

    print(
        "Markets with >=5 meaningful changes/min:",
        int(
            (markets["meaningful_changes_per_min"] >= 5)
            .sum()
        ),
    )

    print("\n=== MOST ACTIVE MARKETS ===")

    columns = [
        "ticker",
        "total_minutes",
        "meaningful_changes",
        "meaningful_changes_per_min",
        "mean_spread_cents",
    ]

    print(
        markets.head(15)[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n=== LEAST ACTIVE MARKETS ===")

    print(
        markets.tail(15)[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n=== SERIES COVERAGE ===")

    print(
        series.sort_values(
            "total_minutes", ascending=False
        ).head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n=== POTENTIAL QUALITY ISSUES ===")

    print(
        "Recording segments with >=30s unchanged midpoint:",
        int((
            recordings[
                "longest_unchanged_midpoint_seconds"
            ] >= 30
        ).sum()),
    )

    print(
        "Segments with sampling gaps >1.5s:",
        int((
            recordings["sampling_gaps_over_1_5s"] > 0
        ).sum()),
    )

    print(
        "Total gaps >1.5s:",
        int(recordings["sampling_gaps_over_1_5s"].sum()),
    )

    recordings.to_csv(
        REPORT_DIR / "recording_segments.csv",
        index=False,
    )

    markets.to_csv(
        REPORT_DIR / "market_quality.csv",
        index=False,
    )

    series.to_csv(
        REPORT_DIR / "series_quality.csv",
        index=False,
    )

    print("\nReports saved in:", REPORT_DIR)

    print(
        "\nNOTE: This report uses sampled midpoint changes."
    )
    print(
        "It cannot measure actual WebSocket delta rates,"
    )
    print(
        "true quote age, exact rotation reasons, or"
    )
    print(
        "verified event-level diversity from CSVs alone."
    )


if __name__ == "__main__":
    main()
