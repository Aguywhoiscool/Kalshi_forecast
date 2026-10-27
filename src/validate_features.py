
"""Validate recorded Kalshi feature CSV files.

Run: python -m src.validate_features
"""

from pathlib import Path
import pandas as pd


def validate_features():
    files = list(Path("data/processed").glob("features_*.csv"))

    if not files:
        print("No feature CSV files found.")
        return

    latest = max(files, key=lambda p: p.stat().st_mtime)
    print(f"Validating: {latest}")

    df = pd.read_csv(latest)

    if df.empty:
        print("Feature file contains no observations.")
        return

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    df = df.sort_values(["ticker", "timestamp"])

    print("\n=== DATASET SUMMARY ===")
    print("Total observations:", len(df))
    print("Markets:", df["ticker"].nunique())
    print("Sessions:", df["session_id"].nunique())

    print("\n=== DATA QUALITY ===")

    invalid_prices = (
        (df["bid"] < 0)
        | (df["ask"] > 1)
        | (df["bid"] > df["ask"])
    ).sum()

    negative_depth = (
        (df["bid_depth"] < 0)
        | (df["ask_depth"] < 0)
    ).sum()

    missing_values = df.isna().sum().sum()

    duplicates = df.duplicated(
        subset=["session_id", "ticker", "timestamp"]
    ).sum()

    print("Invalid prices:", invalid_prices)
    print("Negative depths:", negative_depth)
    print("Missing values:", missing_values)
    print("Duplicate observations:", duplicates)

    print("\n=== MARKET ACTIVITY ===")

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("timestamp")

        time_gaps = group["timestamp"].diff().dt.total_seconds()

        large_gaps = (time_gaps > 1.5).sum()

        midpoint_changes = (
            group["midpoint"].diff().abs() > 0.00001
        ).sum()

        print(f"\n{ticker}")
        print("  Observations:", len(group))
        print("  Gaps >1.5 seconds:", large_gaps)
        print("  Midpoint changes:", midpoint_changes)

    print("\nValidation complete.")


if __name__ == "__main__":
    validate_features()
