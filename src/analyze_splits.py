
"""Analyze market activity and class balance across dataset splits.

Run: python -m src.analyze_splits
"""

from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_DIR = Path("data/processed/splits")


def analyze_split(name):
    with np.load(SPLIT_DIR / f"{name}.npz") as data:
        y = data["y"]

    metadata = pd.read_csv(
        SPLIT_DIR / f"{name}_metadata.csv"
    )

    if len(metadata) != len(y):
        raise ValueError("Metadata and label counts differ.")

    metadata["move_1s"] = y[:, 0] != 1
    metadata["move_5s"] = y[:, 1] != 1
    metadata["move_10s"] = y[:, 2] != 1

    report = (
        metadata.groupby("ticker")
        .agg(
            samples=("ticker", "size"),
            moves_1s=("move_1s", "sum"),
            moves_5s=("move_5s", "sum"),
            moves_10s=("move_10s", "sum"),
        )
        .reset_index()
    )

    for horizon in (1, 5, 10):
        report[f"move_rate_{horizon}s"] = (
            100 * report[f"moves_{horizon}s"]
            / report["samples"]
        )

    report = report.sort_values(
        "samples", ascending=False
    )

    print(f"\n=== {name.upper()} ===")
    print("Examples:", len(metadata))
    print("Markets:", metadata["ticker"].nunique())

    print("\nTop markets by number of examples:")
    print(report.head(15).to_string(index=False))

    inactive = report[
        report["moves_10s"] == 0
    ]

    print("\nMarkets with zero 10-second midpoint movements:",
          len(inactive))
    print("Examples from those markets:",
          int(inactive["samples"].sum()))

    output = SPLIT_DIR / f"{name}_activity_report.csv"
    report.to_csv(output, index=False)

    print("Saved:", output)


def main():
    for name in ("train", "val", "test"):
        analyze_split(name)


if __name__ == "__main__":
    main()
