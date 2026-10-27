
"""Chronologically split the Kalshi GRU dataset.

Run: python -m src.split_dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Input uses 60 observations; the furthest label
# extends another 10 observations.
PURGE_ROWS = 70


def main():
    files = list(DATA_DIR.glob("gru_dataset_*.npz"))

    if not files:
        print("No GRU dataset found.")
        return

    source = max(files, key=lambda p: p.stat().st_mtime)
    metadata_path = source.with_suffix(".metadata.csv")

    with np.load(source) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        feature_names = dataset["feature_names"]
        horizons = dataset["horizons"]

    metadata = pd.read_csv(metadata_path)
    metadata["input_end"] = pd.to_datetime(
        metadata["input_end"], utc=True
    )

    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("Dataset and metadata lengths differ.")

    splits = {"train": [], "val": [], "test": []}

    # Split chronologically within each market/session,
    # never across unrelated market streams.
    for (_, _), group in metadata.groupby(
        ["session_id", "ticker"], sort=False
    ):
        ordered = group.sort_values("input_end")
        indices = ordered.index.to_numpy()
        n = len(indices)

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        # Purge examples around split boundaries to
        # prevent overlapping inputs and future labels.
        partitions = {
            "train": indices[:max(0, train_end - PURGE_ROWS)],
            "val": indices[
                min(n, train_end + PURGE_ROWS):
                max(train_end + PURGE_ROWS, val_end - PURGE_ROWS)
            ],
            "test": indices[min(n, val_end + PURGE_ROWS):],
        }

        for name, partition in partitions.items():
            splits[name].extend(partition.tolist())

    output_dir = DATA_DIR / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, indices in splits.items():
        indices = np.asarray(indices, dtype=np.int64)

        output = output_dir / f"{name}.npz"

        np.savez_compressed(
            output,
            X=X[indices],
            y=y[indices],
            feature_names=feature_names,
            horizons=horizons,
        )

        metadata.iloc[indices].to_csv(
            output_dir / f"{name}_metadata.csv",
            index=False
        )

        print(
            f"{name.upper()}: "
            f"{len(indices):,} examples | "
            f"X shape: {X[indices].shape}"
        )

    print("\nChronological split complete.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
