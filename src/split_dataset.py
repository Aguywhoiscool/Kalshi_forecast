
"""Globally chronological train/validation/test split.

Run: python -m src.split_dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")
SOURCE = DATA_DIR / "gru_dataset_combined.npz"
METADATA = SOURCE.with_suffix(".metadata.csv")
OUTPUT_DIR = DATA_DIR / "splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def main():
    if not SOURCE.exists() or not METADATA.exists():
        raise FileNotFoundError(
            "Run python -m src.dataset_builder first."
        )

    metadata = pd.read_csv(METADATA)

    for column in ("input_start", "input_end", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column], utc=True
        )

    # Establish shared chronological boundaries across
    # every market and every recording session.
    times = metadata["input_end"].sort_values().reset_index(drop=True)

    train_boundary = times.iloc[int(len(times) * TRAIN_RATIO)]
    val_boundary = times.iloc[
        int(len(times) * (TRAIN_RATIO + VAL_RATIO))
    ]

    # Prevent overlapping inputs/labels across partitions.
    masks = {
        "train": (
            metadata["target_end"] < train_boundary
        ),
        "val": (
            (metadata["input_start"] >= train_boundary)
            & (metadata["target_end"] < val_boundary)
        ),
        "test": (
            metadata["input_start"] >= val_boundary
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Training boundary:", train_boundary)
    print("Validation boundary:", val_boundary)

    with np.load(SOURCE) as dataset:
        X = dataset["X"]
        y = dataset["y"]
        feature_names = dataset["feature_names"]
        horizons = dataset["horizons"]

    if not (len(X) == len(y) == len(metadata)):
        raise ValueError("Dataset and metadata lengths differ.")

    for name, mask in masks.items():
        indices = np.flatnonzero(mask.to_numpy())

        output_path = OUTPUT_DIR / f"{name}.npz"

        np.savez_compressed(
            output_path,
            X=X[indices],
            y=y[indices],
            feature_names=feature_names,
            horizons=horizons,
        )

        metadata.iloc[indices].to_csv(
            OUTPUT_DIR / f"{name}_metadata.csv",
            index=False,
        )

        print(
            f"{name.upper()}: {len(indices):,} examples | "
            f"X shape: {X[indices].shape}"
        )

    print("\nGlobal chronological split complete.")


if __name__ == "__main__":
    main()
