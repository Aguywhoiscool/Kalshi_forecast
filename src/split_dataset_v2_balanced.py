
"""Build balanced chronological development splits for Version 2.

Input:
    data/processed/datasets/v2_dev/gru_dataset_combined.npz

Outputs:
    data/processed/datasets/v2_dev_balanced/splits/train.npz
    data/processed/datasets/v2_dev_balanced/splits/val.npz
    Corresponding metadata CSV files

Complete recording sessions are assigned chronologically.
Training is downsampled to reduce redundant overlapping windows.
No test partition is created.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "data/processed/datasets/v2_dev/gru_dataset_combined.npz"
)

OUTPUT_DIR = Path(
    "data/processed/datasets/v2_dev_balanced/splits"
)

TRAIN_SESSION_FRACTION = 0.70
TRAIN_SPACING_SECONDS = 10


def thin_training_indices(metadata, spacing_seconds):
    """Keep approximately one training example per ticker every N seconds."""

    selected = []
    last_kept = {}

    ordered = metadata.sort_values(
        ["input_end", "ticker"],
        kind="stable",
    )

    for index, row in ordered.iterrows():
        # Reset spacing at each session boundary.
        key = (row["session_id"], row["ticker"])
        timestamp = row["input_end"]

        previous = last_kept.get(key)

        if previous is not None:
            elapsed = (timestamp - previous).total_seconds()

            if elapsed < spacing_seconds:
                continue

        selected.append(int(index))
        last_kept[key] = timestamp

    return np.asarray(selected, dtype=np.int64)


def print_coverage(name, metadata):
    print(f"\n=== {name.upper()} COVERAGE ===")

    print("Windows:", f"{len(metadata):,}")
    print("Markets:", metadata["ticker"].nunique())
    print("Sessions:", metadata["session_id"].nunique())
    print("Source files:", metadata["source_file"].nunique())

    print("First input:", metadata["input_start"].min())
    print("Last target:", metadata["target_end"].max())

    print("\nWindows by session:")
    print(
        metadata["session_id"]
        .value_counts()
        .to_string()
    )

    print("\nTop markets:")
    print(
        metadata["ticker"]
        .value_counts()
        .head(10)
        .to_string()
    )


def save_split(name, indices, X, y, y_regression,
               feature_names, metadata):

    path = OUTPUT_DIR / f"{name}.npz"
    metadata_path = OUTPUT_DIR / f"{name}_metadata.csv"

    np.savez_compressed(
        path,
        X=X[indices],
        y=y[indices],
        y_regression=y_regression[indices],
        feature_names=feature_names,
    )

    metadata.iloc[indices].to_csv(
        metadata_path,
        index=False,
    )

    print(f"\nSaved {name}: {path}")
    print(f"Saved metadata: {metadata_path}")


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    # Do not silently replace an existing experiment.
    for name in ("train", "val"):
        for suffix in (".npz", "_metadata.csv"):
            destination = OUTPUT_DIR / f"{name}{suffix}"

            if destination.exists():
                raise FileExistsError(
                    f"Output already exists: {destination}"
                )

    print("Loading Version 2 development dataset...")

    with np.load(SOURCE, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        y_regression = data["y_regression"]
        feature_names = data["feature_names"]

    # Load metadata from the separate CSV file.
    metadata_path = SOURCE.with_suffix(".metadata.csv")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    print(f"Metadata loaded: {metadata_path}")
    print(f"Metadata rows: {len(metadata):,}")

    if not (
        len(X) == len(y) == len(y_regression) == len(metadata)
    ):
        raise ValueError("Dataset arrays are not aligned.")

    if X.ndim != 3 or X.shape[1:] != (60, 7):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if not np.isfinite(y_regression).all():
        raise ValueError("Non-finite regression targets.")

    for column in ("input_start", "input_end", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column],
            utc=True,
            errors="raise",
        )

    metadata = metadata.reset_index(drop=True)

    # Establish chronological order using the first input
    # timestamp within each complete recording session.
    session_summary = (
        metadata.groupby("session_id")
        .agg(
            first=("input_start", "min"),
            last=("target_end", "max"),
            windows=("ticker", "size"),
            markets=("ticker", "nunique"),
        )
        .sort_values("first")
    )

    session_count = len(session_summary)

    if session_count < 4:
        raise ValueError(
            "At least four sessions are required."
        )

    train_session_count = int(
        np.floor(session_count * TRAIN_SESSION_FRACTION)
    )

    train_session_count = max(
        1, min(train_session_count, session_count - 2)
    )

    train_sessions = set(
        session_summary.index[:train_session_count]
    )

    val_sessions = set(
        session_summary.index[train_session_count:]
    )

    train_mask = metadata["session_id"].isin(train_sessions)
    val_mask = metadata["session_id"].isin(val_sessions)

    # Verify strict chronological separation.
    train_last = metadata.loc[
        train_mask, "target_end"
    ].max()

    val_first = metadata.loc[
        val_mask, "input_start"
    ].min()

    if not train_last < val_first:
        raise ValueError(
            "Training and validation sessions overlap in time. "
            "A session-count split cannot safely separate them."
        )

    print("\n=== SESSION ALLOCATION ===")
    print("Total sessions:", session_count)
    print("Training sessions:", len(train_sessions))
    print("Validation sessions:", len(val_sessions))
    print("Training ends:", train_last)
    print("Validation begins:", val_first)

    train_metadata = metadata.loc[train_mask]

    train_indices = thin_training_indices(
        train_metadata,
        TRAIN_SPACING_SECONDS,
    )

    val_indices = np.flatnonzero(val_mask.to_numpy())

    if not len(train_indices) or not len(val_indices):
        raise ValueError("Empty training or validation split.")

    # Put saved arrays and metadata into chronological order.
    train_indices = np.asarray(
        sorted(
            train_indices,
            key=lambda i: metadata.at[i, "input_end"],
        ),
        dtype=np.int64,
    )

    val_indices = np.asarray(
        sorted(
            val_indices,
            key=lambda i: metadata.at[i, "input_end"],
        ),
        dtype=np.int64,
    )

    print("\n=== TRAINING DOWNSAMPLING ===")
    print("Original training windows:", f"{train_mask.sum():,}")
    print("Retained training windows:", f"{len(train_indices):,}")
    print("Spacing per market:", TRAIN_SPACING_SECONDS, "seconds")

    print_coverage(
        "train",
        metadata.iloc[train_indices],
    )

    print_coverage(
        "validation",
        metadata.iloc[val_indices],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_split(
        "train",
        train_indices,
        X,
        y,
        y_regression,
        feature_names,
        metadata,
    )

    save_split(
        "val",
        val_indices,
        X,
        y,
        y_regression,
        feature_names,
        metadata,
    )

    print("\n=== BALANCED SPLIT COMPLETE ===")
    print("No test partition created.")
    print("Existing v2_dev experiment remains unchanged.")


if __name__ == "__main__":
    main()
