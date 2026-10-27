
"""Build GRU-ready datasets from Kalshi feature CSVs.

Run: python -m src.dataset_builder
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")
WINDOW = 60
HORIZONS = (1, 5, 10)
MAX_GAP_SECONDS = 1.5

FEATURE_COLUMNS = [
    "bid",
    "ask",
    "spread",
    "midpoint",
    "bid_depth",
    "ask_depth",
    "imbalance",
]


def classify_move(change):
    """0=DOWN, 1=FLAT, 2=UP."""
    if change > 0:
        return 2
    if change < 0:
        return 0
    return 1


def build_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )

    X = []
    y = []
    metadata = []

    max_horizon = max(HORIZONS)

    for (session, ticker), group in df.groupby(
        ["session_id", "ticker"], sort=False
    ):
        group = group.sort_values("timestamp").reset_index(drop=True)

        features = group[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )
        midpoints = group["midpoint"].to_numpy(
            dtype=np.float64
        )

        timestamps = group["timestamp"]
        intervals = timestamps.diff().dt.total_seconds().to_numpy()

        # Each interval must be near one second.
        valid_interval = (
            (intervals >= 0.5)
            & (intervals <= MAX_GAP_SECONDS)
        )
        valid_interval[0] = False

        # Prefix sums efficiently identify gaps in windows.
        bad_count = np.cumsum(~valid_interval)

        for end in range(WINDOW - 1, len(group) - max_horizon):
            start = end - WINDOW + 1
            future_end = end + max_horizon

            # Check continuity from the first input to
            # the last future target.
            bad_intervals = (
                bad_count[future_end] - bad_count[start]
            )

            if bad_intervals:
                continue

            sample = features[start:end + 1]

            if not np.isfinite(sample).all():
                continue

            # Future labels use only future midpoint values.
            labels = [
                classify_move(
                    midpoints[end + horizon] - midpoints[end]
                )
                for horizon in HORIZONS
            ]

            X.append(sample)
            y.append(labels)

            metadata.append({
                "session_id": session,
                "ticker": ticker,
                "input_end": timestamps.iloc[end].isoformat(),
            })

    if not X:
        print("No valid 60-step training windows found.")
        return

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)

    output = DATA_DIR / (
        f"gru_dataset_{Path(csv_path).stem}.npz"
    )

    np.savez_compressed(
        output,
        X=X,
        y=y,
        feature_names=np.asarray(FEATURE_COLUMNS),
        horizons=np.asarray(HORIZONS),
    )

    metadata_path = output.with_suffix(".metadata.csv")
    pd.DataFrame(metadata).to_csv(
        metadata_path, index=False
    )

    print("\n=== GRU DATASET ===")
    print("Source:", csv_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Features:", FEATURE_COLUMNS)
    print("Horizons:", HORIZONS)
    print("Saved:", output)
    print("Metadata:", metadata_path)

    for index, horizon in enumerate(HORIZONS):
        labels, counts = np.unique(
            y[:, index], return_counts=True
        )
        distribution = {
            int(label): int(count)
            for label, count in zip(labels, counts)
        }
        print(
            f"{horizon}s labels "
            f"(0=DOWN, 1=FLAT, 2=UP):",
            distribution,
        )


if __name__ == "__main__":
    files = list(DATA_DIR.glob("features_*.csv"))

    if not files:
        print("No feature CSV files found.")
    else:
        latest = max(
            files,
            key=lambda path: path.stat().st_mtime
        )
        build_dataset(latest)
