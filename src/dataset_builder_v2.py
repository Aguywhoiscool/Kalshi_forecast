
"""Build one combined GRU dataset from all Kalshi feature recordings.

Run:
    python -m src.dataset_builder
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


# Development dataset. Does not modify frozen Version 2.
SOURCE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/datasets/v2_dev")

OUTPUT_FILE = OUTPUT_DIR / "gru_dataset_combined.npz"

# 60 observations of history, predicting 10 seconds ahead.
WINDOW = 60
HORIZONS = (10,)

# Consecutive observations should be approximately 1 second apart.
MIN_GAP_SECONDS = 0.5
MAX_GAP_SECONDS = 1.5

# Maximum acceptable bid-ask spread: 10 cents = $0.10.
MAX_SPREAD = 0.10

# Actual elapsed forecast time must be approximately 10 seconds.
TARGET_TIME_TOLERANCE_SECONDS = 0.5


FEATURE_COLUMNS = [
    "bid",
    "ask",
    "spread",
    "midpoint",
    "bid_depth",
    "ask_depth",
    "imbalance",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "session_id",
    "ticker",
    *FEATURE_COLUMNS,
]


def classify_move(change):
    """0 = DOWN, 1 = FLAT, 2 = UP."""
    if change > 0:
        return 2
    if change < 0:
        return 0
    return 1



def build_combined_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclude recently modified files that may still
    # be receiving observations from the collector.
    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=10
    )

    files = sorted(
        path
        for path in SOURCE_DIR.glob("features_*.csv")
        if datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        ) < cutoff
    )

    print(
        f"Found {len(files)} completed feature recordings."
    )

    if not files:
        print("No feature CSV files found.")
        return

    print(f"Found {len(files)} feature files.\n")

    X_parts = []
    y_parts = []
    regression_parts = []
    metadata_parts = []

    total_rows = 0
    skipped_files = 0
    processed_files = 0
    max_horizon = max(HORIZONS)

    for file_number, filepath in enumerate(files, 1):
        print(
            f"[{file_number}/{len(files)}] "
            f"Processing {filepath.name}"
        )

        try:
            df = pd.read_csv(filepath)
        except (OSError, pd.errors.ParserError) as error:
            print(f"  Skipped unreadable file: {error}")
            skipped_files += 1
            continue

        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            print(f"  Skipped: missing columns {sorted(missing)}")
            skipped_files += 1
            continue

        if df.empty:
            print("  Skipped: empty recording")
            skipped_files += 1
            continue

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        for column in FEATURE_COLUMNS:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # Remove invalid and duplicate observations.
        df = df.dropna(subset=REQUIRED_COLUMNS)

        df = df[
            (df["bid"] >= 0)
            & (df["bid"] < df["ask"])
            & (df["ask"] <= 1)
            & (df["bid_depth"] >= 0)
            & (df["ask_depth"] >= 0)
            & (df["spread"] > 0)
            & (df["spread"] <= MAX_SPREAD + 1e-6)
            & (
                (df["spread"] - (df["ask"] - df["bid"])).abs()
                <= 1e-6
            )
            & (
                (
                    df["midpoint"]
                    - (df["bid"] + df["ask"]) / 2
                ).abs()
                <= 1e-6
            )
        ]

        df = df.drop_duplicates(
            subset=["session_id", "ticker", "timestamp"],
            keep="first",
        )

        total_rows += len(df)
        file_examples = 0

        # Keep each market and connection session separate.
        for (session, ticker), group in df.groupby(
            ["session_id", "ticker"],
            sort=False,
        ):
            group = group.sort_values(
                "timestamp"
            ).reset_index(drop=True)

            if len(group) < WINDOW + max_horizon:
                continue

            timestamps = group["timestamp"]

            intervals = (
                timestamps.diff()
                .dt.total_seconds()
                .to_numpy()
            )

            valid_interval = (
                (intervals >= MIN_GAP_SECONDS)
                & (intervals <= MAX_GAP_SECONDS)
            )
            valid_interval[0] = False

            bad_count = np.cumsum(~valid_interval)

            features = group[FEATURE_COLUMNS].to_numpy(
                dtype=np.float32
            )
            midpoints = group["midpoint"].to_numpy(
                dtype=np.float64
            )

            market_X = []
            market_y = []
            market_regression = []
            market_metadata = []

            for end in range(
                WINDOW - 1,
                len(group) - max_horizon
            ):
                start = end - WINDOW + 1
                future_end = end + max_horizon

                # Reject windows spanning a timing gap.
                                # Reject windows spanning a timing gap.
                if bad_count[future_end] != bad_count[start]:
                    continue

                # Verify that the target is approximately
                # 10 elapsed seconds after the final input.
                actual_horizon = (
                    timestamps.iloc[future_end]
                    - timestamps.iloc[end]
                ).total_seconds()

                if abs(
                    actual_horizon - HORIZONS[0]
                ) > TARGET_TIME_TOLERANCE_SECONDS:
                    continue

                # Verify that the input history spans
                # approximately 59 elapsed seconds.
                actual_history = (
                    timestamps.iloc[end]
                    - timestamps.iloc[start]
                ).total_seconds()

                if abs(actual_history - (WINDOW - 1)) > 1.5:
                    continue

                sample = features[start:end + 1]

                if not np.isfinite(sample).all():
                    continue

                labels = [
                    classify_move(
                        midpoints[end + horizon]
                        - midpoints[end]
                    )
                    for horizon in HORIZONS
                ]
                price_change_cents = (
                    midpoints[future_end] - midpoints[end]
                ) * 100

                market_X.append(sample)
                market_y.append(labels)
                market_regression.append(price_change_cents)

                market_metadata.append({
                    "source_file": filepath.name,
                    "session_id": session,
                    "ticker": ticker,
                    "input_start": timestamps.iloc[start].isoformat(),
                    "input_end": timestamps.iloc[end].isoformat(),
                    "target_end": timestamps.iloc[
                        future_end
                    ].isoformat(),
                    "price_change_cents": price_change_cents,
                })

            if market_X:
                X_parts.append(np.stack(market_X))
                y_parts.append(
                    np.asarray(market_y, dtype=np.int8)
                )
                regression_parts.append(
                    np.asarray(market_regression, dtype=np.float32)
                )
                metadata_parts.append(
                    pd.DataFrame(market_metadata)
                )

                file_examples += len(market_X)

        processed_files += 1

        print(
            f"  Valid rows: {len(df):,} | "
            f"Training windows: {file_examples:,}"
        )

    if not X_parts:
        print("\nNo valid training windows found.")
        return

    print("\nCombining datasets...")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    y_regression = np.concatenate(
        regression_parts, axis=0
    )

    metadata = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        y_regression=y_regression,
        feature_names=np.asarray(FEATURE_COLUMNS),
        horizons=np.asarray(HORIZONS),
    )

    metadata_file = OUTPUT_FILE.with_suffix(
        ".metadata.csv"
    )
    metadata.to_csv(metadata_file, index=False)

    print("\n" + "=" * 55)
    print("COMBINED GRU DATASET")
    print("=" * 55)

    print("Feature files found:", len(files))
    print("Files processed:", processed_files)
    print("Files skipped:", skipped_files)
    print("Valid source rows:", f"{total_rows:,}")
    print("Training windows:", f"{len(X):,}")
    print("Markets:", metadata["ticker"].nunique())
    print("Sessions:", metadata["session_id"].nunique())

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    for index, horizon in enumerate(HORIZONS):
        labels, counts = np.unique(
            y[:, index],
            return_counts=True,
        )

        print(f"\n{horizon}-SECOND LABEL DISTRIBUTION")

        for label, count in zip(labels, counts):
            name = {0: "DOWN", 1: "FLAT", 2: "UP"}[
                int(label)
            ]

            percentage = 100 * count / len(y)

            print(
                f"  {name}: "
                f"{count:,} ({percentage:.2f}%)"
            )

    print("\nSaved dataset:", OUTPUT_FILE)
    print("Saved metadata:", metadata_file)


if __name__ == "__main__":
    build_combined_dataset()
