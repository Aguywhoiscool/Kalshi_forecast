
"""Build regression targets for existing frozen chronological splits.

Run:
    python -m src.build_regression_dataset
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/processed")
SPLITS = ROOT / "datasets" / "v1" / "splits"
OUTPUT = ROOT / "regression_v1"

# The midpoint is the fourth feature:
# bid, ask, spread, midpoint, bid_depth, ask_depth, imbalance
MIDPOINT_INDEX = 3


def build_split(name):
    input_file = SPLITS / f"{name}.npz"
    metadata_file = SPLITS / f"{name}_metadata.csv"

    with np.load(input_file) as data:
        X = data["X"].copy()

    metadata = pd.read_csv(metadata_file)

    if len(X) != len(metadata):
        raise ValueError(f"{name}: X and metadata do not match.")

    if "target_end" not in metadata.columns:
        raise ValueError(f"{name}: missing target_end metadata.")

    metadata["target_time"] = pd.to_datetime(
        metadata["target_end"], utc=True
    )

    metadata["input_time"] = pd.to_datetime(
        metadata["input_end"], utc=True
    )

    # Current midpoint is the final input timestep.
    metadata["current_midpoint"] = X[:, -1, MIDPOINT_INDEX]

    future_midpoints = np.full(len(metadata), np.nan)

    for filename, positions in metadata.groupby("source_file").groups.items():
        path = ROOT / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing original feature file: {path}")

        source = pd.read_csv(
            path,
            usecols=["timestamp", "session_id", "ticker", "midpoint"],
        )

        source["target_time"] = pd.to_datetime(
            source["timestamp"], utc=True
        )

        # Only use the same recording session and market.
        source = source[
            ["session_id", "ticker", "target_time", "midpoint"]
        ].drop_duplicates(
            ["session_id", "ticker", "target_time"],
            keep="last",
        )

        requested = metadata.loc[
            positions, ["session_id", "ticker", "target_time"]
        ].copy()

        requested["_row"] = requested.index

        matched = requested.merge(
            source,
            on=["session_id", "ticker", "target_time"],
            how="left",
            validate="many_to_one",
        )

        future_midpoints[matched["_row"].to_numpy()] = (
            matched["midpoint"].to_numpy()
        )

    good = np.isfinite(future_midpoints)

    if not good.all():
        raise ValueError(
            f"{name}: {int((~good).sum())} targets could not "
            "be matched. Do not silently discard them."
        )

    y = (
        future_midpoints - metadata["current_midpoint"].to_numpy()
    ) * 100

    OUTPUT.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT / f"{name}.npz",
        X=X.astype(np.float32),
        y=y.astype(np.float32),
    )

    metadata["future_midpoint"] = future_midpoints
    metadata["price_change_cents"] = y

    metadata.to_csv(
        OUTPUT / f"{name}_metadata.csv",
        index=False,
    )

    print(
        f"{name}: {len(X):,} sequences, "
        f"mean change {np.mean(y):+.4f} cents, "
        f"nonzero changes {np.mean(y != 0):.2%}"
    )


def main():
    for name in ("train", "val", "test"):
        build_split(name)

    print("\nRegression dataset created:", OUTPUT)


if __name__ == "__main__":
    main()
