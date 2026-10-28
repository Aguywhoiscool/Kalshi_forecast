
"""Evaluate GRU movement ranking on non-overlapping validation windows.

Run:
    python -m src.analyze_spaced_predictions_v2

Uses validation predictions and metadata only.
Does not access the reserved test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "models/v2_dev/regression/validation_predictions.npz"
)

METADATA_PATH = Path(
    "data/processed/datasets/v2_dev/splits/val_metadata.csv"
)

MOVEMENT_THRESHOLD_CENTS = 0.5


def select_nonoverlapping_windows(metadata):
    """Select chronologically disjoint windows per ticker.

    Each window includes its input history and forecast horizon.
    Apply spacing across sessions for the same ticker.
    """

    selected = []
    last_target_end = {}

    ordered = metadata.sort_values(
        ["input_start", "ticker", "target_end"],
        kind="stable",
    )

    for index, row in ordered.iterrows():
        ticker = row["ticker"]
        start = row["input_start"]
        end = row["target_end"]

        previous_end = last_target_end.get(ticker)

        if previous_end is not None and start <= previous_end:
            continue

        selected.append(index)
        last_target_end[ticker] = end

    return np.asarray(selected, dtype=np.int64)


def report_ranking(actual, predicted, name):
    moving = np.abs(actual) >= MOVEMENT_THRESHOLD_CENTS

    baseline_rate = moving.mean()

    print(f"\n=== {name} ===")
    print(f"Examples: {len(actual):,}")
    print(f"Movement rate: {baseline_rate:.2%}")

    ranked = np.argsort(
        -np.abs(predicted),
        kind="stable",
    )

    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(np.floor(len(actual) * fraction)))
        indices = ranked[:count]

        selected_moving = moving[indices]
        movement_rate = selected_moving.mean()

        lift = (
            movement_rate / baseline_rate
            if baseline_rate > 0
            else float("nan")
        )

        if selected_moving.any():
            moving_actual = actual[indices][selected_moving]
            moving_predicted = predicted[indices][selected_moving]

            direction_accuracy = np.mean(
                np.sign(moving_actual)
                == np.sign(moving_predicted)
            )
        else:
            direction_accuracy = float("nan")

        print(f"\nTop {fraction:.0%}:")
        print(f"  Selected examples: {count:,}")
        print(f"  Actual movements: {selected_moving.sum():,}")
        print(f"  Movement rate: {movement_rate:.2%}")
        print(f"  Lift: {lift:.2f}x")
        print(
            f"  Direction accuracy among movers: "
            f"{direction_accuracy:.2%}"
        )


def main():
    with np.load(PREDICTIONS_PATH) as data:
        actual = data["actual"].reshape(-1)
        predicted = data["predicted"].reshape(-1)

    metadata = pd.read_csv(METADATA_PATH)

    if not (
        len(actual) == len(predicted) == len(metadata)
    ):
        raise ValueError(
            "Prediction and metadata row counts differ."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Non-finite prediction values.")

    # Verify that metadata refers to the same labels.
    if "price_change_cents" not in metadata.columns:
        raise ValueError(
            "Validation metadata must contain price_change_cents."
        )

    metadata_labels = pd.to_numeric(
        metadata["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.allclose(
        actual, metadata_labels, atol=1e-4, rtol=1e-5
    ):
        raise ValueError(
            "Predictions and validation metadata do not align. "
            "Do not rebuild splits between training and analysis."
        )

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column],
            utc=True,
            errors="raise",
        )

    report_ranking(
        actual,
        predicted,
        "ALL VALIDATION WINDOWS",
    )

    indices = select_nonoverlapping_windows(metadata)

    spaced_actual = actual[indices]
    spaced_predicted = predicted[indices]

    print("\n=== OVERLAP REDUCTION ===")
    print(f"Original windows: {len(actual):,}")
    print(f"Non-overlapping windows: {len(indices):,}")
    print(
        f"Remaining fraction: "
        f"{len(indices) / len(actual):.2%}"
    )

    report_ranking(
        spaced_actual,
        spaced_predicted,
        "NON-OVERLAPPING VALIDATION WINDOWS",
    )


if __name__ == "__main__":
    main()
