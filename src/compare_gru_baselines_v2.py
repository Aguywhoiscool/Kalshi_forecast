
"""Compare Version 2 GRU ranking with simple movement baselines.

Run:
    python -m src.compare_gru_baselines_v2

Validation data only. Never reads the reserved test partition.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze_spaced_predictions_v2 import (
    select_nonoverlapping_windows,
)


ROOT = Path("data/processed/datasets/v2_dev")

PREDICTIONS_PATH = Path(
    "models/v2_dev/regression/validation_predictions.npz"
)

SPLIT_PATH = ROOT / "splits/val.npz"
METADATA_PATH = ROOT / "splits/val_metadata.csv"

MOVEMENT_THRESHOLD = 0.5
MIDPOINT_INDEX = 3

TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def evaluate_ranking(actual, scores, fraction):
    """Evaluate movement detection for the highest-ranked scores."""

    n = max(1, int(np.floor(len(actual) * fraction)))

    indices = np.argsort(
        -scores, kind="stable"
    )[:n]

    moving = np.abs(actual) >= MOVEMENT_THRESHOLD
    selected_moving = moving[indices]

    baseline_rate = moving.mean()
    selected_rate = selected_moving.mean()

    lift = (
        selected_rate / baseline_rate
        if baseline_rate > 0
        else float("nan")
    )

    return {
        "selected": n,
        "movements": int(selected_moving.sum()),
        "movement_rate": selected_rate,
        "lift": lift,
        "indices": indices,
    }


def print_comparison(actual, scores_by_name):
    print("\n=== MOVEMENT RANKING COMPARISON ===")
    print("Non-overlapping examples:", len(actual))
    print(
        "Overall movement rate:",
        f"{np.mean(np.abs(actual) >= MOVEMENT_THRESHOLD):.2%}",
    )

    for fraction in TOP_FRACTIONS:
        print(f"\n=== TOP {fraction:.0%} ===")

        for name, scores in scores_by_name.items():
            result = evaluate_ranking(
                actual, scores, fraction
            )

            print(
                f"{name}: "
                f"{result['movements']}/{result['selected']} moved | "
                f"rate={result['movement_rate']:.2%} | "
                f"lift={result['lift']:.2f}x"
            )


def main():
    with np.load(PREDICTIONS_PATH) as data:
        actual = data["actual"].reshape(-1)
        predicted = data["predicted"].reshape(-1)

    with np.load(SPLIT_PATH) as data:
        X = data["X"]
        split_actual = data["y_regression"].reshape(-1)

    metadata = pd.read_csv(METADATA_PATH)

    if not (
        len(X)
        == len(actual)
        == len(predicted)
        == len(split_actual)
        == len(metadata)
    ):
        raise ValueError(
            "Dataset, predictions and metadata sizes differ."
        )

    if X.ndim != 3 or X.shape[1:] != (60, 7):
        raise ValueError(f"Unexpected input shape: {X.shape}")

    if not np.allclose(
        actual, split_actual, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(
            "Prediction labels do not match validation split."
        )

    metadata_actual = pd.to_numeric(
        metadata["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.allclose(
        actual, metadata_actual, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(
            "Prediction labels do not match validation metadata."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
        and np.isfinite(X).all()
    ):
        raise ValueError("Non-finite data encountered.")

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column], utc=True, errors="raise"
        )

    # Use the same non-overlapping selection as the prior analysis.
    indices = select_nonoverlapping_windows(metadata)

    actual = actual[indices]
    predicted = predicted[indices]
    X = X[indices]

    # Raw midpoint in dollars; convert differences into cents.
    midpoints = X[:, :, MIDPOINT_INDEX].astype(np.float64)

    # Sum absolute one-second movements within the input history.
    volatility = (
        np.abs(np.diff(midpoints, axis=1)).sum(axis=1)
        * 100.0
    )

    # Net change over the input history.
    momentum = (
        (midpoints[:, -1] - midpoints[:, 0])
        * 100.0
    )

    scores = {
        "GRU": np.abs(predicted),
        "Recent volatility": volatility,
        "Recent momentum": np.abs(momentum),
    }

    print("\n=== DATA CHECK ===")
    print("Selected windows:", len(indices))
    print("Unique markets:", metadata.iloc[indices]["ticker"].nunique())
    print(
        "Actual movements >= 0.5c:",
        int((np.abs(actual) >= MOVEMENT_THRESHOLD).sum())
    )

    print_comparison(actual, scores)

    # Directional diagnostic on exactly the same top-GRU subset.
    # This is conditional on future movement, NOT trade win rate.
    top_gru = evaluate_ranking(
        actual, np.abs(predicted), 0.10
    )["indices"]

    moving_top = (
        np.abs(actual[top_gru]) >= MOVEMENT_THRESHOLD
    )

    if moving_top.any():
        gru_direction = np.mean(
            np.sign(predicted[top_gru][moving_top])
            == np.sign(actual[top_gru][moving_top])
        )

        momentum_direction = np.mean(
            np.sign(momentum[top_gru][moving_top])
            == np.sign(actual[top_gru][moving_top])
        )

        print("\n=== DIRECTION ON TOP 10% GRU SELECTIONS ===")
        print("Moving examples:", int(moving_top.sum()))
        print("GRU direction accuracy:", f"{gru_direction:.2%}")
        print(
            "Momentum direction accuracy:",
            f"{momentum_direction:.2%}"
        )
    else:
        print("\nNo moving examples in the top 10% GRU selection.")


if __name__ == "__main__":
    main()
