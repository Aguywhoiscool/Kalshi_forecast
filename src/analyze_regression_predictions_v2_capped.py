
"""Analyze whether the Version 2 GRU ranks future movements."""

from pathlib import Path

import numpy as np


PREDICTIONS = Path(
    "models/v2_dev_balanced_capped/regression/validation_predictions.npz"
)

MOVEMENT_THRESHOLD = 0.5


def main():
    with np.load(PREDICTIONS) as data:
        actual = data["actual"].reshape(-1)
        predicted = data["predicted"].reshape(-1)

    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Invalid prediction arrays.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Non-finite prediction values.")

    moving = np.abs(actual) >= MOVEMENT_THRESHOLD

    baseline_rate = moving.mean()

    print("\n=== MOVEMENT RANKING ANALYSIS ===")
    print("Validation examples:", len(actual))
    print("Movement threshold:", MOVEMENT_THRESHOLD, "cents")
    print("Overall movement rate:", f"{baseline_rate:.2%}")

    ranked = np.argsort(-np.abs(predicted), kind="stable")

    for fraction in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(len(actual) * fraction))
        indices = ranked[:n]

        movement_rate = moving[indices].mean()

        lift = (
            movement_rate / baseline_rate
            if baseline_rate > 0
            else float("nan")
        )

        direction = (
            np.sign(predicted[indices])
            == np.sign(actual[indices])
        )

        moving_indices = moving[indices]

        direction_accuracy = (
            direction[moving_indices].mean()
            if moving_indices.any()
            else float("nan")
        )

        print(
            f"\nTop {fraction:.0%} by predicted magnitude:"
            f"\n  Examples: {n:,}"
            f"\n  Actual movement rate: {movement_rate:.2%}"
            f"\n  Lift over baseline: {lift:.2f}x"
            f"\n  Direction accuracy on moving examples: "
            f"{direction_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
