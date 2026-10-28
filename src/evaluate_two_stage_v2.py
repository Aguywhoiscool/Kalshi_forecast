
"""Evaluate volatility selection followed by GRU direction.

Uses only Version 2 development validation data.
Does not retrain or access the reserved test set.

Run:
    python -m src.evaluate_two_stage_v2
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

MIDPOINT_INDEX = 3
MOVEMENT_THRESHOLD_CENTS = 0.5
TOP_FRACTIONS = (0.05, 0.10, 0.20)


def evaluate_direction(actual, predicted_sign):
    """Score directions, including flat outcomes.

    Predictions may be -1 (DOWN), 0 (ABSTAIN), or +1 (UP).
    """

    actual_sign = np.where(
        actual >= MOVEMENT_THRESHOLD_CENTS,
        1,
        np.where(
            actual <= -MOVEMENT_THRESHOLD_CENTS,
            -1,
            0,
        ),
    )

    predicted_sign = np.asarray(predicted_sign)

    correct = int(np.sum(
        (predicted_sign != 0)
        & (predicted_sign == actual_sign)
    ))

    wrong = int(np.sum(
        (predicted_sign != 0)
        & (actual_sign != 0)
        & (predicted_sign != actual_sign)
    ))

    flat = int(np.sum(
        (predicted_sign != 0)
        & (actual_sign == 0)
    ))

    abstained = int(np.sum(predicted_sign == 0))

    total = len(actual)
    directional_calls = total - abstained

    assert correct + wrong + flat + abstained == total

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "flat": flat,
        "abstained": abstained,
        "directional_calls": directional_calls,
        "correct_rate_all": correct / total if total else 0,
        "correct_rate_calls": (
            correct / directional_calls
            if directional_calls else float("nan")
        ),
        "accuracy_when_moving": (
            correct / (correct + wrong)
            if correct + wrong else float("nan")
        ),
    }


def print_direction_report(name, actual, signs):
    result = evaluate_direction(actual, signs)

    print(f"\n{name}")
    print("  Selected windows:", result["total"])
    print("  Directional calls:", result["directional_calls"])
    print("  Correct:", result["correct"])
    print("  Wrong direction:", result["wrong"])
    print("  Flat outcomes:", result["flat"])
    print("  Abstentions:", result["abstained"])

    print(
        "  Correct / all selected:",
        f"{result['correct_rate_all']:.2%}",
    )
    print(
        "  Correct / directional calls:",
        f"{result['correct_rate_calls']:.2%}",
    )
    print(
        "  Direction accuracy when moving:",
        f"{result['accuracy_when_moving']:.2%}",
    )


def main():
    with np.load(PREDICTIONS_PATH) as data:
        actual = data["actual"].reshape(-1)
        gru_prediction = data["predicted"].reshape(-1)

    with np.load(SPLIT_PATH) as data:
        X = data["X"]
        split_actual = data["y_regression"].reshape(-1)

    metadata = pd.read_csv(METADATA_PATH)

    if not (
        len(actual)
        == len(gru_prediction)
        == len(X)
        == len(split_actual)
        == len(metadata)
    ):
        raise ValueError("Prediction/split/metadata length mismatch.")

    if X.ndim != 3 or X.shape[1:] != (60, 7):
        raise ValueError(f"Unexpected features shape: {X.shape}")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(gru_prediction).all()
        and np.isfinite(X).all()
    ):
        raise ValueError("Non-finite data encountered.")

    if not np.allclose(
        actual, split_actual, atol=1e-4, rtol=1e-5
    ):
        raise ValueError("Saved predictions do not match split labels.")

    metadata_actual = pd.to_numeric(
        metadata["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.allclose(
        actual, metadata_actual, atol=1e-4, rtol=1e-5
    ):
        raise ValueError(
            "Saved predictions do not match metadata labels."
        )

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column], utc=True, errors="raise"
        )

    # Select chronologically non-overlapping examples
    # before ranking or evaluating model signals.
    indices = select_nonoverlapping_windows(metadata)

    actual = actual[indices]
    gru_prediction = gru_prediction[indices]
    X = X[indices]
    metadata = metadata.iloc[indices].reset_index(drop=True)

    if not len(actual):
        raise ValueError("No non-overlapping examples found.")

    midpoints = X[:, :, MIDPOINT_INDEX].astype(np.float64)

    # Stage 1: activity/volatility scoring, using only
    # the 60 input observations.
    volatility = (
        np.abs(np.diff(midpoints, axis=1)).sum(axis=1)
        * 100.0
    )

    # Simple directional baseline based on past movement.
    momentum = (
        (midpoints[:, -1] - midpoints[:, 0]) * 100.0
    )

    # Stage 2: GRU direction.
    gru_sign = np.sign(gru_prediction).astype(int)
    momentum_sign = np.sign(momentum).astype(int)

    moving = np.abs(actual) >= MOVEMENT_THRESHOLD_CENTS
    overall_movement_rate = moving.mean()

    print("\n=== TWO-STAGE VALIDATION EXPERIMENT ===")
    print("Non-overlapping examples:", len(actual))
    print("Unique markets:", metadata["ticker"].nunique())
    print("Unique sessions:", metadata["session_id"].nunique())
    print("Actual meaningful movements:", int(moving.sum()))
    print("Overall movement rate:", f"{overall_movement_rate:.2%}")

    # Stable ranking handles equal-volatility ties consistently.
    ranked = np.argsort(
        -volatility,
        kind="stable",
    )

    for fraction in TOP_FRACTIONS:
        n = max(1, int(np.floor(len(actual) * fraction)))
        selected = ranked[:n]

        selected_actual = actual[selected]
        selected_moving = moving[selected]

        movement_rate = selected_moving.mean()
        lift = (
            movement_rate / overall_movement_rate
            if overall_movement_rate else float("nan")
        )

        print(f"\n{'=' * 50}")
        print(f"TOP {fraction:.0%} BY RECENT VOLATILITY")
        print(f"{'=' * 50}")
        print("Selected examples:", n)
        print("Actual movements:", int(selected_moving.sum()))
        print("Actual movement rate:", f"{movement_rate:.2%}")
        print("Movement lift:", f"{lift:.2f}x")

        print_direction_report(
            "GRU DIRECTION",
            selected_actual,
            gru_sign[selected],
        )

        print_direction_report(
            "MOMENTUM DIRECTION",
            selected_actual,
            momentum_sign[selected],
        )

        print_direction_report(
            "ALWAYS UP BASELINE",
            selected_actual,
            np.ones(n, dtype=int),
        )

        print_direction_report(
            "ALWAYS DOWN BASELINE",
            selected_actual,
            -np.ones(n, dtype=int),
        )

        # Show whether these selections are concentrated
        # in one market.
        selected_metadata = metadata.iloc[selected]

        print("\nSelected markets:")
        print(
            selected_metadata["ticker"]
            .value_counts()
            .to_string()
        )

    print("\nReserved test split was not accessed.")


if __name__ == "__main__":
    main()
