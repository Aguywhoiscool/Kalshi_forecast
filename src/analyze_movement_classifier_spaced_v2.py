
"""Compare movement classifier and volatility on spaced validation windows."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze_spaced_predictions_v2 import (
    select_nonoverlapping_windows,
)

ROOT = Path("data/processed/datasets/v2_expanded_dev")

PREDICTIONS = Path(
    "models/v2_expanded_dev/movement_classifier_v2/"
    "validation_predictions.npz"
)

METADATA = ROOT / "splits/val_metadata.csv"


def report(name, scores, actual):
    ranked = np.argsort(-scores, kind="stable")
    baseline = actual.mean()

    print(f"\n=== {name} ===")

    for fraction in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(len(actual) * fraction))
        selected = ranked[:n]

        movements = int(actual[selected].sum())
        rate = movements / n
        lift = rate / baseline if baseline else float("nan")

        print(
            f"Top {fraction:.0%}: "
            f"{movements}/{n} moved | "
            f"rate={rate:.2%} | lift={lift:.2f}x"
        )


def main():
    with np.load(PREDICTIONS) as data:
        actual = data["actual"].reshape(-1)
        probability = data["probability"].reshape(-1)
        volatility = data["volatility"].reshape(-1)
        price_change = data["price_change_cents"].reshape(-1)

    metadata = pd.read_csv(METADATA)

    if not (
        len(actual)
        == len(probability)
        == len(volatility)
        == len(price_change)
        == len(metadata)
    ):
        raise ValueError("Prediction and metadata lengths differ.")

    metadata_changes = pd.to_numeric(
        metadata["price_change_cents"],
        errors="raise",
    ).to_numpy()

    if not np.allclose(
        price_change, metadata_changes, atol=1e-4, rtol=1e-5
    ):
        raise ValueError("Predictions and metadata do not align.")

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column],
            utc=True,
            errors="raise",
        )

    indices = select_nonoverlapping_windows(metadata)

    actual = actual[indices]
    probability = probability[indices]
    volatility = volatility[indices]

    print("\n=== SPACED MOVEMENT CLASSIFIER COMPARISON ===")
    print("Original windows:", len(metadata))
    print("Non-overlapping windows:", len(indices))
    print("Markets:", metadata.iloc[indices]["ticker"].nunique())
    print("Sessions:", metadata.iloc[indices]["session_id"].nunique())
    print("Actual movements:", int(actual.sum()))
    print("Movement base rate:", f"{actual.mean():.2%}")

    report("GRU CLASSIFIER", probability, actual)
    report("RECENT VOLATILITY", volatility, actual)


if __name__ == "__main__":
    main()
