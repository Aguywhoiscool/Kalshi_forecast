
"""Compare GRU, volatility, and combined movement rankings.

Uses existing expanded Version 2 validation predictions only.

Run:
    python -m src.analyze_combined_movement_v2
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.analyze_spaced_predictions_v2 import (
    select_nonoverlapping_windows,
)


ROOT = Path("data/processed/datasets/v2_expanded_dev")

PREDICTIONS_PATH = Path(
    "models/v2_expanded_dev/movement_classifier_v2/"
    "validation_predictions.npz"
)

METADATA_PATH = ROOT / "splits/val_metadata.csv"

MOVEMENT_THRESHOLD = 0.5
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def percentile_scores(values):
    """Convert scores to percentile ranks from 0 to 1."""

    ranks = rankdata(values, method="average")

    if len(values) == 1:
        return np.ones(1)

    return (ranks - 1) / (len(values) - 1)


def evaluate(actual, scores, fraction):
    n = max(1, int(len(actual) * fraction))

    selected = np.argsort(
        -scores,
        kind="stable",
    )[:n]

    moving = actual[selected]
    count = int(moving.sum())

    rate = count / n
    baseline = actual.mean()

    lift = (
        rate / baseline
        if baseline > 0
        else float("nan")
    )

    return {
        "indices": selected,
        "selected": n,
        "movements": count,
        "rate": rate,
        "lift": lift,
    }


def main():
    with np.load(PREDICTIONS_PATH) as data:
        actual = data["actual"].reshape(-1)
        probability = data["probability"].reshape(-1)
        volatility = data["volatility"].reshape(-1)
        changes = data["price_change_cents"].reshape(-1)

    metadata = pd.read_csv(METADATA_PATH)

    if not (
        len(actual)
        == len(probability)
        == len(volatility)
        == len(changes)
        == len(metadata)
    ):
        raise ValueError("Prediction and metadata lengths differ.")

    if not all(
        np.isfinite(array).all()
        for array in (actual, probability, volatility, changes)
    ):
        raise ValueError("Non-finite values found.")

    metadata_changes = pd.to_numeric(
        metadata["price_change_cents"],
        errors="raise",
    ).to_numpy()

    if not np.allclose(
        changes, metadata_changes,
        atol=1e-4,
        rtol=1e-5,
    ):
        raise ValueError(
            "Saved predictions and validation metadata do not align."
        )

    expected_movement = (
        np.abs(changes) >= MOVEMENT_THRESHOLD
    ).astype(int)

    if not np.array_equal(
        actual.astype(int),
        expected_movement,
    ):
        raise ValueError("Movement labels do not match price changes.")

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column],
            utc=True,
            errors="raise",
        )

    # Match the previous non-overlapping evaluation.
    indices = select_nonoverlapping_windows(metadata)

    actual = actual[indices].astype(int)
    probability = probability[indices]
    volatility = volatility[indices]
    metadata = metadata.iloc[indices].reset_index(drop=True)

    gru_rank = percentile_scores(probability)
    volatility_rank = percentile_scores(volatility)

    # Predefined equal-weight combination.
    combined_score = (
        0.5 * gru_rank
        + 0.5 * volatility_rank
    )

    methods = {
        "GRU classifier": probability,
        "Recent volatility": volatility,
        "50/50 combined": combined_score,
    }

    print("\n=== COMBINED MOVEMENT EXPERIMENT ===")
    print("Non-overlapping windows:", len(actual))
    print("Unique markets:", metadata["ticker"].nunique())
    print("Unique sessions:", metadata["session_id"].nunique())
    print("Actual movements:", int(actual.sum()))
    print("Movement base rate:", f"{actual.mean():.2%}")

    print("\n=== MOVEMENT RANKING COMPARISON ===")

    for fraction in TOP_FRACTIONS:
        print(f"\nTOP {fraction:.0%}")

        for name, scores in methods.items():
            result = evaluate(
                actual,
                scores,
                fraction,
            )

            print(
                f"{name}: "
                f"{result['movements']}/{result['selected']} moved | "
                f"rate={result['rate']:.2%} | "
                f"lift={result['lift']:.2f}x"
            )

    print("\n=== TOP 10% SELECTION OVERLAP ===")

    selected_sets = {}

    for name, scores in methods.items():
        result = evaluate(
            actual,
            scores,
            0.10,
        )

        selected_sets[name] = set(
            result["indices"].tolist()
        )

    for first, second in (
        ("GRU classifier", "Recent volatility"),
        ("GRU classifier", "50/50 combined"),
        ("Recent volatility", "50/50 combined"),
    ):
        shared = len(
            selected_sets[first] & selected_sets[second]
        )

        print(f"{first} vs {second}: {shared} shared windows")

    # Examine concentration in the strongest combined selections.
    top_combined = evaluate(
        actual,
        combined_score,
        0.10,
    )["indices"]

    selected_metadata = metadata.iloc[top_combined].copy()
    selected_metadata["moving"] = actual[top_combined]

    moving_metadata = selected_metadata.loc[
        selected_metadata["moving"] == 1
    ]

    print("\n=== COMBINED TOP 10% MARKET COVERAGE ===")
    print(
        "Selected markets:",
        selected_metadata["ticker"].nunique(),
    )
    print(
        "Markets with movements:",
        moving_metadata["ticker"].nunique(),
    )
    print(
        "Sessions with movements:",
        moving_metadata["session_id"].nunique(),
    )

    print("\nTOP MARKETS BY CORRECTLY IDENTIFIED MOVEMENTS:")
    counts = (
        moving_metadata["ticker"]
        .value_counts()
        .head(15)
    )

    if counts.empty:
        print("No movements selected.")
    else:
        print(counts.to_string())

    print("\n=== EXPERIMENT COMPLETE ===")
    print("No models retrained.")
    print("Reserved test data not accessed.")


if __name__ == "__main__":
    main()
