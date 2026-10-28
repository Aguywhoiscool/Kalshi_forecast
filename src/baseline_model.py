
"""Evaluate an always-FLAT baseline for Kalshi price forecasting.

Run:
    python -m src.baseline_model

Labels:
    0 = DOWN
    1 = FLAT
    2 = UP
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


SPLIT_DIR = Path("data/processed/splits")
LABEL_NAMES = ["DOWN", "FLAT", "UP"]


def evaluate_baseline(split_name):
    """Evaluate always predicting FLAT."""

    filepath = SPLIT_DIR / f"{split_name}.npz"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Missing {filepath}. Run split_dataset first."
        )

    with np.load(filepath) as dataset:
        y = dataset["y"]
        horizons = dataset["horizons"]

    print(f"\n{'=' * 55}")
    print(f"{split_name.upper()} DATASET")
    print(f"{'=' * 55}")
    print(f"Examples: {len(y):,}")

    for index, horizon in enumerate(horizons):

        actual = y[:, index]

        # Always predict FLAT (class 1).
        predicted = np.ones_like(actual)

        counts = np.bincount(actual, minlength=3)

        print(f"\n--- {horizon}-SECOND FORECAST ---")

        print("Actual class distribution:")

        for label, count in enumerate(counts):
            percentage = count / len(actual) * 100
            print(
                f"  {LABEL_NAMES[label]}: "
                f"{count:,} ({percentage:.2f}%)"
            )

        accuracy = accuracy_score(actual, predicted)

        balanced_accuracy = balanced_accuracy_score(
            actual, predicted
        )

        macro_f1 = f1_score(
            actual,
            predicted,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )

        print("\nBASELINE PERFORMANCE")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")
        print(f"Macro F1:          {macro_f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                actual,
                predicted,
                labels=[0, 1, 2],
                target_names=LABEL_NAMES,
                zero_division=0,
            )
        )

        print("Confusion matrix:")
        print(
            confusion_matrix(
                actual,
                predicted,
                labels=[0, 1, 2],
            )
        )


def main():
    for split_name in ("train", "val", "test"):
        evaluate_baseline(split_name)


if __name__ == "__main__":
    main()
