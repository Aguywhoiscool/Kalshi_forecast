
"""Train a logistic regression baseline for Kalshi forecasting.

Run:
    python -m src.train_logistic
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


SPLIT_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")

HORIZON_INDEX = 2  # 0=1s, 1=5s, 2=10s


def load_split(name):
    with np.load(SPLIT_DIR / f"{name}.npz") as data:
        X = data["X"]
        y = data["y"][:, HORIZON_INDEX]

    return X, y


def main():
    print("Loading datasets...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    scaler = joblib.load(
        MODEL_DIR / "feature_scaler.joblib"
    )

    # Initial baseline: use the latest timestep only.
    X_train = X_train[:, -1, :]
    X_val = X_val[:, -1, :]

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    print("Training examples:", len(X_train))
    print("Validation examples:", len(X_val))

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    print("\nTraining logistic regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print("\n=== VALIDATION RESULTS: 10 SECONDS ===")

    print(
        "Accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    print(
        "Balanced accuracy:",
        round(
            balanced_accuracy_score(y_val, predictions),
            4
        )
    )

    print(
        "Macro F1:",
        round(
            f1_score(
                y_val,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            ),
            4
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1, 2],
            target_names=["DOWN", "FLAT", "UP"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1, 2],
        )
    )

    output = MODEL_DIR / "logistic_10s.joblib"
    joblib.dump(model, output)

    print("\nModel saved:", output)


if __name__ == "__main__":
    main()
