
"""Compare original and capped Version 2 GRUs on identical validation data.

Run:
    python -m src.compare_gru_models_v2

This script:
- Uses only the balanced/capped development validation split
- Loads each model with its OWN training-fitted scaler
- Does not retrain or overwrite either model
- Does not access reserved test data
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


VALIDATION_PATH = Path(
    "data/processed/datasets/v2_dev_balanced_capped/splits/val.npz"
)

METADATA_PATH = Path(
    "data/processed/datasets/v2_dev_balanced_capped/splits/val_metadata.csv"
)

ORIGINAL_MODEL = Path(
    "models/v2_dev/regression/gru_regression_best.keras"
)

ORIGINAL_SCALER = Path(
    "models/v2_dev/feature_scaler.joblib"
)

CAPPED_MODEL = Path(
    "models/v2_dev_balanced_capped/regression/gru_regression_best.keras"
)

CAPPED_SCALER = Path(
    "models/v2_dev_balanced_capped/feature_scaler.joblib"
)

MOVEMENT_THRESHOLD = 0.5
BATCH_SIZE = 256
MIDPOINT_INDEX = 3
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def load_validation():
    with np.load(VALIDATION_PATH) as data:
        X = data["X"].astype(np.float32)
        actual = data["y_regression"].reshape(-1).astype(np.float32)

    metadata = pd.read_csv(METADATA_PATH)

    if X.ndim != 3 or X.shape[1:] != (60, 7):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if len(X) != len(actual) or len(X) != len(metadata):
        raise ValueError("Validation arrays and metadata do not align.")

    if not np.isfinite(X).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite validation data.")

    if "price_change_cents" in metadata.columns:
        metadata_y = pd.to_numeric(
            metadata["price_change_cents"],
            errors="raise",
        ).to_numpy()

        if not np.allclose(
            actual, metadata_y, rtol=1e-5, atol=1e-4
        ):
            raise ValueError("Validation targets and metadata disagree.")

    return X, actual, metadata


def predict_model(name, model_path, scaler_path, X):
    print(f"\nLoading {name}...")

    for path in (model_path, scaler_path):
        if not path.exists():
            raise FileNotFoundError(path)

    scaler = joblib.load(scaler_path)

    if scaler.n_features_in_ != X.shape[2]:
        raise ValueError(f"{name}: scaler feature count mismatch.")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    predictions = []

    for start in range(0, len(X), BATCH_SIZE):
        batch = X[start:start + BATCH_SIZE]
        shape = batch.shape

        normalized = scaler.transform(
            batch.reshape(-1, shape[2])
        ).reshape(shape).astype(np.float32)

        batch_predictions = model.predict(
            normalized,
            verbose=0,
        ).reshape(-1)

        predictions.append(batch_predictions)

    predicted = np.concatenate(predictions)

    if len(predicted) != len(X):
        raise ValueError(f"{name}: prediction count mismatch.")

    if not np.isfinite(predicted).all():
        raise ValueError(f"{name}: non-finite predictions.")

    return predicted


def regression_report(name, actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    zero_mae = np.mean(np.abs(actual))

    print(f"\n=== {name.upper()} REGRESSION RESULTS ===")
    print("Examples:", f"{len(actual):,}")
    print("GRU MAE:", f"{mae:.4f} cents")
    print("Zero baseline MAE:", f"{zero_mae:.4f} cents")
    print(
        "MAE relative to zero baseline:",
        f"{(mae / zero_mae - 1) * 100:+.2f}%"
        if zero_mae > 0 else "N/A"
    )
    print("Average prediction:", f"{predicted.mean():+.4f} cents")
    print("Prediction std:", f"{predicted.std():.4f} cents")


def ranking_report(name, actual, predicted, scores=None):
    if scores is None:
        scores = np.abs(predicted)

    moving = np.abs(actual) >= MOVEMENT_THRESHOLD
    base_rate = moving.mean()

    ranked = np.argsort(-scores, kind="stable")

    print(f"\n=== {name.upper()} MOVEMENT RANKING ===")
    print("Overall movement rate:", f"{base_rate:.2%}")

    for fraction in TOP_FRACTIONS:
        count = max(1, int(np.floor(len(actual) * fraction)))
        indices = ranked[:count]

        selected_moving = moving[indices]
        movement_count = int(selected_moving.sum())
        movement_rate = selected_moving.mean()

        lift = (
            movement_rate / base_rate
            if base_rate > 0 else float("nan")
        )

        predicted_sign = np.sign(predicted[indices])
        actual_sign = np.sign(actual[indices])

        correct_movers = int(np.sum(
            selected_moving
            & (predicted_sign == actual_sign)
        ))

        direction_accuracy = (
            correct_movers / movement_count
            if movement_count else float("nan")
        )

        # Counts all selected windows, including flat outcomes.
        correct_all_rate = correct_movers / count

        print(f"\nTop {fraction:.0%}:")
        print(f"  Selected: {count}")
        print(f"  Actual movers: {movement_count}")
        print(f"  Movement rate: {movement_rate:.2%}")
        print(f"  Lift: {lift:.2f}x")
        print(
            "  Direction accuracy among movers:",
            f"{direction_accuracy:.2%}",
        )
        print(
            "  Correct meaningful directions / all selected:",
            f"{correct_all_rate:.2%}",
        )


def main():
    print("=== PHASE 7E: SAME-DATA GRU COMPARISON ===")

    X, actual, metadata = load_validation()

    print("Validation windows:", f"{len(X):,}")
    print("Validation markets:", metadata["ticker"].nunique())
    print("Validation sessions:", metadata["session_id"].nunique())

    original_pred = predict_model(
        "Original GRU",
        ORIGINAL_MODEL,
        ORIGINAL_SCALER,
        X,
    )

    capped_pred = predict_model(
        "Balanced capped GRU",
        CAPPED_MODEL,
        CAPPED_SCALER,
        X,
    )

    regression_report(
        "Original GRU", actual, original_pred
    )

    regression_report(
        "Balanced capped GRU", actual, capped_pred
    )

    ranking_report(
        "Original GRU", actual, original_pred
    )

    ranking_report(
        "Balanced capped GRU", actual, capped_pred
    )

    # Simple activity baseline, calculated from raw input data.
    midpoint = X[:, :, MIDPOINT_INDEX].astype(np.float64)

    volatility = (
        np.abs(np.diff(midpoint, axis=1)).sum(axis=1) * 100.0
    )

    print("\n=== VOLATILITY-ONLY MOVEMENT DETECTION ===")
    moving = np.abs(actual) >= MOVEMENT_THRESHOLD
    base_rate = moving.mean()
    ranked = np.argsort(-volatility, kind="stable")

    for fraction in TOP_FRACTIONS:
        count = max(1, int(np.floor(len(actual) * fraction)))
        indices = ranked[:count]
        movement_count = int(moving[indices].sum())
        rate = movement_count / count
        lift = rate / base_rate if base_rate else float("nan")

        print(
            f"Top {fraction:.0%}: "
            f"{movement_count}/{count} moved, "
            f"rate={rate:.2%}, lift={lift:.2f}x"
        )

    # Fairly compare direction on the SAME volatility-selected
    # observations rather than each model's separate selections.
    print("\n=== SHARED TOP 10% VOLATILITY SELECTION ===")

    count = max(1, int(len(actual) * 0.10))
    indices = ranked[:count]
    selected_actual = actual[indices]
    selected_moving = (
        np.abs(selected_actual) >= MOVEMENT_THRESHOLD
    )

    print("Selected:", count)
    print("Actual movers:", int(selected_moving.sum()))

    for name, predicted in (
        ("Original GRU", original_pred),
        ("Balanced capped GRU", capped_pred),
    ):
        selected_pred = predicted[indices]

        correct = int(np.sum(
            selected_moving
            & (np.sign(selected_pred) == np.sign(selected_actual))
        ))

        movers = int(selected_moving.sum())

        print(f"\n{name}:")
        print("  Correct mover directions:", correct)
        print("  Wrong mover directions:", movers - correct)
        print("  Flat outcomes:", count - movers)
        print(
            "  Accuracy among movers:",
            f"{correct / movers:.2%}" if movers else "N/A",
        )
        print("  Correct / all selected:", f"{correct / count:.2%}")

    print("\nComparison complete.")
    print("Reserved test data was not accessed.")


if __name__ == "__main__":
    main()
