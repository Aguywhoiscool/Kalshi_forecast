
"""Train movement-weighted 10-second midpoint regression GRU.

Run:
    python -m src.train_weighted_regression_gru

Uses training labels for weighting and unweighted validation
for model selection. Does not access the reserved test split.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


DATA_DIR = Path("data/processed/regression_v1")
MODEL_DIR = Path("models/regression_weighted_v1")
OLD_PREDICTIONS = Path(
    "models/regression_v1/validation_predictions.npz"
)
SCALER_PATH = Path("models/v1/feature_scaler.joblib")

EPOCHS = 30
BATCH_SIZE = 256

# A movement of at least half a cent qualifies for extra weight.
MOVEMENT_THRESHOLD_CENTS = 0.5
MOVEMENT_WEIGHT = 3.0


def load_split(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

    if X.ndim != 3 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"Invalid {name} dataset dimensions.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Nonfinite values in {name} dataset.")

    return X, y


def normalize(X, scaler):
    shape = X.shape
    return scaler.transform(
        X.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)


def create_weights(y):
    moving = np.abs(y) >= MOVEMENT_THRESHOLD_CENTS

    weights = np.where(
        moving,
        MOVEMENT_WEIGHT,
        1.0,
    ).astype(np.float32)

    # Keep average training weight at 1.
    weights /= weights.mean()

    print("\n=== TRAINING WEIGHTS ===")
    print("Training sequences:", len(y))
    print("Moving sequences:", int(moving.sum()))
    print("Moving fraction:", f"{moving.mean():.2%}")
    print("Moving weight multiplier:", MOVEMENT_WEIGHT)
    print("Mean sample weight:", round(float(weights.mean()), 4))

    return weights


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae_cents"
            ),
        ],
    )

    return model


def calculate_metrics(actual, predicted):
    error = np.abs(actual - predicted)
    moving = np.abs(actual) >= MOVEMENT_THRESHOLD_CENTS
    stationary = ~moving

    predicted_moving = (
        np.abs(predicted) >= MOVEMENT_THRESHOLD_CENTS
    )

    # Ranking diagnostic: top 10% largest predicted moves.
    n_top = max(1, int(np.ceil(len(predicted) * 0.10)))
    top_indices = np.argsort(np.abs(predicted))[-n_top:]
    top_movement_rate = float(moving[top_indices].mean())

    metrics = {
        "mae_all_cents": float(error.mean()),
        "mae_moving_cents": (
            float(error[moving].mean()) if moving.any() else np.nan
        ),
        "mae_stationary_cents": (
            float(error[stationary].mean())
            if stationary.any() else np.nan
        ),
        "prediction_std_cents": float(predicted.std()),
        "predicted_moving_rate": float(predicted_moving.mean()),
        "top_10pct_movement_rate": top_movement_rate,
    }

    return metrics


def main():
    tf.keras.utils.set_random_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen regression datasets...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    weights = create_weights(y_train)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen scaler: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    X_train = normalize(X_train, scaler)
    X_val = normalize(X_val, scaler)

    model = build_model(X_train.shape[1:])

    best_path = MODEL_DIR / "gru_weighted_best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    print("\nTraining movement-weighted GRU...")

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(best_path)

    weighted_predictions = best_model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    comparisons = {
        "Always zero": np.zeros_like(y_val),
        "Weighted GRU": weighted_predictions,
    }

    if OLD_PREDICTIONS.exists():
        with np.load(OLD_PREDICTIONS) as data:
            old_actual = data["actual"]
            old_predicted = data["predicted"].reshape(-1)

        if (
            len(old_actual) != len(y_val)
            or not np.allclose(old_actual, y_val, atol=1e-5)
        ):
            raise ValueError(
                "Original regression predictions use "
                "a different validation dataset."
            )

        comparisons["Original GRU"] = old_predicted

    print("\n=== VALIDATION COMPARISON ===")
    print(
        "Actual movements >= 0.5 cents:",
        f"{(np.abs(y_val) >= MOVEMENT_THRESHOLD_CENTS).mean():.2%}",
    )

    rows = []

    for name, predictions in comparisons.items():
        metrics = calculate_metrics(y_val, predictions)
        rows.append({"model": name, **metrics})

    report = pd.DataFrame(rows)

    print(report.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    report.to_csv(
        MODEL_DIR / "comparison.csv",
        index=False,
    )

    np.savez_compressed(
        MODEL_DIR / "validation_predictions.npz",
        actual=y_val,
        predicted=weighted_predictions,
    )

    print("\nSaved model:", best_path)
    print("Saved comparison:", MODEL_DIR / "comparison.csv")
    print("Reserved test split was not used.")


if __name__ == "__main__":
    main()
