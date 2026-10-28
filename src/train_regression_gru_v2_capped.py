
"""Train a GRU to predict 10-second midpoint changes in cents.

Run:
    python -m src.train_regression_gru
"""

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


DATA_DIR = Path(
    "data/processed/datasets/v2_dev_balanced_capped/splits"
)

MODEL_DIR = Path(
    "models/v2_dev_balanced_capped/regression"
)

SCALER_PATH = Path(
    "models/v2_dev_balanced_capped/feature_scaler.joblib"
)
SCALER_PATH = Path("models/v2_dev/feature_scaler.joblib")

EPOCHS = 30
BATCH_SIZE = 256


def load_split(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y_regression"].astype(np.float32)

    if X.ndim != 3 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"Unexpected {name} dataset shape.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values in {name} dataset.")

    return X, y


def normalize(X, scaler):
    shape = X.shape

    return scaler.transform(
        X.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.GRU(
            64,
            return_sequences=True,
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.GRU(32),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            32,
            activation="relu",
        ),

        # One continuous value: predicted change in cents.
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae_cents"
            )
        ],
    )

    return model


def report_results(y_true, predictions):
    predictions = predictions.reshape(-1)

    model_mae = np.mean(
        np.abs(y_true - predictions)
    )

    baseline_mae = np.mean(
        np.abs(y_true)
    )

    nonzero = np.abs(y_true) > 0.0001

    print("\n=== VALIDATION RESULTS ===")
    print("Validation sequences:", len(y_true))

    print(
        "Actual nonzero movements:",
        f"{nonzero.mean():.2%}",
    )

    print(
        "Always-zero baseline MAE:",
        f"{baseline_mae:.4f} cents",
    )

    print(
        "Regression GRU MAE:",
        f"{model_mae:.4f} cents",
    )

    print(
        "Average actual change:",
        f"{y_true.mean():+.4f} cents",
    )

    print(
        "Average predicted change:",
        f"{predictions.mean():+.4f} cents",
    )

    print(
        "Predicted change std:",
        f"{predictions.std():.4f} cents",
    )

    if nonzero.any():
        print(
            "MAE on actual nonzero movements:",
            f"{np.mean(np.abs(y_true[nonzero] - predictions[nonzero])):.4f}",
            "cents",
        )

    if model_mae < baseline_mae:
        print("Model beats the always-zero MAE baseline.")
    else:
        print("Model does not beat the always-zero MAE baseline.")


def main():
    tf.keras.utils.set_random_seed(42)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading regression datasets...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("Training sequences:", X_train.shape)
    print("Validation sequences:", X_val.shape)

    print("\nLoading frozen Version 1 scaler...")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    X_train = normalize(X_train, scaler)
    X_val = normalize(X_val, scaler)

    model = build_model(X_train.shape[1:])

    model.summary()

    best_path = MODEL_DIR / "gru_regression_best.keras"

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

    print("\nTraining regression GRU...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Evaluate the exact checkpoint selected by validation loss.
    best_model = tf.keras.models.load_model(best_path)

    predictions = best_model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    report_results(y_val, predictions)

    np.savez_compressed(
        MODEL_DIR / "validation_predictions.npz",
        actual=y_val,
        predicted=predictions.reshape(-1),
    )

    print("\nBest model saved:", best_path)
    print("Validation predictions saved.")


if __name__ == "__main__":
    main()
