
"""Train an experimental GRU to predict 10-second midpoint movement.

Run:
    python -m src.train_movement_gru_v2

Target: absolute midpoint change >= 0.5 cents.
Uses expanded development train/validation splits only.
"""

from pathlib import Path
import math

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

DATA_DIR = Path("data/processed/datasets/v2_expanded_dev/splits")
SCALER_PATH = Path("models/v2_expanded_dev/feature_scaler.joblib")
MODEL_DIR = Path("models/v2_expanded_dev/movement_classifier_v2")

THRESHOLD_CENTS = 0.5
BATCH_SIZE = 256
EPOCHS = 20
SEED = 42

MODEL_PATH = MODEL_DIR / "gru_movement_best.keras"
PREDICTIONS_PATH = MODEL_DIR / "validation_predictions.npz"


def load_split(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        changes = data["y_regression"].reshape(-1).astype(np.float32)

    if X.ndim != 3 or X.shape[1:] != (60, 7):
        raise ValueError(f"Invalid {name} feature shape: {X.shape}")

    if len(X) != len(changes):
        raise ValueError(f"{name} features and targets are misaligned")

    if not np.isfinite(X).all() or not np.isfinite(changes).all():
        raise ValueError(f"Non-finite values in {name}")

    y = (np.abs(changes) >= THRESHOLD_CENTS).astype(np.float32)

    return X, y, changes


def normalized_dataset(X, y, scaler, training=False):
    """Stream normalized batches instead of copying the whole dataset."""

    count = len(X)

    def generator():
        indices = np.arange(count)

        if training:
            np.random.default_rng(SEED + generator.epoch).shuffle(indices)
            generator.epoch += 1

        for start in range(0, count, BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            batch = X[batch_indices]

            normalized = scaler.transform(
                batch.reshape(-1, 7)
            ).reshape(batch.shape).astype(np.float32)

            yield normalized, y[batch_indices]

    generator.epoch = 0

    dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 60, 7), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
        )

    # Allow Keras to read the required number of batches
    # across multiple training epochs.
    dataset = dataset.repeat()

    return dataset.prefetch(1)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 7)),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),
            tf.keras.metrics.AUC(
                curve="ROC",
                name="roc_auc",
            ),
        ],
    )

    return model


def predict_in_batches(model, X, scaler):
    predictions = []

    for start in range(0, len(X), BATCH_SIZE):
        batch = X[start:start + BATCH_SIZE]
        normalized = scaler.transform(
            batch.reshape(-1, 7)
        ).reshape(batch.shape).astype(np.float32)

        probabilities = model.predict(
            normalized,
            verbose=0,
        ).reshape(-1)

        predictions.append(probabilities)

    return np.concatenate(predictions)


def print_ranking(y, probabilities, scores, name):
    order = np.argsort(-scores, kind="stable")
    baseline = float(np.mean(y))

    print(f"\n=== {name} MOVEMENT RANKING ===")

    for fraction in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(len(y) * fraction))
        selected = order[:n]

        movements = int(y[selected].sum())
        rate = movements / n
        lift = rate / baseline if baseline > 0 else float("nan")

        print(
            f"Top {fraction:.0%}: "
            f"{movements}/{n} movements | "
            f"rate={rate:.2%} | lift={lift:.2f}x"
        )


def main():
    tf.keras.utils.set_random_seed(SEED)

    if MODEL_PATH.exists() or PREDICTIONS_PATH.exists():
        raise FileExistsError(
            "Movement model outputs already exist. "
            "Preserve the previous experiment before rerunning."
        )

    print("Loading expanded training and validation data...")

    X_train, y_train, _ = load_split("train")
    X_val, y_val, changes_val = load_split("val")

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Training movement rate:", f"{y_train.mean():.2%}")
    print("Validation movement rate:", f"{y_val.mean():.2%}")

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training requires both moving and flat examples.")

    if len(np.unique(y_val)) != 2:
        raise ValueError("Validation requires both moving and flat examples.")

    scaler = joblib.load(SCALER_PATH)

    if scaler.n_features_in_ != 7:
        raise ValueError("Scaler must contain seven input features.")

    train_data = normalized_dataset(
        X_train, y_train, scaler, training=True
    )
    val_data = normalized_dataset(
        X_val, y_val, scaler, training=False
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    print("\nTraining movement classifier...")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(len(y_train) / BATCH_SIZE),
        validation_steps=math.ceil(len(y_val) / BATCH_SIZE),
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    probabilities = predict_in_batches(
        best_model, X_val, scaler
    )

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predicted probabilities outside [0, 1].")

    midpoint = X_val[:, :, 3].astype(np.float64)
    volatility = (
        np.abs(np.diff(midpoint, axis=1)).sum(axis=1) * 100
    )

    print("\n=== MOVEMENT CLASSIFIER VALIDATION RESULTS ===")
    print("Validation windows:", f"{len(y_val):,}")
    print("Actual moving windows:", int(y_val.sum()))
    print("Movement base rate:", f"{y_val.mean():.2%}")
    print(
        "Classifier PR-AUC:",
        f"{average_precision_score(y_val, probabilities):.4f}",
    )
    print(
        "Volatility PR-AUC:",
        f"{average_precision_score(y_val, volatility):.4f}",
    )
    print(
        "Classifier ROC-AUC:",
        f"{roc_auc_score(y_val, probabilities):.4f}",
    )

    print_ranking(
        y_val, probabilities, probabilities, "GRU CLASSIFIER"
    )
    print_ranking(
        y_val, probabilities, volatility, "RECENT VOLATILITY"
    )

    np.savez_compressed(
        PREDICTIONS_PATH,
        actual=y_val,
        probability=probabilities,
        price_change_cents=changes_val,
        volatility=volatility,
    )

    print("\nSaved best model:", MODEL_PATH)
    print("Saved predictions:", PREDICTIONS_PATH)
    print("Reserved test data was not accessed.")


if __name__ == "__main__":
    main()
