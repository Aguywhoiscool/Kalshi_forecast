
"""Phase 7J: Train an UP/DOWN GRU on meaningful movements only.

Uses training and validation splits. Never reads test indices or labels.

Run:
    python -m src.train_direction_gru_phase7j
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


ROOT = Path("data/processed/datasets/v2_phase7j")
DATASET = ROOT / "gru_dataset_combined.npz"
METADATA = ROOT / "gru_dataset_combined.metadata.csv"
SPLITS = ROOT / "splits"

SCALER_PATH = Path("models/v2_phase7j/feature_scaler.joblib")
MODEL_DIR = Path("models/v2_phase7j/direction_corrected")

MODEL_PATH = MODEL_DIR / "gru_direction_best.keras"
PREDICTIONS_PATH = MODEL_DIR / "validation_predictions.npz"

THRESHOLD_CENTS = 0.5
BATCH_SIZE = 256
EPOCHS = 20
SEED = 42


def select_moving_indices(metadata, split):
    indices = np.load(SPLITS / f"{split}_indices.npy")

    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"Invalid {split} indices.")

    # Preserve float64 precision when applying the movement threshold.
    changes = metadata.iloc[indices][
        "price_change_cents"
    ].to_numpy(dtype=np.float64)

    moving = np.abs(changes) >= THRESHOLD_CENTS

    selected_indices = indices[moving]
    selected_changes = changes[moving]

    labels = (selected_changes > 0).astype(np.float32)

    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError(f"{split} does not contain both directions.")

    return selected_indices, labels, selected_changes


def normalize(X, scaler):
    result = scaler.transform(
        X.reshape(-1, 7)
    ).reshape(X.shape)

    return result.astype(np.float32)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 7)),

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

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="roc_auc"
            ),
        ],
    )

    return model


def print_metrics(name, actual, predicted):
    accuracy = accuracy_score(actual, predicted)

    balanced = balanced_accuracy_score(
        actual,
        predicted,
    )

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced accuracy: {balanced:.2%}")


def main():
    tf.keras.utils.set_random_seed(SEED)

    if MODEL_PATH.exists() or PREDICTIONS_PATH.exists():
        raise FileExistsError(
            "Directional model outputs already exist. "
            "Refusing to overwrite them."
        )

    print("Reading training and validation metadata...")
    metadata = pd.read_csv(METADATA)

    train_idx, y_train, _ = select_moving_indices(
        metadata, "train"
    )

    val_idx, y_val, val_changes = select_moving_indices(
        metadata, "val"
    )

    print("\n=== DIRECTIONAL DATA ===")
    print("Training moving windows:", len(train_idx))
    print("Validation moving windows:", len(val_idx))
    print("Training UP:", int(y_train.sum()))
    print("Training DOWN:", int((1 - y_train).sum()))
    print("Validation UP:", int(y_val.sum()))
    print("Validation DOWN:", int((1 - y_val).sum()))

    scaler = joblib.load(SCALER_PATH)

    if getattr(scaler, "n_features_in_", None) != 7:
        raise ValueError("Scaler does not have 7 features.")

    print("\nLoading feature arrays...")

    # NPZ is compressed, so loading X requires memory.
    # Only selected training and validation windows are
    # retained after extracting the feature array.
    with np.load(DATASET) as data:
        X = data["X"]

    if X.shape != (len(metadata), 60, 7):
        raise ValueError(
            f"Unexpected dataset shape: {X.shape}"
        )

    X_train = X[train_idx].astype(np.float32)
    X_val = X[val_idx].astype(np.float32)

    del X

    if not (
        np.isfinite(X_train).all()
        and np.isfinite(X_val).all()
    ):
        raise ValueError("Non-finite feature values detected.")

    # Simple price-momentum baselines use raw midpoint
    # values before normalization. Feature 3 = midpoint.
    midpoint = X_val[:, :, 3]

    momentum_60s = midpoint[:, -1] - midpoint[:, 0]
    momentum_10s = midpoint[:, -1] - midpoint[:, -11]

    # A zero momentum predicts UP consistently.
    momentum_60s_predictions = (
        momentum_60s >= 0
    ).astype(int)

    momentum_10s_predictions = (
        momentum_10s >= 0
    ).astype(int)

    print("Normalizing selected windows...")

    X_train = normalize(X_train, scaler)
    X_val = normalize(X_val, scaler)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\n=== TRAINING DIRECTIONAL GRU ===")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    best_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    probabilities = best_model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).reshape(-1)

    predictions = (probabilities >= 0.5).astype(int)

    actual = y_val.astype(int)

    print("\n=== PHASE 7J DIRECTION VALIDATION ===")
    print("Moving validation windows:", len(actual))
    print("UP base rate:", f"{actual.mean():.2%}")

    print_metrics("GRU DIRECTION", actual, predictions)

    print(
        "GRU ROC-AUC:",
        f"{roc_auc_score(actual, probabilities):.4f}",
    )

    print_metrics(
        "60-SECOND MOMENTUM",
        actual,
        momentum_60s_predictions,
    )

    print_metrics(
        "10-SECOND MOMENTUM",
        actual,
        momentum_10s_predictions,
    )

    always_up = np.ones_like(actual)
    always_down = np.zeros_like(actual)

    print_metrics("ALWAYS UP", actual, always_up)
    print_metrics("ALWAYS DOWN", actual, always_down)

    print("\nGRU CONFUSION MATRIX")
    print("Rows: actual DOWN, UP")
    print("Columns: predicted DOWN, UP")

    print(confusion_matrix(actual, predictions))

    np.savez_compressed(
        PREDICTIONS_PATH,
        actual=actual,
        probability_up=probabilities,
        price_change_cents=val_changes,
        momentum_60s=momentum_60s,
        momentum_10s=momentum_10s,
        validation_indices=val_idx,
    )

    print("\n=== TRAINING COMPLETE ===")
    print("Model:", MODEL_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("Test set was not accessed.")


if __name__ == "__main__":
    main()
