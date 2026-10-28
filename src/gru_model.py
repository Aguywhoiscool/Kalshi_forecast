
"""Train a GRU to forecast Kalshi midpoint direction.

Run:
    python -m src.gru_model

Forecast horizon: 10 seconds
Classes: 0=DOWN, 1=FLAT, 2=UP
"""

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight


SPLIT_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")

HORIZON_INDEX = 2  # 10-second prediction
BATCH_SIZE = 256
EPOCHS = 20

tf.keras.utils.set_random_seed(42)


def load_split(name):
    with np.load(SPLIT_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"][:, HORIZON_INDEX].astype(np.int32)

    return X, y


def normalize(X, scaler):
    """Apply the training-fitted scaler to all timesteps."""
    shape = X.shape

    X = scaler.transform(
        X.reshape(-1, shape[-1])
    ).reshape(shape)

    X = X.astype(np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite normalized features")

    return X


def build_gru():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 7)),

        tf.keras.layers.GRU(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.GRU(32),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training and validation data...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    scaler = joblib.load(
        MODEL_DIR / "feature_scaler.joblib"
    )

    X_train = normalize(X_train, scaler)
    X_val = normalize(X_val, scaler)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # Give rare directional movements more influence
    # during training.
    classes = np.array([0, 1, 2])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    sample_weights = weights[y_train].astype(np.float32)

    print("Class weights:", dict(zip(classes, weights)))

    model = build_gru()
    model.summary()

    checkpoint = MODEL_DIR / "gru_10s_best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    print("\nTraining GRU...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on validation, not the reserved test set.
    probabilities = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
    )

    predictions = np.argmax(probabilities, axis=1)

    print("\n=== GRU VALIDATION RESULTS: 10 SECONDS ===")

    print(
        "Accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    print(
        "Balanced accuracy:",
        round(
            balanced_accuracy_score(y_val, predictions),
            4,
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
            4,
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

    model.save(MODEL_DIR / "gru_10s_final.keras")
    print("\nModels saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
