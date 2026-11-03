
"""Phase 7K: Train a binary movement classifier on Phase 7J data.

Predicts whether abs(10-second midpoint change) >= 0.5 cents.

Uses TRAIN and VAL only. Does not access the reserved TEST set.

Run:
    python -m src.train_movement_gru_phase7k
"""

from pathlib import Path

import math

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path("data/processed/datasets/v2_phase7j")
DATASET = ROOT / "gru_dataset_combined.npz"
METADATA_PATH = ROOT / "gru_dataset_combined.metadata.csv"
SPLIT_DIR = ROOT / "splits"

SCALER_PATH = Path(
    "models/v2_phase7j/feature_scaler.joblib"
)

MODEL_DIR = Path(
    "models/v2_phase7k/movement"
)

MODEL_PATH = MODEL_DIR / "gru_movement_best.keras"
PREDICTIONS_PATH = MODEL_DIR / "validation_predictions.npz"

THRESHOLD_CENTS = 0.5
BATCH_SIZE = 256
EPOCHS = 20
SEED = 42
FEATURE_COUNT = 7


class IndexedBatches(tf.keras.utils.Sequence):
    """Load and normalize only the needed rows for each batch."""

    def __init__(
        self,
        X,
        indices,
        labels,
        scaler,
        training=False,
    ):
        super().__init__()

        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.scaler = scaler
        self.training = training

        if len(self.indices) != len(self.labels):
            raise ValueError("Indices and labels do not match.")

        self.order = np.arange(len(self.indices))
        self.rng = np.random.default_rng(SEED)

        if training:
            self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / BATCH_SIZE)

    def __getitem__(self, batch_number):
        start = batch_number * BATCH_SIZE
        stop = min(start + BATCH_SIZE, len(self.order))

        positions = self.order[start:stop]
        original_indices = self.indices[positions]

        batch = self.X[original_indices].astype(
            np.float32,
            copy=True,
        )

        if not np.isfinite(batch).all():
            raise ValueError("Non-finite feature values.")

        batch = self.scaler.transform(
            batch.reshape(-1, FEATURE_COUNT)
        ).reshape(batch.shape).astype(np.float32)

        return batch, self.labels[positions]

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.order)


def load_labels(metadata, split_name):
    indices = np.load(
        SPLIT_DIR / f"{split_name}_indices.npy"
    ).astype(np.int64)

    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"Invalid {split_name} indices.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate {split_name} indices.")

    if indices.min() < 0 or indices.max() >= len(metadata):
        raise ValueError(f"{split_name} indices out of range.")

    # IMPORTANT: Preserve float64 precision for the
    # 0.5-cent threshold. Do not convert before comparing.
    changes = pd.to_numeric(
        metadata.iloc[indices]["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(changes).all():
        raise ValueError("Non-finite target changes.")

    labels = (
        np.abs(changes) >= THRESHOLD_CENTS
    ).astype(np.float32)

    return indices, labels, changes


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, FEATURE_COUNT)),
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


def print_ranking(name, actual, scores):
    order = np.argsort(-scores, kind="stable")
    baseline = actual.mean()

    print(f"\n=== {name} MOVEMENT RANKING ===")

    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(len(actual) * fraction))
        chosen = order[:count]

        movements = int(actual[chosen].sum())
        rate = movements / count

        print(
            f"Top {fraction:.0%}: "
            f"{movements}/{count} movements | "
            f"rate={rate:.2%} | "
            f"lift={rate / baseline:.2f}x"
        )


def main():
    tf.keras.utils.set_random_seed(SEED)

    if MODEL_PATH.exists() or PREDICTIONS_PATH.exists():
        raise FileExistsError(
            "Phase 7K outputs already exist. "
            "Refusing to overwrite an experiment."
        )

    metadata = pd.read_csv(METADATA_PATH)

    train_idx, y_train, _ = load_labels(
        metadata, "train"
    )

    val_idx, y_val, val_changes = load_labels(
        metadata, "val"
    )

    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError("Train/validation indices overlap.")

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training labels need both classes.")

    if len(np.unique(y_val)) != 2:
        raise ValueError("Validation labels need both classes.")

    print("\n=== PHASE 7K MOVEMENT DATA ===")

    for name, y in (("TRAIN", y_train), ("VAL", y_val)):
        print(f"\n{name}")
        print("Windows:", len(y))
        print("Moving:", int(y.sum()))
        print("Flat:", int(len(y) - y.sum()))
        print("Movement rate:", f"{y.mean():.2%}")

    scaler = joblib.load(SCALER_PATH)

    if getattr(scaler, "n_features_in_", None) != FEATURE_COUNT:
        raise ValueError("Scaler must contain 7 features.")

    print("\nLoading combined feature array...")

    with np.load(DATASET) as data:
        X = data["X"]

    if X.shape != (len(metadata), 60, FEATURE_COUNT):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    # Share one feature array between train and validation.
    # Avoid creating additional full-sized normalized copies.
    train_batches = IndexedBatches(
        X, train_idx, y_train, scaler, training=True
    )

    val_batches = IndexedBatches(
        X, val_idx, y_val, scaler, training=False
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\n=== TRAINING PHASE 7K MOVEMENT GRU ===")

    model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    # Sequence batches are ordered during validation.
    # Predict in the same order as val_indices.
    probabilities = best_model.predict(
        val_batches,
        verbose=1,
    ).reshape(-1)

    if len(probabilities) != len(y_val):
        raise ValueError("Prediction count mismatch.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions.")

    # Feature index 3 is midpoint, in dollars.
    # Convert recent midpoint changes to cents.
    print("\nCalculating recent-volatility baseline...")

    volatility = np.empty(
        len(val_idx),
        dtype=np.float64,
    )

    for start in range(0, len(val_idx), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(val_idx))

        midpoints = X[val_idx[start:stop], :, 3].astype(
            np.float64
        )

        volatility[start:stop] = (
            np.abs(np.diff(midpoints, axis=1))
            .sum(axis=1)
            * 100
        )

    actual = y_val.astype(int)

    gru_ap = average_precision_score(
        actual, probabilities
    )

    volatility_ap = average_precision_score(
        actual, volatility
    )

    print("\n=== PHASE 7K MOVEMENT VALIDATION ===")
    print("Validation windows:", len(actual))
    print("Actual moving windows:", int(actual.sum()))
    print("Movement base rate:", f"{actual.mean():.2%}")
    print("GRU PR-AUC:", f"{gru_ap:.4f}")
    print("Volatility PR-AUC:", f"{volatility_ap:.4f}")
    print(
        "GRU ROC-AUC:",
        f"{roc_auc_score(actual, probabilities):.4f}",
    )

    print_ranking(
        "GRU CLASSIFIER",
        actual,
        probabilities,
    )

    print_ranking(
        "RECENT VOLATILITY",
        actual,
        volatility,
    )

    np.savez_compressed(
        PREDICTIONS_PATH,
        validation_indices=val_idx,
        actual=actual,
        probability_movement=probabilities,
        price_change_cents=val_changes,
        volatility=volatility,
    )

    print("\n=== PHASE 7K COMPLETE ===")
    print("Saved model:", MODEL_PATH)
    print("Saved predictions:", PREDICTIONS_PATH)
    print("No test data accessed.")


if __name__ == "__main__":
    main()
