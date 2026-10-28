
"""Fit a StandardScaler using Version 2 development training data only.

Run:
    python -m src.normalize_features_v2

Input:
    data/processed/datasets/v2_dev/splits/train.npz

Output:
    models/v2_dev/feature_scaler.joblib
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


SPLIT_DIR = Path("data/processed/datasets/v2_dev/splits")
MODEL_DIR = Path("models/v2_dev")

BATCH_SIZE = 1000


def main():
    train_path = SPLIT_DIR / "train.npz"
    output_path = MODEL_DIR / "feature_scaler.joblib"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_path}"
        )

    # Never silently replace an existing scaler.
    if output_path.exists():
        raise FileExistsError(
            f"Scaler already exists: {output_path}. "
            "Refusing to overwrite it."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Version 2 development training data...")

    with np.load(train_path) as dataset:
        X_train = dataset["X"].astype(np.float32)
        feature_names = dataset["feature_names"].tolist()

    if X_train.ndim != 3:
        raise ValueError(
            f"Expected 3D training data, got {X_train.shape}"
        )

    if X_train.shape[1:] != (60, 7):
        raise ValueError(
            f"Expected (N, 60, 7), got {X_train.shape}"
        )

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training data contains non-finite values."
        )

    print("Training data shape:", X_train.shape)
    print("Features:", feature_names)

    scaler = StandardScaler()

    # Fit only on the TRAIN split.
    # Process batches to limit temporary memory usage.
    for start in range(0, len(X_train), BATCH_SIZE):
        batch = X_train[start:start + BATCH_SIZE]

        flat_batch = batch.reshape(
            -1, batch.shape[-1]
        )

        scaler.partial_fit(flat_batch)

    # Verify the scaler using one training example.
    example = X_train[:1]

    normalized = scaler.transform(
        example.reshape(-1, example.shape[-1])
    ).reshape(example.shape)

    if not np.isfinite(normalized).all():
        raise ValueError(
            "Scaler produced non-finite values."
        )

    # Save only after successful validation.
    joblib.dump(scaler, output_path)

    print("\n=== VERSION 2 NORMALIZATION COMPLETE ===")
    print("Scaler saved:", output_path)

    print("\nFeature statistics:")

    for name, mean, scale in zip(
        feature_names,
        scaler.mean_,
        scaler.scale_,
    ):
        print(
            f"{name}: mean={mean:.6f}, "
            f"scale={scale:.6f}"
        )

    print("\nFirst timestep before normalization:")
    print(example[0, 0])

    print("\nFirst timestep after normalization:")
    print(normalized[0, 0])

    print("\nScaler verification: PASSED")


if __name__ == "__main__":
    main()
