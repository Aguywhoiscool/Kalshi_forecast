
"""Fit a feature scaler using training data only.

Run:
    python -m src.normalize_features
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


SPLIT_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")
BATCH_SIZE = 1000


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLIT_DIR / "train.npz"

    with np.load(train_path) as dataset:
        X_train = dataset["X"]
        feature_names = dataset["feature_names"]

    print("Training data shape:", X_train.shape)
    print("Features:", feature_names.tolist())

    scaler = StandardScaler()

    # Fit using ONLY the training dataset.
    # Process in batches to reduce temporary memory usage.
    for start in range(0, len(X_train), BATCH_SIZE):
        batch = X_train[start:start + BATCH_SIZE]

        # Convert (samples, timesteps, features)
        # to (observations, features).
        flat_batch = batch.reshape(
            -1, batch.shape[-1]
        )

        scaler.partial_fit(flat_batch)

    output_path = MODEL_DIR / "feature_scaler.joblib"

    joblib.dump(scaler, output_path)

    print("\nNORMALIZATION COMPLETE")
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

    # Demonstrate the transformation.
    example = X_train[:1]
    normalized = scaler.transform(
        example.reshape(-1, example.shape[-1])
    ).reshape(example.shape)

    print("\nFirst timestep before normalization:")
    print(example[0, 0])

    print("\nFirst timestep after normalization:")
    print(normalized[0, 0])

    assert np.isfinite(normalized).all()

    print("\nScaler verification passed.")


if __name__ == "__main__":
    main()
