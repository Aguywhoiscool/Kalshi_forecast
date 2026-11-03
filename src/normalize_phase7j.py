
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/processed/datasets/v2_phase7j")
DATASET = ROOT / "gru_dataset_combined.npz"
TRAIN_INDICES = ROOT / "splits/train_indices.npy"
OUTPUT = Path("models/v2_phase7j/feature_scaler.joblib")

BATCH_SIZE = 1024
NUM_FEATURES = 7


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Scaler already exists: {OUTPUT}"
        )

    indices = np.load(TRAIN_INDICES)

    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("Invalid training indices.")

    print("Loading Phase 7J dataset...")

    with np.load(DATASET) as data:
        X = data["X"]

    if X.ndim != 3 or X.shape[1:] != (60, NUM_FEATURES):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if indices.min() < 0 or indices.max() >= len(X):
        raise ValueError("Training indices are out of range.")

    scaler = StandardScaler()

    print("Training windows:", len(indices))
    print("Feature shape:", X.shape[1:])

    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch = X[batch_indices]

        flattened = batch.reshape(-1, NUM_FEATURES)

        if not np.isfinite(flattened).all():
            raise ValueError("Non-finite feature values found.")

        scaler.partial_fit(flattened)

        processed = min(start + BATCH_SIZE, len(indices))

        if processed % (BATCH_SIZE * 50) < BATCH_SIZE:
            print(f"Processed {processed:,}/{len(indices):,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, OUTPUT)

    print("\n=== PHASE 7J NORMALIZATION ===")
    print("Training windows:", len(indices))
    print("Features:", scaler.n_features_in_)
    print("Feature means:", scaler.mean_)
    print("Feature standard deviations:", scaler.scale_)
    print("Saved scaler:", OUTPUT)
    print("Validation and test data were not used.")


if __name__ == "__main__":
    main()
