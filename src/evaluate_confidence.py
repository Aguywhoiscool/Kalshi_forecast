
"""Evaluate GRU confidence thresholds on validation data.

Run:
    python -m src.evaluate_confidence

This script does NOT place trades.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


SPLIT_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}


def main():
    print("Loading GRU and validation data...")

    model = tf.keras.models.load_model(
        MODEL_DIR / "gru_10s_best.keras"
    )

    scaler = joblib.load(
        MODEL_DIR / "feature_scaler.joblib"
    )

    with np.load(SPLIT_DIR / "val.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"][:, 2].astype(np.int32)

    original_shape = X.shape

    X = scaler.transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(original_shape).astype(np.float32)

    probabilities = model.predict(
        X, batch_size=256, verbose=1
    )

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    results = []

    print("\n=== CONFIDENCE ANALYSIS: 10 SECONDS ===")

    for threshold in THRESHOLDS:
        # Require the predicted class to be directional.
        # FLAT predictions are always no-trade.
        signal = (
            (predictions != 1)
            & (confidence >= threshold)
        )

        count = int(signal.sum())

        if count == 0:
            print(f"{threshold:.0%}: No signals")
            continue

        correct = predictions[signal] == y[signal]

        directional_precision = float(correct.mean())

        # Among all validation rows, how many would
        # qualify as directional signals?
        signal_rate = count / len(y)

        print(
            f"{threshold:.0%} confidence | "
            f"Signals: {count:,} | "
            f"Signal rate: {signal_rate:.2%} | "
            f"Directional precision: "
            f"{directional_precision:.2%}"
        )

        for label in (0, 2):
            selected = signal & (predictions == label)
            class_count = int(selected.sum())

            if class_count:
                class_precision = float(
                    (y[selected] == label).mean()
                )
            else:
                class_precision = np.nan

            results.append({
                "threshold": threshold,
                "direction": LABELS[label],
                "signals": class_count,
                "precision": class_precision,
                "overall_signal_rate": signal_rate,
            })

    report = pd.DataFrame(results)

    output = MODEL_DIR / "gru_confidence_report.csv"
    report.to_csv(output, index=False)

    print("\n=== DIRECTIONAL BREAKDOWN ===")
    if not report.empty:
        print(report.to_string(index=False))

    print("\nReport saved:", output)
    print("No test-set data was used.")


if __name__ == "__main__":
    main()
