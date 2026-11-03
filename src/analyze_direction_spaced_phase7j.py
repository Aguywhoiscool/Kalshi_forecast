
"""Phase 7J.1: Evaluate directional predictions on spaced windows.

Uses the corrected directional GRU's saved validation predictions.
Selects non-overlapping observations BEFORE checking outcomes.

Does not retrain or access the reserved test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path("data/processed/datasets/v2_phase7j")

METADATA_PATH = ROOT / "gru_dataset_combined.metadata.csv"
VAL_INDICES_PATH = ROOT / "splits/val_indices.npy"

PREDICTIONS_PATH = Path(
    "models/v2_phase7j/direction_corrected/"
    "validation_predictions.npz"
)

THRESHOLD_CENTS = 0.5


def choose_nonoverlapping_indices(metadata):
    """Select non-overlapping input-to-target intervals per ticker.

    Does not inspect movement labels or model predictions.
    """

    selected = []

    ordered = metadata.sort_values(
        ["ticker", "input_start", "target_end"],
        kind="stable",
    )

    for ticker, group in ordered.groupby("ticker", sort=False):
        last_end = None

        for idx, row in group.iterrows():
            start = row["input_start"]
            end = row["target_end"]

            if last_end is None or start >= last_end:
                selected.append(int(idx))
                last_end = end

    return np.array(sorted(selected), dtype=np.int64)


def report(name, actual, predictions):
    accuracy = accuracy_score(actual, predictions)
    balanced = balanced_accuracy_score(actual, predictions)

    print(
        f"{name:<24} "
        f"accuracy={accuracy:.2%} | "
        f"balanced={balanced:.2%}"
    )


def main():
    print("Loading validation metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column],
            utc=True,
            errors="raise",
        )

    val_indices = np.load(VAL_INDICES_PATH)

    if len(np.unique(val_indices)) != len(val_indices):
        raise ValueError("Duplicate validation indices.")

    val_metadata = metadata.iloc[val_indices].copy()

    # Preserve original dataset indices in the dataframe index.
    # Spacing is selected WITHOUT using price changes.
    spaced_indices = choose_nonoverlapping_indices(val_metadata)

    with np.load(PREDICTIONS_PATH) as data:
        saved_indices = data["validation_indices"].astype(np.int64)
        saved_actual = data["actual"].astype(int)
        probability_up = data["probability_up"].reshape(-1)
        momentum_10s = data["momentum_10s"].reshape(-1)
        momentum_60s = data["momentum_60s"].reshape(-1)
        saved_changes = data["price_change_cents"].reshape(-1)

    if not (
        len(saved_indices)
        == len(saved_actual)
        == len(probability_up)
        == len(momentum_10s)
        == len(momentum_60s)
        == len(saved_changes)
    ):
        raise ValueError("Prediction array lengths differ.")

    if len(np.unique(saved_indices)) != len(saved_indices):
        raise ValueError("Duplicate saved prediction indices.")

    if not np.isfinite(
        np.column_stack(
            [probability_up, momentum_10s, momentum_60s, saved_changes]
        )
    ).all():
        raise ValueError("Non-finite prediction values detected.")

    if not np.isin(saved_indices, val_indices).all():
        raise ValueError("Predictions contain non-validation indices.")

    # Verify saved labels against the original float64 metadata.
    saved_metadata_changes = pd.to_numeric(
        metadata.iloc[saved_indices]["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    expected_moving = np.abs(saved_metadata_changes) >= THRESHOLD_CENTS
    expected_up = (saved_metadata_changes > 0).astype(int)

    if not expected_moving.all():
        raise ValueError(
            "Saved predictions include non-moving windows "
            "under the float64 threshold."
        )

    if not np.array_equal(saved_actual, expected_up):
        raise ValueError("Saved directional labels do not match metadata.")

    # Match independently spaced observations to saved moving predictions.
    # Only moving windows have GRU directional predictions.
    lookup = pd.Series(
        np.arange(len(saved_indices), dtype=np.int64),
        index=saved_indices,
    )

    matching_positions = lookup.reindex(spaced_indices)
    moving_mask = matching_positions.notna().to_numpy()

    evaluated_indices = spaced_indices[moving_mask]
    positions = matching_positions[moving_mask].astype(int).to_numpy()

    actual = saved_actual[positions]
    probability = probability_up[positions]
    mom10 = momentum_10s[positions]
    mom60 = momentum_60s[positions]

    gru_prediction = (probability >= 0.5).astype(int)

    # Match the momentum definitions in the training script:
    # zero momentum predicts UP.
    normal10 = (mom10 >= 0).astype(int)
    normal60 = (mom60 >= 0).astype(int)

    reverse10 = 1 - normal10
    reverse60 = 1 - normal60

    print("\n=== PHASE 7J.1 SPACED DIRECTION EVALUATION ===")
    print("All validation windows:", len(val_indices))
    print("Non-overlapping windows:", len(spaced_indices))
    print("Spaced windows with meaningful movement:", len(actual))
    print("Movement rate:", f"{len(actual) / len(spaced_indices):.2%}")

    if len(actual) == 0:
        print("No moving observations selected. Cannot evaluate direction.")
        return

    selected_metadata = metadata.iloc[evaluated_indices].copy()
    selected_metadata["actual_up"] = actual
    selected_metadata["gru_correct"] = (gru_prediction == actual)

    print("Evaluated markets:", selected_metadata["ticker"].nunique())
    print("Evaluated sessions:", selected_metadata["session_id"].nunique())
    print("Actual DOWN:", int((actual == 0).sum()))
    print("Actual UP:", int((actual == 1).sum()))

    print("\n=== DIRECTION ACCURACY COMPARISON ===")
    report("GRU", actual, gru_prediction)
    report("10-second momentum", actual, normal10)
    report("Reverse 10-second", actual, reverse10)
    report("60-second momentum", actual, normal60)
    report("Reverse 60-second", actual, reverse60)
    report("Always UP", actual, np.ones_like(actual))
    report("Always DOWN", actual, np.zeros_like(actual))

    if len(np.unique(actual)) == 2:
        print(
            "\nGRU ROC-AUC:",
            f"{roc_auc_score(actual, probability):.4f}",
        )
    else:
        print("\nGRU ROC-AUC: undefined (only one actual class)")

    print("\n=== GRU CONFIDENCE ANALYSIS ===")
    print("Confidence = max(P(UP), P(DOWN))")

    confidence = np.maximum(probability, 1 - probability)

    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        mask = confidence >= threshold
        n = int(mask.sum())

        if n == 0:
            print(f"Confidence >= {threshold:.0%}: no observations")
            continue

        correct = int((gru_prediction[mask] == actual[mask]).sum())

        print(
            f"Confidence >= {threshold:.0%}: "
            f"{correct}/{n} correct | "
            f"accuracy={correct / n:.2%}"
        )

    print("\n=== MARKET AND SESSION CONCENTRATION ===")

    market_counts = selected_metadata["ticker"].value_counts()
    session_counts = selected_metadata["session_id"].value_counts()

    print("Markets containing movements:", len(market_counts))
    print("Sessions containing movements:", len(session_counts))

    print(
        "Movement observations from 5 busiest markets:",
        int(market_counts.head(5).sum()),
        f"({market_counts.head(5).sum() / len(actual):.2%})",
    )

    print(
        "Movement observations from 5 busiest sessions:",
        int(session_counts.head(5).sum()),
        f"({session_counts.head(5).sum() / len(actual):.2%})",
    )

    print("\n=== TOP 10 MARKETS BY MOVEMENT COUNT ===")

    summary = (
        selected_metadata.groupby("ticker")
        .agg(
            movements=("actual_up", "size"),
            correct=("gru_correct", "sum"),
        )
        .sort_values("movements", ascending=False)
        .head(10)
    )

    summary["accuracy"] = summary["correct"] / summary["movements"]

    print(summary.to_string())

    print("\n=== FINAL CHECK ===")
    print("Used corrected directional predictions.")
    print("Selected spaced timestamps without using outcomes.")
    print("No models retrained.")
    print("Reserved test set not accessed.")


if __name__ == "__main__":
    main()
