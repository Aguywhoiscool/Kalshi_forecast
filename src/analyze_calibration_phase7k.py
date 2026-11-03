
"""Phase 7K.2: Calibration and market-level signal analysis.

Uses saved two-stage validation predictions.
Uses training labels only to establish a constant-frequency baseline.
Does not retrain models or access the reserved test split.

Run:
    python -m src.analyze_calibration_phase7k
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)

ROOT = Path("data/processed/datasets/v2_phase7j")
METADATA_PATH = ROOT / "gru_dataset_combined.metadata.csv"
TRAIN_INDICES = ROOT / "splits/train_indices.npy"
VAL_INDICES = ROOT / "splits/val_indices.npy"

PREDICTIONS_PATH = Path(
    "models/v2_phase7k/two_stage/validation_predictions.npz"
)

THRESHOLD_CENTS = 0.5
EPS = 1e-7


def classes_from_changes(changes):
    """Class order: DOWN=0, FLAT=1, UP=2."""
    return np.where(
        changes >= THRESHOLD_CENTS,
        2,
        np.where(changes <= -THRESHOLD_CENTS, 0, 1),
    ).astype(int)


def calibration_table(name, scores, outcomes):
    print(f"\n=== {name} CALIBRATION ===")
    print(
        f"{'Predicted range':<18} {'Count':>8} "
        f"{'Mean predicted':>16} {'Actual rate':>14}"
    )

    boundaries = np.linspace(0, 1, 6)

    for i in range(5):
        low = boundaries[i]
        high = boundaries[i + 1]

        if i == 4:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)

        n = int(mask.sum())

        if n == 0:
            print(f"{low:.0%}-{high:.0%}: no observations")
            continue

        print(
            f"{low:.0%}-{high:.0%}:".ljust(18)
            + f"{n:>8}"
            + f"{scores[mask].mean():>16.2%}"
            + f"{outcomes[mask].mean():>14.2%}"
        )


def report_signal_ranking(frame):
    print("\n=== DIRECTIONAL SIGNAL QUALITY ===")

    ranked = frame.sort_values(
        "signal_score",
        ascending=False,
        kind="stable",
    )

    for fraction in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(len(ranked) * fraction))
        chosen = ranked.iloc[:n]

        moved = int(chosen["moving"].sum())
        correct = int(chosen["correct_directional_move"].sum())

        conditional = (
            correct / moved if moved > 0 else float("nan")
        )

        print(
            f"Top {fraction:.0%}: "
            f"{correct}/{n} correct directional moves | "
            f"hit rate={correct / n:.2%} | "
            f"movement rate={moved / n:.2%} | "
            f"direction if moved={conditional:.2%} | "
            f"markets={chosen['ticker'].nunique()} | "
            f"sessions={chosen['session_id'].nunique()}"
        )

    return ranked


def main():
    metadata = pd.read_csv(METADATA_PATH)

    train_idx = np.load(TRAIN_INDICES).astype(np.int64)
    val_idx = np.load(VAL_INDICES).astype(np.int64)

    if (
        len(np.unique(train_idx)) != len(train_idx)
        or len(np.unique(val_idx)) != len(val_idx)
    ):
        raise ValueError("Duplicate split indices.")

    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError("Train/validation overlap.")

    with np.load(PREDICTIONS_PATH) as data:
        saved_idx = data["validation_indices"].astype(np.int64)
        spaced_positions = data["spaced_positions"].astype(np.int64)

        p_move_all = data["probability_movement"]
        p_up_cond_all = data["probability_up_given_move"]

        p_down_all = data["probability_down"]
        p_flat_all = data["probability_flat"]
        p_up_all = data["probability_up"]

        actual_class_all = data["actual_class"]
        saved_changes = data["actual_price_change_cents"]

    if not np.array_equal(saved_idx, val_idx):
        raise ValueError("Saved validation indices do not align.")

    if (
        spaced_positions.ndim != 1
        or len(spaced_positions) == 0
        or len(np.unique(spaced_positions)) != len(spaced_positions)
        or spaced_positions.min() < 0
        or spaced_positions.max() >= len(val_idx)
    ):
        raise ValueError("Invalid spaced positions.")

    arrays = [
        p_move_all, p_up_cond_all,
        p_down_all, p_flat_all, p_up_all,
        actual_class_all, saved_changes,
    ]

    if any(len(a) != len(val_idx) for a in arrays):
        raise ValueError("Saved array length mismatch.")

    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Non-finite saved data.")

    for probabilities in (
        p_move_all, p_up_cond_all, p_down_all,
        p_flat_all, p_up_all,
    ):
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Invalid probabilities.")

    metadata_changes = pd.to_numeric(
        metadata.iloc[val_idx]["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.allclose(
        saved_changes, metadata_changes,
        rtol=1e-5, atol=1e-5,
    ):
        raise ValueError("Saved changes do not match metadata.")

    expected_classes = classes_from_changes(metadata_changes)

    if not np.array_equal(actual_class_all, expected_classes):
        raise ValueError("Saved outcome labels do not match metadata.")

    # Training-only class frequencies form our baseline.
    train_changes = pd.to_numeric(
        metadata.iloc[train_idx]["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    train_classes = classes_from_changes(train_changes)

    train_frequencies = (
        np.bincount(train_classes, minlength=3) / len(train_classes)
    )

    # Everything below evaluates spaced validation observations.
    y_class = expected_classes[spaced_positions]
    y_move = (y_class != 1).astype(int)

    p_move = np.asarray(p_move_all[spaced_positions], dtype=float)
    p_up_cond = np.asarray(p_up_cond_all[spaced_positions], dtype=float)

    p_three = np.column_stack([
        p_down_all[spaced_positions],
        p_flat_all[spaced_positions],
        p_up_all[spaced_positions],
    ]).astype(float)

    if not np.allclose(p_three.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Three-class probabilities do not sum to 1.")

    # Tiny numerical clipping for log-loss calculations.
    p_three_clipped = np.clip(p_three, EPS, 1 - EPS)
    p_three_clipped /= p_three_clipped.sum(axis=1, keepdims=True)

    constant_three = np.tile(
        train_frequencies, (len(y_class), 1)
    )

    constant_move = float(
        train_frequencies[0] + train_frequencies[2]
    )

    print("\n=== PHASE 7K.2 CALIBRATION SUMMARY ===")
    print("Spaced validation windows:", len(y_class))
    print("Actual movements:", int(y_move.sum()))
    print("Actual movement rate:", f"{y_move.mean():.2%}")
    print("Training movement rate:", f"{constant_move:.2%}")

    print("\n=== MOVEMENT PROBABILITY QUALITY ===")
    print(
        "GRU average predicted movement probability:",
        f"{p_move.mean():.2%}",
    )
    print(
        "Actual movement frequency:",
        f"{y_move.mean():.2%}",
    )
    print(
        "GRU Brier score:",
        f"{brier_score_loss(y_move, p_move):.4f}",
    )
    print(
        "Constant baseline Brier score:",
        f"{brier_score_loss(
            y_move, np.full(len(y_move), constant_move)
        ):.4f}",
    )
    print(
        "GRU PR-AUC:",
        f"{average_precision_score(y_move, p_move):.4f}",
    )

    calibration_table(
        "MOVEMENT PROBABILITY", p_move, y_move
    )

    print("\n=== THREE-CLASS PROBABILITY QUALITY ===")
    print("Class order: DOWN / FLAT / UP")

    print(
        "Training class frequencies:",
        np.round(train_frequencies, 4),
    )

    print(
        "Validation class frequencies:",
        np.round(
            np.bincount(y_class, minlength=3) / len(y_class),
            4,
        ),
    )

    print(
        "Two-stage log loss:",
        f"{log_loss(
            y_class, p_three_clipped, labels=[0, 1, 2]
        ):.4f}",
    )

    print(
        "Constant-frequency log loss:",
        f"{log_loss(
            y_class, constant_three, labels=[0, 1, 2]
        ):.4f}",
    )

    print(
        "Two-stage accuracy:",
        f"{(np.argmax(p_three, axis=1) == y_class).mean():.2%}",
    )

    majority_class = int(np.argmax(train_frequencies))

    print(
        "Training-majority baseline accuracy:",
        f"{(y_class == majority_class).mean():.2%}",
    )

    # Directional signal selection.
    predicted_up = p_up_cond >= 0.5
    actual_up = y_class == 2
    actual_down = y_class == 0

    correct_directional_move = (
        (predicted_up & actual_up)
        | (~predicted_up & actual_down)
    )

    signal_score = p_move * np.abs(2 * p_up_cond - 1)

    frame = metadata.iloc[
        val_idx[spaced_positions]
    ][["ticker", "session_id"]].copy()

    frame["moving"] = y_move
    frame["correct_directional_move"] = (
        correct_directional_move.astype(int)
    )
    frame["signal_score"] = signal_score

    # Useful descriptive grouping, not a true Kalshi event ID.
    frame["market_family"] = frame["ticker"].str.split("-").str[0]

    ranked = report_signal_ranking(frame)

    print("\n=== TOP 10% SIGNAL CONCENTRATION ===")

    top_n = max(1, int(len(ranked) * 0.10))
    top = ranked.iloc[:top_n]

    print("Selected observations:", len(top))
    print("Selected markets:", top["ticker"].nunique())
    print("Selected sessions:", top["session_id"].nunique())

    print(
        "Correct directional moves:",
        int(top["correct_directional_move"].sum()),
    )

    print("\nTop 10 markets by selected observations:")

    market_summary = (
        top.groupby("ticker")
        .agg(
            selected=("moving", "size"),
            movements=("moving", "sum"),
            correct=("correct_directional_move", "sum"),
        )
        .sort_values("selected", ascending=False)
        .head(10)
    )

    market_summary["hit_rate"] = (
        market_summary["correct"] / market_summary["selected"]
    )

    print(market_summary.to_string())

    print("\n=== MARKET FAMILY SIGNAL QUALITY ===")

    family_summary = (
        top.groupby("market_family")
        .agg(
            selected=("moving", "size"),
            movements=("moving", "sum"),
            correct=("correct_directional_move", "sum"),
            markets=("ticker", "nunique"),
        )
        .sort_values("selected", ascending=False)
    )

    family_summary["hit_rate"] = (
        family_summary["correct"] / family_summary["selected"]
    )

    print(family_summary.head(15).to_string())

    print("\n=== FINAL CHECK ===")
    print("Used existing saved validation predictions.")
    print("Compared probabilities against training-only class frequencies.")
    print("No models trained or modified.")
    print("Reserved test set not accessed.")


if __name__ == "__main__":
    main()
