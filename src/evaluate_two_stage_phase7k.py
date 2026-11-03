
"""Phase 7K.1: Evaluate combined movement and direction forecasts.

Uses existing trained models and Phase 7J validation data only.
Predicts direction for ALL validation windows, including flat outcomes.
Selects non-overlapping windows without examining future outcomes.

Does not train models or access the reserved test split.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)


ROOT = Path("data/processed/datasets/v2_phase7j")
DATASET = ROOT / "gru_dataset_combined.npz"
METADATA_PATH = ROOT / "gru_dataset_combined.metadata.csv"
VAL_INDICES_PATH = ROOT / "splits/val_indices.npy"

SCALER_PATH = Path("models/v2_phase7j/feature_scaler.joblib")

MOVEMENT_PREDICTIONS = Path(
    "models/v2_phase7k/movement/validation_predictions.npz"
)

DIRECTION_MODEL = Path(
    "models/v2_phase7j/direction_corrected/"
    "gru_direction_best.keras"
)

OUTPUT_DIR = Path("models/v2_phase7k/two_stage")
OUTPUT_FILE = OUTPUT_DIR / "validation_predictions.npz"

THRESHOLD_CENTS = 0.5
BATCH_SIZE = 512
EPS = 1e-7


def select_spaced_indices(frame):
    """Select non-overlapping intervals per market without using labels."""
    chosen = []

    ordered = frame.sort_values(
        ["ticker", "input_start", "target_end"],
        kind="stable",
    )

    for _, group in ordered.groupby("ticker", sort=False):
        last_end = None

        for index, row in group.iterrows():
            if last_end is None or row["input_start"] >= last_end:
                chosen.append(int(index))
                last_end = row["target_end"]

    return np.asarray(sorted(chosen), dtype=np.int64)


def report_movement_ranking(name, labels, scores):
    print(f"\n=== {name} MOVEMENT RANKING ===")
    ranking = np.argsort(-scores, kind="stable")
    base = labels.mean()

    for fraction in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(len(labels) * fraction))
        selected = ranking[:n]
        count = int(labels[selected].sum())
        rate = count / n

        print(
            f"Top {fraction:.0%}: {count}/{n} moved | "
            f"rate={rate:.2%} | "
            f"lift={rate / base:.2f}x"
        )


def report_combined_signals(
    name,
    scores,
    direction,
    actual_movement,
    actual_direction,
):
    """Evaluate sign calls on ALL selected windows, including flat ones."""
    print(f"\n=== {name} SIGNAL RANKING ===")

    ranking = np.argsort(-scores, kind="stable")

    for fraction in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(len(scores) * fraction))
        selected = ranking[:n]

        moved = actual_movement[selected]
        correct_direction = direction[selected] == actual_direction[selected]

        moving_count = int(moved.sum())
        correct_count = int((moved & correct_direction).sum())

        all_window_hit_rate = correct_count / n

        conditional_accuracy = (
            correct_count / moving_count
            if moving_count
            else float("nan")
        )

        print(
            f"Top {fraction:.0%}: "
            f"{correct_count}/{n} correct directional moves | "
            f"all-window hit rate={all_window_hit_rate:.2%} | "
            f"movement rate={moving_count / n:.2%} | "
            f"direction accuracy if moved={conditional_accuracy:.2%}"
        )


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Results already exist: {OUTPUT_FILE}. "
            "Refusing to overwrite them."
        )

    print("Loading validation metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    for column in ("input_start", "target_end"):
        metadata[column] = pd.to_datetime(
            metadata[column],
            utc=True,
            errors="raise",
        )

    val_idx = np.load(
        VAL_INDICES_PATH
    ).astype(np.int64)

    if len(np.unique(val_idx)) != len(val_idx):
        raise ValueError("Duplicate validation indices.")

    if val_idx.min() < 0 or val_idx.max() >= len(metadata):
        raise ValueError("Validation indices out of range.")

    validation_metadata = metadata.iloc[val_idx]

    # Determine evaluation timestamps before examining outcomes.
    spaced_original_indices = select_spaced_indices(
        validation_metadata
    )

    with np.load(MOVEMENT_PREDICTIONS) as data:
        movement_idx = data["validation_indices"].astype(np.int64)
        movement_prob = data[
            "probability_movement"
        ].reshape(-1).astype(np.float64)

        volatility = data[
            "volatility"
        ].reshape(-1).astype(np.float64)

        saved_movement_labels = data["actual"].reshape(-1)

    # Confirm movement predictions have exactly the same row order.
    if not np.array_equal(movement_idx, val_idx):
        raise ValueError(
            "Movement predictions do not match validation index order."
        )

    if len(movement_prob) != len(val_idx):
        raise ValueError("Incorrect movement prediction count.")

    if not (
        np.isfinite(movement_prob).all()
        and np.isfinite(volatility).all()
    ):
        raise ValueError("Non-finite movement predictions.")

    if (
        np.any(movement_prob < 0)
        or np.any(movement_prob > 1)
    ):
        raise ValueError("Movement probabilities outside [0,1].")

    # Use original float64 metadata for the 0.5-cent threshold.
    changes = pd.to_numeric(
        validation_metadata["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    actual_move = np.abs(changes) >= THRESHOLD_CENTS

    if not np.array_equal(
        saved_movement_labels.astype(bool),
        actual_move,
    ):
        raise ValueError(
            "Movement predictions and original metadata labels differ."
        )

    scaler = joblib.load(SCALER_PATH)

    if getattr(scaler, "n_features_in_", None) != 7:
        raise ValueError("Unexpected scaler feature count.")

    print("Loading saved directional GRU...")

    direction_model = tf.keras.models.load_model(
        DIRECTION_MODEL,
        compile=False,
    )

    print("Loading combined feature data...")

    with np.load(DATASET) as data:
        X = data["X"]

    if X.shape != (len(metadata), 60, 7):
        raise ValueError(f"Unexpected data shape: {X.shape}")

    # Predict on all validation windows, NOT just actual movers.
    direction_prob = np.empty(
        len(val_idx),
        dtype=np.float32,
    )

    momentum10 = np.empty(
        len(val_idx),
        dtype=np.float64,
    )

    print(
        f"Generating directional predictions for "
        f"{len(val_idx):,} validation windows..."
    )

    for start in range(0, len(val_idx), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(val_idx))

        batch = X[val_idx[start:stop]].astype(
            np.float32,
            copy=True,
        )

        if not np.isfinite(batch).all():
            raise ValueError("Non-finite input features.")

        # Feature index 3 is midpoint.
        momentum10[start:stop] = (
            batch[:, -1, 3].astype(np.float64)
            - batch[:, -11, 3].astype(np.float64)
        )

        normalized = scaler.transform(
            batch.reshape(-1, 7)
        ).reshape(batch.shape).astype(np.float32)

        probabilities = direction_model(
            normalized,
            training=False,
        ).numpy().reshape(-1)

        direction_prob[start:stop] = probabilities

        if stop % 10240 < BATCH_SIZE or stop == len(val_idx):
            print(
                f"Processed {stop:,}/{len(val_idx):,}"
            )

    del X

    if not np.isfinite(direction_prob).all():
        raise ValueError("Non-finite directional probabilities.")

    if np.any(direction_prob < 0) or np.any(direction_prob > 1):
        raise ValueError("Directional probabilities outside [0,1].")

    # Three-class probabilities:
    # P(UP)   = P(move) * P(UP | move)
    # P(DOWN) = P(move) * P(DOWN | move)
    # P(FLAT) = 1 - P(move)
    p_up = movement_prob * direction_prob
    p_down = movement_prob * (1 - direction_prob)
    p_flat = 1 - movement_prob

    three_class_prob = np.column_stack([
        p_down,
        p_flat,
        p_up,
    ])

    if not np.allclose(
        three_class_prob.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError("Combined probabilities do not sum to 1.")

    # Class labels: DOWN=0, FLAT=1, UP=2.
    actual_class = np.where(
        changes >= THRESHOLD_CENTS,
        2,
        np.where(
            changes <= -THRESHOLD_CENTS,
            0,
            1,
        ),
    )

    predicted_class = np.argmax(
        three_class_prob,
        axis=1,
    )

    # Map original dataset indices back to validation positions.
    position_lookup = pd.Series(
        np.arange(len(val_idx), dtype=np.int64),
        index=val_idx,
    )

    spaced_positions = (
        position_lookup
        .loc[spaced_original_indices]
        .to_numpy(dtype=np.int64)
    )

    y_move = actual_move[spaced_positions]
    y_class = actual_class[spaced_positions]
    y_direction = (changes[spaced_positions] > 0).astype(int)

    p_move_spaced = movement_prob[spaced_positions]
    p_dir_spaced = direction_prob[spaced_positions]
    vol_spaced = volatility[spaced_positions]
    mom_spaced = momentum10[spaced_positions]

    p_3class_spaced = three_class_prob[spaced_positions]
    class_prediction = predicted_class[spaced_positions]

    # GRU UP=1, DOWN=0.
    gru_direction = (p_dir_spaced >= 0.5).astype(int)

    # Reversed momentum: positive recent change predicts DOWN.
    # Zero momentum is assigned UP, matching the previous
    # reverse-momentum baseline.
    reverse_momentum_direction = (mom_spaced < 0).astype(int)

    # Predefined signal-strength rules, no weight fitting.
    # GRU: P(move) times directional certainty.
    gru_signal_score = p_move_spaced * np.abs(
        2 * p_dir_spaced - 1
    )

    # Simple baseline: volatility ranking with reverse
    # recent-momentum direction.
    volatility_signal_score = vol_spaced

    print("\n=== PHASE 7K.1 FULL SYSTEM EVALUATION ===")
    print("All validation windows:", len(val_idx))
    print("Spaced validation windows:", len(spaced_positions))
    print("Spaced moving windows:", int(y_move.sum()))
    print("Movement base rate:", f"{y_move.mean():.2%}")

    spaced_frame = metadata.iloc[
        spaced_original_indices
    ]

    print(
        "Spaced markets:",
        spaced_frame["ticker"].nunique(),
    )
    print(
        "Spaced sessions:",
        spaced_frame["session_id"].nunique(),
    )

    print("\n=== MOVEMENT DETECTION ===")
    print(
        "GRU PR-AUC:",
        f"{average_precision_score(y_move, p_move_spaced):.4f}",
    )
    print(
        "Volatility PR-AUC:",
        f"{average_precision_score(y_move, vol_spaced):.4f}",
    )

    if len(np.unique(y_move)) == 2:
        print(
            "GRU ROC-AUC:",
            f"{roc_auc_score(y_move, p_move_spaced):.4f}",
        )

    report_movement_ranking(
        "GRU", y_move, p_move_spaced
    )

    report_movement_ranking(
        "VOLATILITY", y_move, vol_spaced
    )

    print("\n=== THREE-CLASS PREDICTION ===")
    print(
        "DOWN:",
        int((y_class == 0).sum()),
        "FLAT:",
        int((y_class == 1).sum()),
        "UP:",
        int((y_class == 2).sum()),
    )

    print(
        "Combined accuracy:",
        f"{accuracy_score(y_class, class_prediction):.2%}",
    )

    print(
        "Always FLAT accuracy:",
        f"{(y_class == 1).mean():.2%}",
    )

    print(
        "Combined log loss:",
        f"{log_loss(y_class, np.clip(
            p_3class_spaced, EPS, 1 - EPS
        ), labels=[0, 1, 2]):.4f}",
    )

    # Mean probability assigned to the actual outcome.
    print(
        "Mean probability of actual outcome:",
        f"{p_3class_spaced[
            np.arange(len(y_class)), y_class
        ].mean():.4f}",
    )

    print("\n=== DIRECTION WHEN A MOVEMENT OCCURRED ===")

    if y_move.any():
        print(
            "GRU conditional accuracy:",
            f"{(gru_direction[y_move] == y_direction[y_move]).mean():.2%}",
        )
        print(
            "Reverse 10s conditional accuracy:",
            f"{(
                reverse_momentum_direction[y_move]
                == y_direction[y_move]
            ).mean():.2%}",
        )

    report_combined_signals(
        "TWO-STAGE GRU",
        gru_signal_score,
        gru_direction,
        y_move,
        y_direction,
    )

    report_combined_signals(
        "VOLATILITY + REVERSE MOMENTUM",
        volatility_signal_score,
        reverse_momentum_direction,
        y_move,
        y_direction,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        validation_indices=val_idx,
        spaced_positions=spaced_positions,
        probability_movement=movement_prob,
        probability_up_given_move=direction_prob,
        probability_down=p_down,
        probability_flat=p_flat,
        probability_up=p_up,
        actual_class=actual_class,
        actual_price_change_cents=changes,
        volatility=volatility,
        momentum10=momentum10,
    )

    print("\n=== PHASE 7K.1 COMPLETE ===")
    print("Saved:", OUTPUT_FILE)
    print("No model training performed.")
    print("Reserved test data not accessed.")
    print(
        "These are forecasting diagnostics, "
        "not trading profitability results."
    )


if __name__ == "__main__":
    main()
