
"""Evaluate whether the regression GRU identifies useful movements.

Run:
    python -m src.evaluate_movement_signals

Validation only. No orders and no retraining.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_FILE = Path(
    "models/regression_v1/validation_predictions.npz"
)

METADATA_FILE = Path(
    "data/processed/regression_v1/val_metadata.csv"
)

OUTPUT_DIR = Path("models/regression_v1")

MOVEMENT_THRESHOLD = 0.5  # cents
TOP_FRACTION = 0.10
COOLDOWN_SECONDS = 10


def summarize(label, df):
    if df.empty:
        print(f"\n{label}: No observations")
        return

    moving = df["actual"].abs() >= MOVEMENT_THRESHOLD

    # Evaluate direction only when an actual movement occurred.
    correct = (
        np.sign(df["predicted"])
        == np.sign(df["actual"])
    )

    directional_accuracy = (
        correct[moving].mean() if moving.any() else float("nan")
    )

    print(f"\n=== {label} ===")
    print("Observations:", len(df))
    print("Unique markets:", df["ticker"].nunique())
    print("Actual movement rate:", f"{moving.mean():.2%}")
    print(
        "Direction accuracy on moves >= 0.5 cents:",
        f"{directional_accuracy:.2%}",
    )
    print(
        "Average absolute actual movement:",
        f"{df['actual'].abs().mean():.4f} cents",
    )
    print(
        "Average absolute predicted movement:",
        f"{df['predicted'].abs().mean():.4f} cents",
    )

    return {
        "group": label,
        "observations": len(df),
        "unique_markets": df["ticker"].nunique(),
        "movement_rate": moving.mean(),
        "directional_accuracy_on_moves": directional_accuracy,
        "avg_abs_actual_move_cents": df["actual"].abs().mean(),
        "avg_abs_predicted_move_cents": df["predicted"].abs().mean(),
    }


def select_nonoverlapping(df):
    """Select chronologically without overlapping same-market periods."""

    selected = []
    next_allowed = {}

    for row in df.sort_values("input_end").itertuples():
        # Ticker is sufficient to prevent overlapping trades
        # across reconnects and recording sessions.
        key = row.ticker

        if (
            key in next_allowed
            and row.input_end < next_allowed[key]
        ):
            continue

        selected.append(row.Index)

        next_allowed[key] = (
            row.input_end
            + pd.Timedelta(seconds=COOLDOWN_SECONDS)
        )

    return df.loc[selected].copy()


def main():
    with np.load(PREDICTIONS_FILE) as data:
        actual = data["actual"].reshape(-1)
        predicted = data["predicted"].reshape(-1)

    metadata = pd.read_csv(METADATA_FILE)

    if not (
        len(metadata) == len(actual) == len(predicted)
    ):
        raise ValueError(
            "Predictions and validation metadata do not align."
        )

    if not {"ticker", "input_end"}.issubset(metadata.columns):
        raise ValueError("Missing ticker or input_end metadata.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Nonfinite predictions or labels found.")

    df = metadata.copy()

    df["input_end"] = pd.to_datetime(
        df["input_end"], utc=True
    )

    df["actual"] = actual
    df["predicted"] = predicted
    df["abs_predicted"] = np.abs(predicted)

    # Select top 10% across the entire validation dataset.
    n_top = max(
        1,
        int(np.ceil(len(df) * TOP_FRACTION))
    )

    top = df.nlargest(
        n_top, "abs_predicted"
    ).copy()

    nonoverlapping = select_nonoverlapping(top)

    print("\nREGRESSION GRU MOVEMENT ANALYSIS")
    print("Movement threshold:", MOVEMENT_THRESHOLD, "cents")

    rows = []

    for name, subset in [
        ("Entire validation set", df),
        ("Top 10% predicted magnitude", top),
        ("Top 10% non-overlapping", nonoverlapping),
    ]:
        result = summarize(name, subset)
        if result:
            rows.append(result)

    baseline_rate = (
        df["actual"].abs() >= MOVEMENT_THRESHOLD
    ).mean()

    top_rate = (
        top["actual"].abs() >= MOVEMENT_THRESHOLD
    ).mean()

    print("\n=== MOVEMENT DETECTION ===")
    print(
        "Top 10% movement lift:",
        f"{top_rate / baseline_rate:.2f}x"
        if baseline_rate > 0 else "N/A",
    )

    print("\n=== MOST REPRESENTED MARKETS ===")
    print(
        top["ticker"].value_counts().head(10).to_string()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "movement_signal_summary.csv",
        index=False,
    )

    top.to_csv(
        OUTPUT_DIR / "movement_top10_signals.csv",
        index=False,
    )

    nonoverlapping.to_csv(
        OUTPUT_DIR / "movement_nonoverlapping.csv",
        index=False,
    )

    print("\nReports saved to:", OUTPUT_DIR)
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
