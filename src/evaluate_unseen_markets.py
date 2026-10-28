
"""Evaluate regression GRU predictions on seen vs unseen markets.

Run:
    python -m src.evaluate_unseen_markets

Validation-only analysis. Does not train or place orders.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/regression_v1")
PREDICTIONS = Path(
    "models/regression_v1/validation_predictions.npz"
)
OUTPUT_DIR = Path(
    "models/regression_v1/unseen_market_analysis"
)

MOVE_THRESHOLD = 0.5
HORIZON_SECONDS = 10


def load_data():
    train = pd.read_csv(DATA_DIR / "train_metadata.csv")
    val = pd.read_csv(DATA_DIR / "val_metadata.csv")

    with np.load(PREDICTIONS) as data:
        actual = data["actual"].reshape(-1)
        predicted = data["predicted"].reshape(-1)

    if not (len(val) == len(actual) == len(predicted)):
        raise ValueError(
            "Validation metadata and predictions do not align."
        )

    if not {"ticker", "input_end"}.issubset(val.columns):
        raise ValueError("Missing ticker/input_end in validation metadata.")

    if "ticker" not in train.columns:
        raise ValueError("Missing ticker in training metadata.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite actual or predicted values.")

    train_tickers = set(train["ticker"].dropna().astype(str))

    val["ticker"] = val["ticker"].astype(str)
    val["input_end"] = pd.to_datetime(
        val["input_end"], utc=True
    )

    val["actual"] = actual
    val["predicted"] = predicted
    val["abs_predicted"] = np.abs(predicted)

    val["market_group"] = np.where(
        val["ticker"].isin(train_tickers),
        "Seen",
        "Unseen",
    )

    val["moving"] = np.abs(val["actual"]) >= MOVE_THRESHOLD

    return val


def nonoverlapping(df):
    """Greedy chronological selection: 10s cooldown per ticker."""
    selected = []
    next_allowed = {}

    for idx, row in df.sort_values(
        ["input_end", "ticker"]
    ).iterrows():
        ticker = row["ticker"]
        timestamp = row["input_end"]

        if (
            ticker in next_allowed
            and timestamp < next_allowed[ticker]
        ):
            continue

        selected.append(idx)
        next_allowed[ticker] = (
            timestamp + pd.Timedelta(seconds=HORIZON_SECONDS)
        )

    return df.loc[selected].copy()


def summarize(name, df):
    if df.empty:
        return {
            "group": name,
            "windows": 0,
            "markets": 0,
            "movement_rate": np.nan,
            "mae_cents": np.nan,
            "zero_baseline_mae_cents": np.nan,
            "direction_on_moves": np.nan,
            "avg_actual_magnitude_cents": np.nan,
            "avg_predicted_magnitude_cents": np.nan,
        }

    actual = df["actual"].to_numpy()
    predicted = df["predicted"].to_numpy()
    moving = np.abs(actual) >= MOVE_THRESHOLD

    direction_correct = (
        np.sign(actual) == np.sign(predicted)
    )

    return {
        "group": name,
        "windows": len(df),
        "markets": df["ticker"].nunique(),
        "movement_rate": float(moving.mean()),
        "mae_cents": float(np.abs(actual - predicted).mean()),
        "zero_baseline_mae_cents": float(np.abs(actual).mean()),
        "direction_on_moves": (
            float(direction_correct[moving].mean())
            if moving.any()
            else np.nan
        ),
        "avg_actual_magnitude_cents": float(np.abs(actual).mean()),
        "avg_predicted_magnitude_cents": float(np.abs(predicted).mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    summaries = []

    for group_name in ("Seen", "Unseen"):
        group = df[df["market_group"] == group_name].copy()
        spaced = nonoverlapping(group)

        summaries.append(summarize(group_name, group))
        summaries.append(
            summarize(f"{group_name} non-overlapping", spaced)
        )

        group.to_csv(
            OUTPUT_DIR / f"{group_name.lower()}_windows.csv",
            index=False,
        )

        spaced.to_csv(
            OUTPUT_DIR / f"{group_name.lower()}_nonoverlapping.csv",
            index=False,
        )

    report = pd.DataFrame(summaries)

    print("\n=== SEEN VS UNSEEN MARKETS ===")
    print(
        report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    per_market = df.groupby(
        ["market_group", "ticker"]
    ).agg(
        windows=("moving", "size"),
        movement_rate=("moving", "mean"),
        avg_abs_predicted=("abs_predicted", "mean"),
        avg_abs_actual=("actual", lambda s: s.abs().mean()),
    ).reset_index()

    print("\n=== UNSEEN MARKETS ===")
    unseen = per_market[
        per_market["market_group"] == "Unseen"
    ].sort_values("windows", ascending=False)

    print(
        unseen.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    report.to_csv(
        OUTPUT_DIR / "seen_vs_unseen_summary.csv",
        index=False,
    )

    per_market.to_csv(
        OUTPUT_DIR / "per_market_summary.csv",
        index=False,
    )

    print("\nSaved reports to:", OUTPUT_DIR)
    print("Reserved test dataset was not used.")


if __name__ == "__main__":
    main()
