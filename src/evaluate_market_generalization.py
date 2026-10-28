
"""Evaluate whether regression GRU movement signals generalize.

Run:
    python -m src.evaluate_market_generalization

Uses training history for the activity baseline and
validation data for evaluation. Does not access test data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/processed/regression_v1")
PRED_FILE = Path(
    "models/regression_v1/validation_predictions.npz"
)
OUT = Path("models/regression_v1/generalization")

MOVE_CENTS = 0.5
TOP_FRACTION = 0.10
COOLDOWN_SECONDS = 10

# Smooth historical rates toward the global training rate,
# especially for markets with few training observations.
PRIOR_STRENGTH = 100


def load_data():
    with np.load(PRED_FILE) as saved:
        actual = saved["actual"].reshape(-1)
        predicted = saved["predicted"].reshape(-1)

    val = pd.read_csv(ROOT / "val_metadata.csv")
    train = pd.read_csv(ROOT / "train_metadata.csv")

    if len(val) != len(actual) or len(val) != len(predicted):
        raise ValueError(
            "Validation predictions and metadata do not align."
        )

    required_val = {"ticker", "input_end"}
    required_train = {"ticker", "price_change_cents"}

    if not required_val.issubset(val.columns):
        raise ValueError("Validation ticker/time metadata missing.")

    if not required_train.issubset(train.columns):
        raise ValueError(
            "Training price_change_cents metadata missing."
        )

    val["input_end"] = pd.to_datetime(
        val["input_end"], utc=True
    )

    val["actual"] = actual
    val["predicted"] = predicted
    val["score"] = np.abs(predicted)
    val["moving"] = np.abs(actual) >= MOVE_CENTS

    train["moving"] = (
        train["price_change_cents"].abs() >= MOVE_CENTS
    )

    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite actual values.")
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite predictions.")

    return train, val


def add_historical_activity(train, val):
    """Calculate market activity rates from training data only."""

    global_rate = float(train["moving"].mean())

    history = train.groupby("ticker").agg(
        train_count=("moving", "size"),
        train_moves=("moving", "sum"),
    )

    history["historical_rate"] = (
        history["train_moves"]
        + PRIOR_STRENGTH * global_rate
    ) / (
        history["train_count"] + PRIOR_STRENGTH
    )

    result = val.join(
        history[["train_count", "historical_rate"]],
        on="ticker"
    )

    # Markets absent from training receive the global prior.
    result["historical_rate"] = (
        result["historical_rate"].fillna(global_rate)
    )
    result["train_count"] = (
        result["train_count"].fillna(0).astype(int)
    )

    return result, global_rate


def top_global(df, score_column):
    n = max(1, int(np.ceil(len(df) * TOP_FRACTION)))
    return df.nlargest(n, score_column).copy()


def top_within_market(df):
    groups = []

    for _, market in df.groupby("ticker"):
        n = max(
            1, int(np.ceil(len(market) * TOP_FRACTION))
        )
        groups.append(market.nlargest(n, "score"))

    return pd.concat(groups).copy()


def nonoverlapping(df):
    """Greedy 10-second cooldown per ticker."""

    selected = []
    next_allowed = {}

    for index, row in df.sort_values(
        ["input_end", "ticker"]
    ).iterrows():
        ticker = row["ticker"]
        timestamp = row["input_end"]

        if (
            ticker in next_allowed
            and timestamp < next_allowed[ticker]
        ):
            continue

        selected.append(index)

        next_allowed[ticker] = (
            timestamp
            + pd.Timedelta(seconds=COOLDOWN_SECONDS)
        )

    return df.loc[selected].copy()


def summarize(name, df):
    if df.empty:
        return None

    moving = df["moving"]
    correct = (
        np.sign(df["predicted"])
        == np.sign(df["actual"])
    )

    accuracy = (
        float(correct[moving].mean())
        if moving.any()
        else float("nan")
    )

    return {
        "method": name,
        "observations": len(df),
        "markets": df["ticker"].nunique(),
        "movement_rate": float(moving.mean()),
        "direction_accuracy_on_moves": accuracy,
        "avg_actual_move_cents": float(
            df["actual"].abs().mean()
        ),
        "avg_predicted_move_cents": float(
            df["predicted"].abs().mean()
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train, val = load_data()
    val, train_global_rate = add_historical_activity(
        train, val
    )

    global_gru = top_global(val, "score")
    historical = top_global(val, "historical_rate")
    within = top_within_market(val)

    selections = [
        ("All validation", val),
        ("GRU global top 10%", global_gru),
        ("Historical activity top 10%", historical),
        ("GRU within-market top 10%", within),
        (
            "GRU global non-overlapping",
            nonoverlapping(global_gru),
        ),
        (
            "Historical non-overlapping",
            nonoverlapping(historical),
        ),
        (
            "GRU within-market non-overlapping",
            nonoverlapping(within),
        ),
    ]

    results = [
        summarize(name, subset)
        for name, subset in selections
    ]

    summary = pd.DataFrame(
        [result for result in results if result]
    )

    print("\n=== MARKET GENERALIZATION ===")
    print(
        "Global training movement rate:",
        f"{train_global_rate:.2%}"
    )

    print(
        "Markets unseen in training:",
        val.loc[val["train_count"] == 0, "ticker"].nunique()
    )

    print("\n=== METHOD COMPARISON ===")
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    per_market = val.groupby("ticker").agg(
        observations=("moving", "size"),
        actual_movement_rate=("moving", "mean"),
        avg_predicted_magnitude=("score", "mean"),
        avg_actual_magnitude=("actual", lambda s: s.abs().mean()),
        historical_movement_rate=("historical_rate", "first"),
        training_observations=("train_count", "first"),
    ).reset_index()

    within_counts = (
        within["ticker"].value_counts()
        .rename("selected_windows")
    )

    within_move_rate = (
        within.groupby("ticker")["moving"]
        .mean()
        .rename("selected_movement_rate")
    )

    per_market = per_market.join(
        within_counts, on="ticker"
    ).join(
        within_move_rate, on="ticker"
    )

    per_market["within_market_lift"] = (
        per_market["selected_movement_rate"]
        / per_market["actual_movement_rate"].replace(0, np.nan)
    )

    print("\n=== GRU GLOBAL TOP 10% MARKET CONCENTRATION ===")
    print(
        global_gru["ticker"].value_counts()
        .head(10).to_string()
    )

    print("\n=== WITHIN-MARKET PERFORMANCE ===")
    print(
        per_market.sort_values(
            "observations", ascending=False
        ).head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary.to_csv(
        OUT / "method_comparison.csv", index=False
    )
    per_market.to_csv(
        OUT / "per_market.csv", index=False
    )

    global_gru.to_csv(
        OUT / "global_gru_signals.csv", index=False
    )
    historical.to_csv(
        OUT / "historical_activity_signals.csv", index=False
    )
    within.to_csv(
        OUT / "within_market_signals.csv", index=False
    )

    print("\nReports saved to:", OUT)
    print("Reserved test split was not used.")


if __name__ == "__main__":
    main()
