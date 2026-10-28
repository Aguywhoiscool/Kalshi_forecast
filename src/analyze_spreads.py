
"""Analyze GRU backtest performance by bid-ask spread.

Run:
    python -m src.analyze_spreads
"""

from pathlib import Path
import pandas as pd

MODEL_DIR = Path("models")

SPREAD_BINS = [-0.001, 1, 2, 5, float("inf")]
SPREAD_LABELS = [
    "0-1 cents",
    ">1-2 cents",
    ">2-5 cents",
    ">5 cents",
]


def main():
    for threshold in (60, 70):
        path = MODEL_DIR / f"backtest_trades_{threshold}.csv"

        if not path.exists():
            print(f"Missing: {path}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        df["entry_spread_cents"] = df["entry_spread"] * 100

        df["spread_group"] = pd.cut(
            df["entry_spread_cents"],
            bins=SPREAD_BINS,
            labels=SPREAD_LABELS,
            include_lowest=True,
        )

        print(f"\n=== {threshold}% CONFIDENCE ===")

        summary = (
            df.groupby("spread_group", observed=True)
            .agg(
                trades=("ticker", "size"),
                directional_precision=("direction_correct", "mean"),
                avg_midpoint_move_cents=(
                    "midpoint_movement_cents", "mean"
                ),
                avg_spread_cost_cents=("spread_cost_cents", "mean"),
                avg_break_even_cents=(
                    "break_even_midpoint_move_cents", "mean"
                ),
                net_pnl=("net_pnl_est", "sum"),
            )
            .reset_index()
        )

        summary["directional_precision"] *= 100

        print(summary.to_string(index=False))

        output = MODEL_DIR / f"spread_analysis_{threshold}.csv"
        summary.to_csv(output, index=False)

        print("Saved:", output)


if __name__ == "__main__":
    main()
