
"""Analyze existing GRU backtest results.

Run:
    python -m src.analyze_backtest
"""

from pathlib import Path
import pandas as pd


MODEL_DIR = Path("models")


def main():
    files = sorted(MODEL_DIR.glob("backtest_trades_*.csv"))

    if not files:
        print("No backtest trade files found.")
        return

    for filepath in files:
        df = pd.read_csv(filepath)

        if df.empty:
            continue
        
        required = [
            "midpoint_movement_cents",
            "spread_cost_cents",
            "break_even_midpoint_move_cents",
        ]

        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                f"{filepath.name} is missing columns: {missing}. "
                "Check the return dictionary in calculate_trade()."
            )

        print(f"\n{'=' * 60}")
        print(f"BACKTEST: {filepath.name}")
        print(f"{'=' * 60}")

        

        df["price_movement_cents"] = (
            df["exit_price"] - df["entry_price"]
        ) * 100

        df["total_fees"] = (
            df["entry_fee_est"] + df["exit_fee_est"]
        )

        df["break_even_movement_cents"] = (
            df["total_fees"] / df["contracts"]
        ) * 100

        
    for side, group in df.groupby("side"):
        print(f"\n--- {side} TRADES ---")
        print("Trades:", len(group))

        print(
            "Average directional midpoint movement:",
            round(group["midpoint_movement_cents"].mean(), 4),
            "cents",
        )

        print(
            "Average spread cost:",
            round(group["spread_cost_cents"].mean(), 4),
            "cents per contract",
        )

        print(
            "Average full break-even movement:",
            round(group["break_even_midpoint_move_cents"].mean(), 4),
            "cents",
        )

        print(
            "Average net P&L:",
            round(group["net_pnl_est"].mean(), 4),
            "dollars",
        )

        print(
            "Trades with favorable midpoint movement:",
            int((group["midpoint_movement_cents"] > 0).sum()),
        )

        print(
            "Profitable trades:",
            int((group["net_pnl_est"] > 0).sum()),
        )

    print("\n=== OVERALL ===")
    print("Total trades:", len(df))

    print(
        "Average directional midpoint movement:",
        round(df["midpoint_movement_cents"].mean(), 4),
        "cents",
    )

    print(
        "Average spread cost:",
        round(df["spread_cost_cents"].mean(), 4),
        "cents",
    )

    print(
        "Average full break-even movement:",
        round(df["break_even_midpoint_move_cents"].mean(), 4),
        "cents",
    )

    print(
        "Total estimated net P&L:",
        round(df["net_pnl_est"].sum(), 2),
        "dollars",
    )


if __name__ == "__main__":
    main()
