
"""Read-only 10-second GRU trading backtest.

Run:
    python -m src.backtest_signals

Uses validation data only.
No Kalshi orders are submitted.
"""

from pathlib import Path
from decimal import Decimal, ROUND_CEILING

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


DATA_DIR = Path("data/processed")
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = Path("models")

THRESHOLDS = (0.60, 0.70, 0.80)
HOLD_SECONDS = 10
CONTRACTS = 10

# Standard taker fee estimate. Some markets differ.
FEE_RATE = Decimal("0.07")

# Maximum distance between requested and recorded quote.
QUOTE_TOLERANCE = pd.Timedelta(milliseconds=600)


def estimated_fee(price, contracts):
    """General Kalshi taker-fee estimate, rounded up to cents."""
    p = Decimal(str(price))
    c = Decimal(str(contracts))

    fee = FEE_RATE * c * p * (Decimal("1") - p)

    return float(
        fee.quantize(
            Decimal("0.01"),
            rounding=ROUND_CEILING,
        )
    )


def load_predictions():
    model = tf.keras.models.load_model(
        MODEL_DIR / "gru_10s_best.keras"
    )
    scaler = joblib.load(
        MODEL_DIR / "feature_scaler.joblib"
    )

    with np.load(SPLIT_DIR / "val.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"][:, 2].astype(np.int32)

    metadata = pd.read_csv(
        SPLIT_DIR / "val_metadata.csv"
    )

    if len(X) != len(metadata):
        raise ValueError("Validation metadata does not match X.")

    if not {"source_file", "session_id", "ticker", "input_end"}.issubset(
        metadata.columns
    ):
        raise ValueError(
            "Missing combined-dataset metadata. "
            "Rebuild using the combined dataset builder."
        )

    shape = X.shape
    X = scaler.transform(
        X.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)

    probabilities = model.predict(
        X, batch_size=256, verbose=1
    )

    metadata["prediction"] = np.argmax(
        probabilities, axis=1
    )
    metadata["confidence"] = probabilities.max(axis=1)
    metadata["actual_label"] = y
    metadata["input_end"] = pd.to_datetime(
        metadata["input_end"], utc=True
    )

    return metadata


def attach_quotes(signals):
    """Match entry and 10-second exit quotes to predictions."""

    results = []

    for source_file, file_signals in signals.groupby("source_file"):
        source_path = DATA_DIR / source_file

        if not source_path.exists():
            print("Missing feature recording:", source_file)
            continue

        quotes = pd.read_csv(
            source_path,
            usecols=[
                "timestamp", "session_id", "ticker",
                "bid", "ask", "bid_depth", "ask_depth"
            ],
        )

        quotes["timestamp"] = pd.to_datetime(
            quotes["timestamp"], utc=True
        )

        for (session, ticker), group in file_signals.groupby(
            ["session_id", "ticker"]
        ):
            book = quotes[
                (quotes["session_id"] == session)
                & (quotes["ticker"] == ticker)
            ].copy()

            if book.empty:
                continue

            book = book.sort_values("timestamp")

            requests = group.copy().sort_values("input_end")

            # Entry uses the latest known quote at signal time.
            entry_book = book.rename(columns={
                "timestamp": "quote_time",
                "bid": "entry_bid",
                "ask": "entry_ask",
                "bid_depth": "entry_bid_depth",
                "ask_depth": "entry_ask_depth",
            })

            entry = pd.merge_asof(
                requests,
                entry_book[
                    [
                        "quote_time", "entry_bid", "entry_ask",
                        "entry_bid_depth", "entry_ask_depth"
                    ]
                ],
                left_on="input_end",
                right_on="quote_time",
                direction="backward",
                tolerance=QUOTE_TOLERANCE,
            )

            entry["target_time"] = (
                entry["input_end"]
                + pd.Timedelta(seconds=HOLD_SECONDS)
            )

            # Exit uses the latest observed quote at or before
            # the specified exit time.
            exit_book = book.rename(columns={
                "timestamp": "exit_quote_time",
                "bid": "exit_bid",
                "ask": "exit_ask",
                "bid_depth": "exit_bid_depth",
                "ask_depth": "exit_ask_depth",
            })

            matched = pd.merge_asof(
                entry.sort_values("target_time"),
                exit_book[
                    [
                        "exit_quote_time", "exit_bid", "exit_ask",
                        "exit_bid_depth", "exit_ask_depth"
                    ]
                ],
                left_on="target_time",
                right_on="exit_quote_time",
                direction="backward",
                tolerance=QUOTE_TOLERANCE,
            )

            matched["source_file"] = source_file
            matched["session_id"] = session
            matched["ticker"] = ticker

            results.append(matched)

    if not results:
        return pd.DataFrame()

    output = pd.concat(results, ignore_index=True)

    required = [
        "entry_bid", "entry_ask",
        "exit_bid", "exit_ask",
        "entry_bid_depth", "entry_ask_depth",
        "exit_bid_depth", "exit_ask_depth",
    ]

    return output.dropna(subset=required)


def calculate_trade(row, contracts):
    """Simulate buying YES or NO and liquidating 10s later."""

    direction = int(row["prediction"])

    if direction == 2:
        # UP: buy YES at ask; sell YES at future bid.
        entry_price = float(row["entry_ask"])
        exit_price = float(row["exit_bid"])
        available_entry = float(row["entry_ask_depth"])
        available_exit = float(row["exit_bid_depth"])
        side = "YES"

    elif direction == 0:
        # DOWN: buy NO at 1 - YES bid;
        # sell NO at 1 - future YES ask.
        entry_price = 1 - float(row["entry_bid"])
        exit_price = 1 - float(row["exit_ask"])
        available_entry = float(row["entry_bid_depth"])
        available_exit = float(row["exit_ask_depth"])
        side = "NO"

    else:
        return None

    # Require enough displayed best-price liquidity.
    if min(available_entry, available_exit) < contracts:
        return None

    if not (0 < entry_price < 1 and 0 < exit_price < 1):
        return None
    entry_midpoint = (
        float(row["entry_bid"]) + float(row["entry_ask"])
    ) / 2

    exit_midpoint = (
        float(row["exit_bid"]) + float(row["exit_ask"])
    ) / 2

    entry_spread = float(row["entry_ask"]) - float(row["entry_bid"])
    exit_spread = float(row["exit_ask"]) - float(row["exit_bid"])

    direction_multiplier = 1 if direction == 2 else -1

    midpoint_movement = (
        exit_midpoint - entry_midpoint
    ) * direction_multiplier

    spread_cost = (entry_spread + exit_spread) / 2
    entry_fee = estimated_fee(entry_price, contracts)
    exit_fee = estimated_fee(exit_price, contracts)

    gross_pnl = contracts * (exit_price - entry_price)
    net_pnl = gross_pnl - entry_fee - exit_fee

    return {
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "contracts": contracts,
        "entry_fee_est": entry_fee,
        "exit_fee_est": exit_fee,
        "gross_pnl": gross_pnl,
        "net_pnl_est": net_pnl,
        "entry_midpoint": entry_midpoint,
        "exit_midpoint": exit_midpoint,
        "entry_spread": entry_spread,
        "exit_spread": exit_spread,
        "midpoint_movement_cents": midpoint_movement * 100,
        "spread_cost_cents": spread_cost * 100,
        "break_even_midpoint_move_cents": (
            spread_cost + (entry_fee + exit_fee) / contracts
        ) * 100,
    }


def evaluate_threshold(quotes, threshold):
    candidates = quotes[
        (quotes["prediction"] != 1)
        & (quotes["confidence"] >= threshold)
    ].copy()

    candidates = candidates.sort_values(
        ["session_id", "ticker", "input_end"]
    )

    trades = []
    last_exit = {}

    for _, row in candidates.iterrows():
        key = (row["session_id"], row["ticker"])
        signal_time = row["input_end"]

        # No overlapping positions in the same market.
        if key in last_exit and signal_time < last_exit[key]:
            continue

        trade = calculate_trade(row, CONTRACTS)

        if trade is None:
            continue

        last_exit[key] = row["target_time"]

        trades.append({
            "threshold": threshold,
            "source_file": row["source_file"],
            "session_id": row["session_id"],
            "ticker": row["ticker"],
            "signal_time": signal_time,
            "target_time": row["target_time"],
            "entry_quote_time": row["quote_time"],
            "exit_quote_time": row["exit_quote_time"],
            "confidence": row["confidence"],
            "actual_label": row["actual_label"],
            "prediction": row["prediction"],
            "direction_correct": (
                row["prediction"] == row["actual_label"]
            ),
            **trade,
        })

    return pd.DataFrame(trades)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    signals = load_predictions()

    directional = signals[
        signals["prediction"] != 1
    ].copy()

    print("\nDirectional predictions:", len(directional))

    matched = attach_quotes(directional)

    print("Predictions with usable quote matches:", len(matched))

    summaries = []

    for threshold in THRESHOLDS:
        trades = evaluate_threshold(matched, threshold)

        print(f"\n=== {threshold:.0%} CONFIDENCE BACKTEST ===")

        if trades.empty:
            print("No eligible simulated trades.")
            continue

        trades_path = MODEL_DIR / (
            f"backtest_trades_{int(threshold * 100)}.csv"
        )
        trades.to_csv(trades_path, index=False)

        count = len(trades)
        wins = int((trades["net_pnl_est"] > 0).sum())
        net = float(trades["net_pnl_est"].sum())
        gross = float(trades["gross_pnl"].sum())

        accuracy = float(trades["direction_correct"].mean())

        print("Non-overlapping simulated trades:", count)
        print(f"Directional precision: {accuracy:.2%}")
        print(f"Net-positive trades: {wins} ({wins/count:.2%})")
        print(f"Gross P&L: ${gross:.2f}")
        print(f"Estimated net P&L: ${net:.2f}")
        print(f"Average net P&L/trade: ${net/count:.4f}")
        print("Saved:", trades_path)

        summaries.append({
            "threshold": threshold,
            "trades": count,
            "directional_precision": accuracy,
            "net_win_rate": wins / count,
            "gross_pnl": gross,
            "estimated_net_pnl": net,
            "average_net_pnl": net / count,
        })

    if summaries:
        summary_path = MODEL_DIR / "backtest_summary.csv"
        pd.DataFrame(summaries).to_csv(
            summary_path, index=False
        )
        print("\nSummary saved:", summary_path)

    print("\nVALIDATION-ONLY, QUOTE-BASED SIMULATION")
    print("No real trades were placed.")


if __name__ == "__main__":
    main()
