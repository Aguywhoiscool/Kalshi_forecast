
"""Phase 7N.10 — Maker quote-touch opportunity audit.

Replays raw Kalshi order-book updates over the top 10% of
GRU validation signals. Does not assume fills or place orders.

BUY YES at prevailing best YES bid when GRU predicts UP.
BUY NO at prevailing best NO bid when GRU predicts DOWN.

Touch: opposite best ask falls to or below fixed limit.
Through: opposite best ask falls strictly below fixed limit.

Neither condition proves an actual execution.
"""

from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.orderbook import OrderBookManager


RAW = Path("data/raw/kalshi_2026-10-28_18-47-15.jsonl")
SIGNALS = Path("models/v2_phase7l/gross_simulation_results.csv")
OUTPUT = Path("models/v2_phase7n/maker_opportunities.csv")

FRACTION = 0.10
HOLD_SECONDS = 10
EPS = 1e-8


def get_opposite_ask(prices, direction):
    if prices is None:
        return None

    if direction == "YES":
        return prices["yes_ask"]

    # NO ask = 1 - YES bid
    bid = prices["yes_bid"]
    return None if bid is None else 1.0 - float(bid)


def activate_signal(row, manager):
    prices = manager.get_prices(row["ticker"])

    result = {
        "ticker": row["ticker"],
        "session_id": row["session_id"],
        "input_end": row["input_end"],
        "gru_score": row["gru_score"],
        "direction": row["direction"],
        "end_time": row["input_end"] + pd.Timedelta(
            seconds=HOLD_SECONDS
        ),
        "raw_entry_available": False,
        "entry_bid_matches": False,
        "entry_ask_matches": False,
        "maker_limit": np.nan,
        "first_touch_time": pd.NaT,
        "first_through_time": pd.NaT,
        "book_updates": 0,
        "sequence_valid": True,
    }

    if prices is None:
        return result

    yes_bid = prices["yes_bid"]
    yes_ask = prices["yes_ask"]

    if yes_bid is None or yes_ask is None:
        return result

    bid = float(yes_bid)
    ask = float(yes_ask)

    result["raw_entry_available"] = True
    result["entry_bid_matches"] = bool(
        np.isclose(bid, row["entry_bid"], atol=1e-5)
    )
    result["entry_ask_matches"] = bool(
        np.isclose(ask, row["entry_ask"], atol=1e-5)
    )

    # Fixed maker limit, as observed at signal time.
    result["maker_limit"] = (
        bid if row["direction"] == "YES" else 1.0 - ask
    )

    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    df = pd.read_csv(SIGNALS)

    required = {
        "ticker", "session_id", "input_end", "gru_score",
        "gru_up", "entry_bid", "entry_ask",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["input_end"] = pd.to_datetime(
        df["input_end"], utc=True
    )

    # Handle CSV boolean values explicitly.
    directions = df["gru_up"].astype(str).str.lower()
    if not directions.isin(("true", "false")).all():
        raise ValueError("Unexpected gru_up values")

    df["direction"] = np.where(
        directions.eq("true"), "YES", "NO"
    )

    count = max(1, int(len(df) * FRACTION))
    selected = (
        df.sort_values(
            "gru_score", ascending=False, kind="stable"
        )
        .head(count)
        .sort_values("input_end", kind="stable")
        .reset_index(drop=True)
    )

    if selected["input_end"].isna().any():
        raise ValueError("Missing signal timestamps")

    manager = OrderBookManager()
    active = {}
    finished = []

    next_signal = 0
    generation = 0
    counters = Counter()
    last_time = None

    print("\n=== PHASE 7N.10 MAKER OPPORTUNITY AUDIT ===")
    print("Signals:", len(selected))
    print("Markets:", selected["ticker"].nunique())
    print("Scanning raw book events...")

    with RAW.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            now = pd.Timestamp(record["received_at"])
            last_time = now

            # Activate signals BEFORE processing an event at
            # or after the signal timestamp. No future quotes.
            while (
                next_signal < len(selected)
                and selected.iloc[next_signal]["input_end"] <= now
            ):
                row = selected.iloc[next_signal]
                state = activate_signal(row, manager)
                active[next_signal] = state
                next_signal += 1

            # Remove expired signals before processing this event.
            expired = [
                idx for idx, state in active.items()
                if now >= state["end_time"]
            ]
            for idx in expired:
                finished.append(active.pop(idx))

            message = record.get("message", {})
            kind = message.get("type")

            if kind == "subscribed":
                # A reset should never occur during a covered
                # signal window, per Phase 7N.9.
                for state in active.values():
                    state["sequence_valid"] = False

                manager = OrderBookManager()
                generation += 1
                counters["generations"] += 1
                continue

            if kind not in (
                "orderbook_snapshot", "orderbook_delta"
            ):
                continue

            ticker = message.get("msg", {}).get(
                "market_ticker"
            )

            try:
                prices = manager.process(message)
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Replay error at raw line {line_number}: {exc}"
                ) from exc

            if manager.invalid_subscriptions:
                raise RuntimeError(
                    "Sequence integrity failure in raw replay: "
                    f"{manager.invalid_subscriptions} "
                    f"at line {line_number}"
                )

            if prices is None:
                continue

            for state in active.values():
                if (
                    state["ticker"] != ticker
                    or not state["raw_entry_available"]
                ):
                    continue

                state["book_updates"] += 1

                opposite_ask = get_opposite_ask(
                    prices, state["direction"]
                )

                if opposite_ask is None:
                    continue

                opposite_ask = float(opposite_ask)
                limit = float(state["maker_limit"])

                if (
                    pd.isna(state["first_touch_time"])
                    and opposite_ask <= limit + EPS
                ):
                    state["first_touch_time"] = now

                if (
                    pd.isna(state["first_through_time"])
                    and opposite_ask < limit - EPS
                ):
                    state["first_through_time"] = now

            if line_number % 1_000_000 == 0:
                print(f"Processed {line_number:,} messages")

            # All selected signals are over, so no need
            # to scan the remaining raw messages.
            if (
                next_signal == len(selected)
                and not active
            ):
                break

    if next_signal != len(selected):
        raise RuntimeError(
            f"Only reached {next_signal}/{len(selected)} signals"
        )

    # If the file ends during a holding window, it is uncovered.
    if active:
        raise RuntimeError(
            f"Raw file ended during {len(active)} signal windows"
        )

    result = pd.DataFrame(finished)

    if len(result) != len(selected):
        raise RuntimeError("Incomplete signal results")

    result["touched"] = result[
        "first_touch_time"
    ].notna()

    result["through"] = result[
        "first_through_time"
    ].notna()

    result["seconds_to_touch"] = (
        result["first_touch_time"]
        - result["input_end"]
    ).dt.total_seconds()

    result["seconds_to_through"] = (
        result["first_through_time"]
        - result["input_end"]
    ).dt.total_seconds()

    print("\n=== RAW ENTRY VALIDATION ===")
    print(
        "Valid raw entry books:",
        int(result["raw_entry_available"].sum()),
        "/", len(result)
    )
    print(
        "Bid matches:",
        int(result["entry_bid_matches"].sum()),
        "/", len(result)
    )
    print(
        "Ask matches:",
        int(result["entry_ask_matches"].sum()),
        "/", len(result)
    )
    print(
        "Sequence-valid windows:",
        int(result["sequence_valid"].sum()),
        "/", len(result)
    )

    eligible = result[
        result["raw_entry_available"]
        & result["entry_bid_matches"]
        & result["entry_ask_matches"]
        & result["sequence_valid"]
    ].copy()

    print("\n=== MAKER PRICE OPPORTUNITIES ===")
    print("Eligible signals:", len(eligible))

    if not eligible.empty:
        print(
            "Opposite ask touched maker limit:",
            f"{eligible['touched'].sum()}/{len(eligible)}",
            f"({eligible['touched'].mean():.2%})",
        )
        print(
            "Opposite ask moved through maker limit:",
            f"{eligible['through'].sum()}/{len(eligible)}",
            f"({eligible['through'].mean():.2%})",
        )
        print(
            "No touch within 10 seconds:",
            int((~eligible["touched"]).sum()),
        )
        print(
            "Median updates during window:",
            f"{eligible['book_updates'].median():.0f}",
        )

        for column, label in (
            ("seconds_to_touch", "touch"),
            ("seconds_to_through", "trade-through proxy"),
        ):
            observed = eligible[column].dropna()
            if not observed.empty:
                print(
                    f"Median time to {label}: "
                    f"{observed.median():.3f}s"
                )

        print("\n=== BY PREDICTED DIRECTION ===")
        for direction, part in eligible.groupby("direction"):
            print(
                f"{direction}: n={len(part)}, "
                f"touch={part['touched'].mean():.2%}, "
                f"through={part['through'].mean():.2%}"
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("No orders placed; no fills inferred.")
    print("No reserved test data accessed.")


if __name__ == "__main__":
    main()
