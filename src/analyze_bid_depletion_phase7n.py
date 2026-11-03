
"""Phase 7N.11 — Bid-side depletion audit.

Read-only validation analysis on the top 10% GRU signals.

A hypothetical BUY YES posts at the existing YES bid.
A hypothetical BUY NO posts at the existing NO bid.

Tracks displayed depth changes at that fixed price for 10 seconds.
Negative deltas may be trades OR cancellations.
No fills or profitability are inferred.
Reserved test data is never accessed.
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import heapq
import json

import numpy as np
import pandas as pd

from src.orderbook import OrderBookManager


RAW = Path("data/raw/kalshi_2026-10-28_18-47-15.jsonl")
SIGNALS = Path("models/v2_phase7l/gross_simulation_results.csv")
OUTPUT = Path("models/v2_phase7n/bid_depletion_audit.csv")

TOP_FRACTION = 0.10
HOLD_SECONDS = 10
PRICE_TOL = Decimal("0.00001")
D = Decimal


def parse_time(value):
    return datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    ).astimezone(timezone.utc)


def load_signals():
    df = pd.read_csv(SIGNALS)

    needed = {
        "ticker", "session_id", "input_end",
        "gru_score", "gru_up", "entry_bid", "entry_ask",
    }
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Missing signal fields: {sorted(missing)}")

    flags = df["gru_up"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Unexpected gru_up values")

    df["direction"] = np.where(flags == "true", "YES", "NO")
    df["input_end"] = pd.to_datetime(
        df["input_end"], utc=True
    )

    n = max(1, int(len(df) * TOP_FRACTION))

    return (
        df.sort_values(
            "gru_score", ascending=False, kind="stable"
        )
        .head(n)
        .sort_values("input_end", kind="stable")
        .reset_index(drop=True)
    )


def get_midpoint(manager, ticker):
    prices = manager.get_prices(ticker)
    if prices is None or prices["midpoint"] is None:
        return None
    return prices["midpoint"]


def activate(row, manager):
    ticker = row["ticker"]
    prices = manager.get_prices(ticker)
    book = manager.books.get(ticker)

    state = {
        "ticker": ticker,
        "session_id": row["session_id"],
        "input_end": parse_time(row["input_end"]),
        "end_time": parse_time(row["input_end"])
        + timedelta(seconds=HOLD_SECONDS),
        "gru_score": float(row["gru_score"]),
        "direction": row["direction"],
        "book_valid": False,
        "entry_quote_matches": False,
        "window_valid": True,
        "limit_price": None,
        "initial_depth": None,
        "final_depth": None,
        "minimum_depth": None,
        "maximum_depth": None,
        "removed_quantity": D("0"),
        "added_quantity": D("0"),
        "negative_delta_events": 0,
        "positive_delta_events": 0,
        "total_book_updates": 0,
        "limit_level_updates": 0,
        "first_depletion_time": None,
        "first_initial_depth_removed_time": None,
        "entry_midpoint": None,
        "last_midpoint": None,
    }

    if not book or not book["valid"] or prices is None:
        return state

    if (
        prices["yes_bid"] is None
        or prices["yes_ask"] is None
    ):
        return state

    actual_bid = prices["yes_bid"]
    actual_ask = prices["yes_ask"]

    expected_bid = D(str(row["entry_bid"]))
    expected_ask = D(str(row["entry_ask"]))

    state["entry_quote_matches"] = (
        abs(actual_bid - expected_bid) <= PRICE_TOL
        and abs(actual_ask - expected_ask) <= PRICE_TOL
    )

    if row["direction"] == "YES":
        side = "yes"
        limit = actual_bid
    else:
        side = "no"
        limit = D("1") - actual_ask

    depth = book[side].get(limit, D("0"))

    state["book_valid"] = depth > 0
    state["side"] = side
    state["limit_price"] = limit
    state["initial_depth"] = depth
    state["final_depth"] = depth
    state["minimum_depth"] = depth
    state["maximum_depth"] = depth
    state["entry_midpoint"] = prices["midpoint"]
    state["last_midpoint"] = prices["midpoint"]

    return state


def observe_update(state, manager, message, now):
    if not state["book_valid"]:
        return

    ticker = state["ticker"]
    book = manager.books.get(ticker)

    if book is None or not book["valid"]:
        state["window_valid"] = False
        return

    state["total_book_updates"] += 1

    midpoint = get_midpoint(manager, ticker)
    if midpoint is not None:
        state["last_midpoint"] = midpoint

    payload = message["msg"]

    # Snapshots during an active window are not comparable
    # to incremental depletion; exclude that window.
    if message["type"] == "orderbook_snapshot":
        state["window_valid"] = False
        return

    if payload.get("side") != state["side"]:
        return

    price = D(str(payload.get("price_dollars")))
    if price != state["limit_price"]:
        return

    state["limit_level_updates"] += 1

    delta_value = payload.get("delta_fp")
    if delta_value is None:
        delta_value = payload.get("delta")

    delta = D(str(delta_value))

    if delta < 0:
        state["negative_delta_events"] += 1
        state["removed_quantity"] += -delta
    elif delta > 0:
        state["positive_delta_events"] += 1
        state["added_quantity"] += delta

    depth = book[state["side"]].get(
        state["limit_price"], D("0")
    )

    state["final_depth"] = depth
    state["minimum_depth"] = min(
        state["minimum_depth"], depth
    )
    state["maximum_depth"] = max(
        state["maximum_depth"], depth
    )

    if (
        depth == 0
        and state["first_depletion_time"] is None
    ):
        state["first_depletion_time"] = now

    if (
        state["removed_quantity"] >= state["initial_depth"]
        and state["first_initial_depth_removed_time"] is None
    ):
        state["first_initial_depth_removed_time"] = now


def as_float(value):
    return np.nan if value is None else float(value)


def finish(state):
    start = state["input_end"]

    depleted_at = state["first_depletion_time"]
    removed_at = state["first_initial_depth_removed_time"]

    initial_depth = state["initial_depth"]

    # Because additions can occur, cumulative removals
    # are NOT equivalent to a fill or queue advancement.
    depletion_fraction = (
        float(
            state["removed_quantity"] / initial_depth
        )
        if initial_depth is not None and initial_depth > 0
        else np.nan
    )

    favorable_move = np.nan
    if (
        state["entry_midpoint"] is not None
        and state["last_midpoint"] is not None
    ):
        change = (
            state["last_midpoint"]
            - state["entry_midpoint"]
        ) * D("100")

        if state["direction"] == "NO":
            change = -change

        favorable_move = float(change)

    return {
        "ticker": state["ticker"],
        "session_id": state["session_id"],
        "input_end": start,
        "direction": state["direction"],
        "gru_score": state["gru_score"],
        "book_valid": state["book_valid"],
        "entry_quote_matches": state["entry_quote_matches"],
        "window_valid": state["window_valid"],
        "maker_limit": as_float(state["limit_price"]),
        "initial_depth": as_float(initial_depth),
        "final_depth": as_float(state["final_depth"]),
        "minimum_depth": as_float(state["minimum_depth"]),
        "maximum_depth": as_float(state["maximum_depth"]),
        "removed_quantity": float(
            state["removed_quantity"]
        ),
        "added_quantity": float(
            state["added_quantity"]
        ),
        "removal_to_initial_depth_ratio": depletion_fraction,
        "negative_delta_events": state[
            "negative_delta_events"
        ],
        "positive_delta_events": state[
            "positive_delta_events"
        ],
        "limit_level_updates": state[
            "limit_level_updates"
        ],
        "total_book_updates": state[
            "total_book_updates"
        ],
        "fully_depleted": depleted_at is not None,
        "cumulative_removal_ge_initial": removed_at is not None,
        "seconds_to_depletion": (
            (depleted_at - start).total_seconds()
            if depleted_at is not None else np.nan
        ),
        "seconds_to_initial_depth_removed": (
            (removed_at - start).total_seconds()
            if removed_at is not None else np.nan
        ),
        "directional_midpoint_change_cents": favorable_move,
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    signals = load_signals()
    manager = OrderBookManager()

    active = {}
    active_by_ticker = defaultdict(set)
    expirations = []
    completed = []

    next_signal = 0
    counters = Counter()

    print("\n=== PHASE 7N.11 BID-SIDE DEPLETION AUDIT ===")
    print("Signals:", len(signals))
    print("Markets:", signals["ticker"].nunique())
    print("Holding period:", HOLD_SECONDS, "seconds")
    print("Replaying raw order-book messages...")

    def expire(signal_id):
        state = active.pop(signal_id)
        active_by_ticker[state["ticker"]].discard(signal_id)
        completed.append(finish(state))

    with RAW.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            now = parse_time(record["received_at"])

            # Finish before processing any event at or
            # after the end of a signal's holding window.
            while expirations and expirations[0][0] <= now:
                _, signal_id = heapq.heappop(expirations)
                if signal_id in active:
                    expire(signal_id)

            # Activate using only the order-book state
            # reconstructed before the current raw event.
            while (
                next_signal < len(signals)
                and parse_time(
                    signals.iloc[next_signal]["input_end"]
                ) <= now
            ):
                row = signals.iloc[next_signal]
                state = activate(row, manager)
                signal_id = next_signal

                active[signal_id] = state
                active_by_ticker[state["ticker"]].add(
                    signal_id
                )

                heapq.heappush(
                    expirations,
                    (state["end_time"], signal_id),
                )

                next_signal += 1

            message = record.get("message", {})
            kind = message.get("type")

            if kind == "subscribed":
                # A subscription reset invalidates active
                # windows rather than inventing continuity.
                for state in active.values():
                    state["window_valid"] = False

                manager = OrderBookManager()
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
                manager.process(message)
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Replay failed at line {line_number}: "
                    f"{exc}"
                ) from exc

            if manager.invalid_subscriptions:
                raise RuntimeError(
                    "Raw sequence continuity failure at "
                    f"line {line_number}"
                )

            for signal_id in active_by_ticker.get(
                ticker, ()
            ):
                observe_update(
                    active[signal_id],
                    manager,
                    message,
                    now,
                )

            if line_number % 1_000_000 == 0:
                print(
                    f"Processed {line_number:,} raw messages"
                )

            if (
                next_signal == len(signals)
                and not active
            ):
                break

    if len(completed) != len(signals):
        raise RuntimeError(
            f"Only completed {len(completed)} "
            f"of {len(signals)} signals"
        )

    result = pd.DataFrame(completed)

    eligible = result.loc[
        result["book_valid"]
        & result["entry_quote_matches"]
        & result["window_valid"]
    ].copy()

    print("\n=== ENTRY AND WINDOW VALIDATION ===")
    print("Completed:", len(result))
    print(
        "Valid entry book:",
        int(result["book_valid"].sum()),
    )
    print(
        "Matching entry quotes:",
        int(result["entry_quote_matches"].sum()),
    )
    print(
        "Valid windows:",
        int(result["window_valid"].sum()),
    )
    print("Fully eligible:", len(eligible))

    print("\n=== BID-SIDE DEPLETION RESULTS ===")

    if not eligible.empty:
        n = len(eligible)

        def show_rate(column, label):
            count = int(eligible[column].sum())
            print(
                f"{label}: {count}/{n} "
                f"({count / n:.2%})"
            )

        show_rate(
            "fully_depleted",
            "Price level completely depleted",
        )
        show_rate(
            "cumulative_removal_ge_initial",
            "Cumulative removals >= initial depth",
        )

        changed = (
            eligible["negative_delta_events"] > 0
        )
        print(
            "Any negative delta at maker price:",
            f"{changed.sum()}/{n} ({changed.mean():.2%})",
        )

        print(
            "Median initial depth:",
            f"{eligible['initial_depth'].median():.2f}",
        )
        print(
            "Median cumulative removed quantity:",
            f"{eligible['removed_quantity'].median():.2f}",
        )
        print(
            "Median cumulative added quantity:",
            f"{eligible['added_quantity'].median():.2f}",
        )
        print(
            "Median updates at maker price:",
            f"{eligible['limit_level_updates'].median():.0f}",
        )

        depleted = eligible.loc[
            eligible["fully_depleted"]
        ]
        if not depleted.empty:
            print(
                "Median seconds to full depletion:",
                f"{depleted['seconds_to_depletion'].median():.3f}",
            )
            print(
                "Mean favorable midpoint change "
                "for depleted signals:",
                f"{depleted['directional_midpoint_change_cents'].mean():+.3f}c",
            )

        print("\n=== BY PREDICTED DIRECTION ===")
        for direction, part in eligible.groupby(
            "direction"
        ):
            print(
                f"{direction}: n={len(part)} | "
                f"any removal="
                f"{(part['negative_delta_events'] > 0).mean():.2%} | "
                f"full depletion="
                f"{part['fully_depleted'].mean():.2%}"
            )

        print("\n=== DIRECTIONAL MOVEMENT ===")
        for label, subset in (
            ("Fully depleted", eligible.loc[
                eligible["fully_depleted"]
            ]),
            ("Not fully depleted", eligible.loc[
                ~eligible["fully_depleted"]
            ]),
        ):
            if not subset.empty:
                print(
                    f"{label}: n={len(subset)}, "
                    f"mean signed midpoint change="
                    f"{subset['directional_midpoint_change_cents'].mean():+.3f}c"
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\n=== PHASE 7N.11 COMPLETE ===")
    print("Saved:", OUTPUT)
    print("Quantity removal is not a confirmed trade.")
    print("Depth depletion is not a confirmed fill.")
    print("No orders placed.")
    print("Reserved test data not accessed.")


if __name__ == "__main__":
    main()
