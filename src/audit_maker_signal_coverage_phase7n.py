
"""Phase 7N.9 — Verify raw order-book coverage for GRU signals.

Read-only audit. Does not access reserved test data.
Does not simulate fills or place orders.
"""

from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import json

import numpy as np
import pandas as pd


RAW = Path("data/raw/kalshi_2026-10-28_18-47-15.jsonl")
SIGNALS = Path("models/v2_phase7l/gross_simulation_results.csv")
OUTPUT = Path("models/v2_phase7n/signal_raw_coverage.csv")

HOLD_SECONDS = 10
TOP_FRACTION = 0.10


def timestamp(value):
    return pd.Timestamp(value).tz_convert("UTC")


def main():
    df = pd.read_csv(SIGNALS)

    required = {"ticker", "input_end", "gru_score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing signal columns: {missing}")

    df["input_end"] = pd.to_datetime(
        df["input_end"], utc=True
    )

    n = max(1, int(len(df) * TOP_FRACTION))
    signals = (
        df.sort_values(
            "gru_score", ascending=False, kind="stable"
        )
        .head(n)
        .copy()
        .reset_index(drop=True)
    )

    wanted = set(signals["ticker"])

    # Each interval begins when the market's snapshot arrives.
    # It ends when the subscription generation is replaced.
    intervals = defaultdict(list)
    active = {}
    generation = 0
    last_received = None

    counts = defaultdict(int)

    print("\n=== PHASE 7N.9 MARKET-LEVEL RAW COVERAGE ===")
    print("GRU signals:", len(signals))
    print("Markets of interest:", len(wanted))
    print("Scanning raw WebSocket recording...")

    with RAW.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)

            received = timestamp(record["received_at"])
            last_received = received

            message = record.get("message", {})
            kind = message.get("type")
            payload = message.get("msg", {})

            if kind == "subscribed":
                # Close every interval from the previous generation.
                for ticker, interval in active.items():
                    interval["end"] = received

                active.clear()
                generation += 1
                counts["generations"] += 1
                continue

            if kind != "orderbook_snapshot":
                continue

            ticker = payload.get("market_ticker")

            if ticker not in wanted:
                continue

            counts["wanted_snapshots"] += 1

            # A fresh snapshot replaces the old state for this ticker.
            if ticker in active:
                active[ticker]["end"] = received

            interval = {
                "start": received,
                "end": None,
                "generation": generation,
            }

            intervals[ticker].append(interval)
            active[ticker] = interval

            if line_number % 1_000_000 == 0:
                print(f"Processed {line_number:,} messages")

    if last_received is None:
        raise ValueError("Raw recording is empty.")

    # Final generation ends at the last recorded message.
    for interval in active.values():
        interval["end"] = last_received

    records = []

    for _, signal in signals.iterrows():
        ticker = signal["ticker"]
        signal_time = signal["input_end"]
        exit_time = signal_time + timedelta(
            seconds=HOLD_SECONDS
        )

        has_entry = False
        has_full_window = False
        snapshot_age = np.nan
        generation_id = None

        for interval in intervals.get(ticker, []):
            if (
                interval["start"]
                <= signal_time
                < interval["end"]
            ):
                has_entry = True
                snapshot_age = (
                    signal_time - interval["start"]
                ).total_seconds()
                generation_id = interval["generation"]

                if exit_time < interval["end"]:
                    has_full_window = True

                break

        records.append({
            "ticker": ticker,
            "session_id": signal.get("session_id"),
            "input_end": signal_time,
            "gru_score": signal["gru_score"],
            "has_raw_entry_book": has_entry,
            "has_full_10s_window": has_full_window,
            "snapshot_age_seconds": snapshot_age,
            "raw_generation": generation_id,
        })

    result = pd.DataFrame(records)

    entry_count = int(result["has_raw_entry_book"].sum())
    full_count = int(result["has_full_10s_window"].sum())

    print("\n=== MARKET-LEVEL COVERAGE RESULTS ===")
    print("Signals:", len(result))
    print(
        "Market snapshot available at entry:",
        f"{entry_count}/{len(result)}",
        f"({entry_count / len(result):.2%})",
    )
    print(
        "Same generation covers full 10 seconds:",
        f"{full_count}/{len(result)}",
        f"({full_count / len(result):.2%})",
    )
    print(
        "Markets with full-window coverage:",
        result.loc[
            result["has_full_10s_window"], "ticker"
        ].nunique(),
    )
    print(
        "Missing entry snapshots:",
        len(result) - entry_count,
    )
    print(
        "Generation ends within holding window:",
        entry_count - full_count,
    )

    valid_ages = result.loc[
        result["has_raw_entry_book"],
        "snapshot_age_seconds",
    ]

    if not valid_ages.empty:
        print(
            "Median time since snapshot:",
            f"{valid_ages.median():.2f}s",
        )

    print("\n=== UNCOVERED SIGNALS BY MARKET ===")
    missing_rows = result.loc[
        ~result["has_full_10s_window"]
    ]

    if missing_rows.empty:
        print("None")
    else:
        print(
            missing_rows["ticker"]
            .value_counts()
            .head(15)
            .to_string()
        )

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )
    result.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("Coverage only; no fill assumptions.")
    print("Reserved test data not accessed.")


if __name__ == "__main__":
    main()
