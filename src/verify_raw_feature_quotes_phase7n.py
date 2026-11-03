
"""Phase 7N.6: Compare raw reconstructed books with feature CSVs.

Read-only. Replays the first 100,000 raw WebSocket messages.
Uses only raw messages received at or before each feature timestamp.
Does not access model training/validation/test arrays.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd

from src.orderbook import OrderBookManager


RAW = Path("data/raw/kalshi_2026-10-28_18-47-15.jsonl")
FEATURE_DIR = Path("data/processed")

MAX_RAW_MESSAGES = 100_000
MAX_SAMPLES_PER_FILE = 300
MAX_QUOTE_AGE_SECONDS = 2.0
PRICE_TOLERANCE = 0.00001
DEPTH_TOLERANCE = 0.01

FIELDS = [
    "timestamp", "ticker",
    "bid", "ask", "spread",
    "bid_depth", "ask_depth",
]


def parse_time(value):
    return datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    ).astimezone(timezone.utc)


def raw_time_range():
    first = None
    last = None

    with RAW.open("r", encoding="utf-8") as source:
        for i, line in enumerate(source):
            if i >= MAX_RAW_MESSAGES:
                break

            record = json.loads(line)
            timestamp = record.get("received_at")

            if timestamp:
                parsed = parse_time(timestamp)

                if first is None:
                    first = parsed

                last = parsed

    if first is None or last is None:
        raise ValueError("No raw timestamps found.")

    return first, last


def load_samples(start, end):
    selected_files = []
    frames = []

    for path in sorted(FEATURE_DIR.glob("features_*.csv")):
        try:
            session_start = datetime.strptime(
                path.name.removeprefix("features_")[:19],
                "%Y-%m-%d_%H-%M-%S",
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if not (
            start - timedelta(minutes=5)
            <= session_start <= end
        ):
            continue

        frame = pd.read_csv(path, usecols=FIELDS)

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True, errors="coerce"
        )

        frame = frame.loc[
            frame["timestamp"].between(
                pd.Timestamp(start), pd.Timestamp(end)
            )
        ].copy()

        if frame.empty:
            continue

        step = max(
            1,
            int(np.ceil(len(frame) / MAX_SAMPLES_PER_FILE)),
        )

        frame = frame.iloc[::step].copy()
        frame["source_file"] = path.name

        selected_files.append(path.name)
        frames.append(frame)

    if not frames:
        raise ValueError("No overlapping feature rows found.")

    samples = pd.concat(frames, ignore_index=True)

    samples = samples.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    return selected_files, samples


def main():
    start, end = raw_time_range()
    files, samples = load_samples(start, end)

    print("\n=== PHASE 7N.6 RAW/FEATURE QUOTE COMPARISON ===")
    print("Raw interval:", start, "to", end)
    print("Feature files:", len(files))
    print("Sampled feature rows:", len(samples))
    print("Unique markets:", samples["ticker"].nunique())

    manager = OrderBookManager()

    counts = Counter()
    exact = Counter()
    discrepancies = []
    quote_ages = []

    # Last raw update for each ticker, reset with each subscription.
    last_update = {}

    sample_pos = 0

    def compare_sample(sample):
        ticker = sample["ticker"]
        prices = manager.get_prices(ticker)

        if prices is None:
            counts["no_reconstructed_book"] += 1
            return

        quote_time = last_update.get(ticker)

        if quote_time is None:
            counts["no_raw_update"] += 1
            return

        age = (
            sample["timestamp"].to_pydatetime() - quote_time
        ).total_seconds()

        if age < 0:
            counts["timing_error"] += 1
            return

        if age > MAX_QUOTE_AGE_SECONDS:
            counts["older_than_2_seconds"] += 1

        # Continue comparing against the last valid reconstructed
        # book state, even if no update has occurred recently.

        fields = {
            "bid": "yes_bid",
            "ask": "yes_ask",
            "spread": "spread",
            "bid_depth": "yes_bid_depth",
            "ask_depth": "yes_ask_depth",
        }

        if any(prices[key] is None for key in fields.values()):
            counts["one_sided_raw_quote"] += 1
            return

        if any(
            pd.isna(sample[key]) for key in fields
        ):
            counts["missing_feature_value"] += 1
            return

        counts["comparable_rows"] += 1
        quote_ages.append(age)

        differences = {}
        row_matches = True

        for feature_field, raw_field in fields.items():
            recorded = float(sample[feature_field])
            reconstructed = float(prices[raw_field])

            difference = abs(recorded - reconstructed)
            differences[feature_field] = difference

            tolerance = (
                DEPTH_TOLERANCE
                if "depth" in feature_field
                else PRICE_TOLERANCE
            )

            if difference <= tolerance:
                exact[feature_field] += 1
            else:
                row_matches = False

        if row_matches:
            counts["all_fields_match"] += 1
        else:
            counts["rows_with_difference"] += 1

            if len(discrepancies) < 10:
                discrepancies.append({
                    "file": sample["source_file"],
                    "time": str(sample["timestamp"]),
                    "ticker": ticker,
                    "raw_age_seconds": round(age, 4),
                    "differences": {
                        key: round(value, 6)
                        for key, value in differences.items()
                    },
                    "feature_bid": float(sample["bid"]),
                    "raw_bid": float(prices["yes_bid"]),
                    "feature_ask": float(sample["ask"]),
                    "raw_ask": float(prices["yes_ask"]),
                })

    # Reconstruct the book and compare each feature row with
    # the latest raw state available at that instant.
    with RAW.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number > MAX_RAW_MESSAGES:
                break

            record = json.loads(line)
            received = parse_time(record["received_at"])

            # Process any feature samples strictly BEFORE this raw
            # message, preventing future information from leaking in.
            while (
                sample_pos < len(samples)
                and samples.iloc[sample_pos][
                    "timestamp"
                ].to_pydatetime() < received
            ):
                compare_sample(samples.iloc[sample_pos])
                sample_pos += 1

            message = record.get("message", {})
            kind = message.get("type")

            if kind == "subscribed":
                manager = OrderBookManager()
                last_update.clear()
                counts["subscription_generations"] += 1
                continue

            if kind not in (
                "orderbook_snapshot", "orderbook_delta"
            ):
                continue

            ticker = message.get("msg", {}).get(
                "market_ticker"
            )

            try:
                result = manager.process(message)
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Raw replay error at line {line_number}: "
                    f"{exc}"
                ) from exc

            if result is not None:
                last_update[ticker] = received

    # Finish samples no later than the audited raw interval end.
    while sample_pos < len(samples):
        compare_sample(samples.iloc[sample_pos])
        sample_pos += 1

    print("\n=== COMPARISON COVERAGE ===")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value:,}")

    total = counts["comparable_rows"]

    print("\n=== FIELD AGREEMENT ===")
    if total:
        for key in (
            "bid", "ask", "spread",
            "bid_depth", "ask_depth"
        ):
            print(
                f"{key}: {exact[key]}/{total} "
                f"({exact[key] / total:.2%})"
            )

        print(
            "All fields:",
            f"{counts['all_fields_match']}/{total}",
            f"({counts['all_fields_match'] / total:.2%})",
        )

        print(
            "Median raw quote age:",
            f"{np.median(quote_ages):.4f} seconds",
        )
        print(
            "90th percentile raw quote age:",
            f"{np.percentile(quote_ages, 90):.4f} seconds",
        )
        print(
            "Maximum raw quote age:",
            f"{max(quote_ages):.4f} seconds",
        )
    else:
        print("No comparable observations.")

    print("\n=== FIRST DISCREPANCIES ===")
    if discrepancies:
        for item in discrepancies:
            print(item)
    else:
        print("None")

    print("\n=== PHASE 7N.6 COMPLETE ===")
    print("Read-only replay. No orders placed.")
    print("Reserved model test data not accessed.")


if __name__ == "__main__":
    main()
