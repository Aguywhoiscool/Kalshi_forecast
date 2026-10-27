
"""Record one-second order-book features to CSV."""

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.feature_engineering import extract_features


PROCESSED_DIR = Path("data/processed")

FIELDNAMES = [
    "timestamp",
    "session_id",
    "ticker",
    "bid",
    "ask",
    "spread",
    "midpoint",
    "bid_depth",
    "ask_depth",
    "imbalance",
]


class FeatureSampler:
    def __init__(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        started = datetime.now(timezone.utc)
        self.session_id = started.strftime(
            "%Y-%m-%d_%H-%M-%S_%f"
        )

        self.filepath = (
            PROCESSED_DIR
            / f"features_{self.session_id}.csv"
        )

        # Create the CSV file with its header.
        with self.filepath.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )
            writer.writeheader()

        self.samples_written = 0

        print(f"Feature recording: {self.filepath}")

    def record(self, manager, tickers):
        """Sample every valid two-sided market book."""

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        rows = []

        for ticker in tickers:
            prices = manager.get_prices(ticker)

            if prices is None:
                continue

            features = extract_features(prices)

            if features is None:
                continue

            row = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                **features,
            }

            rows.append(row)

        if not rows:
            return 0

        # Write one batch per sampling interval.
        with self.filepath.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES
            )
            writer.writerows(rows)

        self.samples_written += len(rows)

        return len(rows)
