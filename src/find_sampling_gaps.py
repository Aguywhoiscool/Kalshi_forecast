
"""Find the largest within-session sampling gaps."""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main():
    largest = []

    for filepath in sorted(
        Path("data/processed").glob("features_*.csv")
    ):
        market_times = defaultdict(list)

        with filepath.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            for row in csv.DictReader(file):
                market_times[row["ticker"]].append(
                    datetime.fromisoformat(row["timestamp"])
                )

        for ticker, times in market_times.items():
            times = sorted(set(times))

            for previous, current in zip(
                times, times[1:]
            ):
                seconds = (
                    current - previous
                ).total_seconds()

                if seconds > 1.5:
                    largest.append(
                        (
                            seconds,
                            filepath.name,
                            ticker,
                            previous.isoformat(),
                            current.isoformat(),
                        )
                    )

    largest.sort(reverse=True)

    print("\n=== TOP 20 SAMPLING GAPS ===")

    for seconds, filename, ticker, start, end in largest[:20]:
        print(f"\nGap: {seconds:.2f} seconds")
        print(f"File: {filename}")
        print(f"Market: {ticker}")
        print(f"From: {start}")
        print(f"To: {end}")


if __name__ == "__main__":
    main()
