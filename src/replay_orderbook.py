import json
from pathlib import Path
from src.orderbook import OrderBookManager


def replay_file(filepath):
    manager = OrderBookManager()

    total_messages = 0
    snapshots = 0
    deltas = 0
    errors = 0

    with open(filepath, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                # Your JSONL collector wraps the Kalshi message.
                message = record["message"]

                result = manager.process(message)

                total_messages += 1

                if message.get("type") == "orderbook_snapshot":
                    snapshots += 1

                elif message.get("type") == "orderbook_delta":
                    deltas += 1

                if result is not None and deltas % 500 == 0:
                    print(
                        f"Processed {deltas} deltas"
                    )

            except (ValueError, KeyError, TypeError) as error:
                errors += 1
                print(
                    f"Error on line {line_number}: {error}"
                )

    print("\nREPLAY RESULTS")
    print("=" * 55)
    print("Messages processed:", total_messages)
    print("Snapshots:", snapshots)
    print("Deltas:", deltas)
    print("Errors:", errors)
    print("Markets reconstructed:", len(manager.books))
    print("Invalid subscriptions:", manager.invalid_subscriptions)

    print("\nRECONSTRUCTED ORDER BOOKS")

    for ticker in manager.books:
        prices = manager.get_prices(ticker)

        if prices is None:
            print(ticker, "- Invalid or unsynchronized")
            continue

        print("-" * 55)
        print("Ticker:", ticker)
        print("YES bid:", prices["yes_bid"])
        print("YES ask:", prices["yes_ask"])
        print("Spread:", prices["spread"])
        print("Midpoint:", prices["midpoint"])
        print("Bid depth:", prices["yes_bid_depth"])
        print("Ask depth:", prices["yes_ask_depth"])

    return manager


if __name__ == "__main__":
    raw_directory = Path("data/raw")

    files = list(raw_directory.glob("*.jsonl"))

    if not files:
        print("No recorded market data found.")
    else:
        latest_file = max(
            files,
            key=lambda path: path.stat().st_mtime
        )

        print("Replaying:", latest_file)
        replay_file(latest_file)
