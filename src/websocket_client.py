
"""Read-only multi-market Kalshi streaming monitor.

Run: python -m src.websocket_client

Includes:
- Market discovery and ranking
- Raw JSONL message recording
- Live order-book reconstruction
- 30-second market activity reports
- Automatic reconnection and resynchronization
"""

import asyncio
import json
import os
import time
from collections import Counter

import websockets
from dotenv import load_dotenv

from src.authentication import get_auth_headers
from src.market_ranker import rank_active_markets
from src.data_collector import save_raw_message
from src.orderbook import OrderBookManager
from src.feature_sampler import FeatureSampler


load_dotenv()

WS_URL = os.getenv(
    "KALSHI_WS_URL",
    "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
)

REPORT_SECONDS = 30
RECONNECT_INITIAL_SECONDS = 2
RECONNECT_MAX_SECONDS = 30

# Set to 15 during testing to force a reconnect.
# Keep at 0 for normal operation.
TEST_RECONNECT_AFTER_SECONDS = 0


class ResyncRequired(Exception):
    """Raised when the current order books cannot be trusted."""


async def stream_markets(tickers):
    """Run one WebSocket session with fresh order books."""

    # Create a NEW manager for each connection.
    orderbook_manager = OrderBookManager()
    feature_sampler = FeatureSampler()
    next_sample = time.monotonic() + 1.0

    counts = Counter()
    snapshots = set()

    connected_since = time.monotonic()
    last_report = connected_since

    # Authentication must be regenerated for each connection.
    auth = get_auth_headers("GET", "/trade-api/ws/v2")

    print(f"\nConnecting to {len(tickers)} markets...")

    async with websockets.connect(
        WS_URL,
        additional_headers=auth,
        ping_interval=20,
        ping_timeout=20,
        max_size=8 * 1024 * 1024,
        open_timeout=15,
    ) as ws:

        await ws.send(json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }))

        print(
            "Subscription sent. Waiting for confirmation "
            "and fresh snapshots."
        )

        while True:

            # Optional forced reconnection for testing.
            elapsed = time.monotonic() - connected_since

            if (
                TEST_RECONNECT_AFTER_SECONDS > 0
                and elapsed >= TEST_RECONNECT_AFTER_SECONDS
            ):
                raise ResyncRequired(
                    "Test reconnect timer reached"
                )

            try:
                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=max(0.001, min(1, next_sample - time.monotonic())),
                )
            except asyncio.TimeoutError:
                raw = None

            if raw is not None:

                event = json.loads(raw)

                msg = event.get("msg") or {}
                kind = event.get("type")
                ticker = msg.get("market_ticker")

                # Preserve every received message.
                save_raw_message(
                    event,
                    market_ticker=ticker,
                )

                # Update the reconstructed order books.
                if kind in (
                    "orderbook_snapshot",
                    "orderbook_delta",
                ):
                    try:
                        prices = orderbook_manager.process(event)

                    except (ValueError, KeyError, TypeError) as error:
                        raise ResyncRequired(
                            f"Order-book processing error: {error}"
                        ) from error

                    # Manager detects missing/out-of-order sequences.
                    if orderbook_manager.invalid_subscriptions:
                        raise ResyncRequired(
                            "Sequence gap detected on subscription(s): "
                            f"{orderbook_manager.invalid_subscriptions}"
                        )

                    if prices is not None and kind == "orderbook_snapshot":
                        print(
                            f"{ticker} | "
                            f"Bid: {prices['yes_bid']} | "
                            f"Ask: {prices['yes_ask']} | "
                            f"Spread: {prices['spread']} | "
                            f"Midpoint: {prices['midpoint']}"
                        )

                # Preserve market-activity monitoring.
                if kind == "subscribed":
                    print("Subscription confirmed:", msg)

                elif kind == "orderbook_snapshot":
                    snapshots.add(ticker)

                    print(
                        "Snapshot:",
                        ticker,
                        "YES levels:",
                        len(
                            msg.get(
                                "yes_dollars_fp",
                                msg.get("yes_dollars", []),
                            )
                        ),
                        "NO levels:",
                        len(
                            msg.get(
                                "no_dollars_fp",
                                msg.get("no_dollars", []),
                            )
                        ),
                    )

                elif kind == "orderbook_delta":
                    counts[ticker] += 1

                elif kind == "error":
                    raise ResyncRequired(
                        f"Kalshi subscription error: {msg}"
                    )

                elif kind not in ("heartbeat",):
                    print("Message:", kind)

            # Produce activity summaries every 30 seconds.
            now = time.monotonic()
            
            # Save one-second features from reconstructed books.
            if now >= next_sample:
                feature_sampler.record(
                    orderbook_manager,
                    tickers
                )

                # Advance the sampling schedule.
                next_sample += 1.0

                # If processing was delayed, skip missed intervals
                # rather than creating misleading duplicate samples.
                if next_sample <= now:
                    next_sample = now + 1.0


            if now - last_report >= REPORT_SECONDS:

                duration = max(
                    now - connected_since,
                    1,
                )

                valid_books = sum(
                    orderbook_manager.get_prices(ticker) is not None
                    for ticker in tickers
                )

                print(
                    f"\n=== Activity report: "
                    f"{duration:.0f} seconds ==="
                )

                print(
                    f"Valid order books: "
                    f"{valid_books}/{len(tickers)}"
                )

                for ticker in tickers:
                    count = counts[ticker]

                    print(
                        f"{ticker}: "
                        f"{count} deltas "
                        f"({count * 60 / duration:.1f}/min), "
                        f"snapshot="
                        f"{'yes' if ticker in snapshots else 'no'}"
                    )

                last_report = now


async def connect_to_kalshi():
    """Discover markets and reconnect after stream failures."""

    selected = await asyncio.to_thread(
        rank_active_markets
    )

    tickers = [
        market["ticker"]
        for market in selected
    ]

    if not tickers:
        print(
            "No markets passed filters. "
            "Adjust discovery settings."
        )
        return

    reconnect_delay = RECONNECT_INITIAL_SECONDS

    while True:

        connection_started = time.monotonic()

        try:
            await stream_markets(tickers)

            # A clean return is unexpected for our
            # continuously running subscription.
            print("Stream ended. Reconnecting.")

        except ResyncRequired as error:
            print(f"\nRESYNC REQUIRED: {error}")

        except (
            websockets.exceptions.ConnectionClosed,
            OSError,
            asyncio.TimeoutError,
        ) as error:
            print(
                f"\nWebSocket connection interrupted: {error}"
            )

        # Don't keep increasing the delay if a connection
        # successfully ran for a substantial period.
        if time.monotonic() - connection_started >= 60:
            reconnect_delay = RECONNECT_INITIAL_SECONDS

        print(
            f"Reconnecting in {reconnect_delay} seconds..."
        )

        await asyncio.sleep(reconnect_delay)

        reconnect_delay = min(
            reconnect_delay * 2,
            RECONNECT_MAX_SECONDS,
        )


if __name__ == "__main__":
    try:
        asyncio.run(connect_to_kalshi())

    except KeyboardInterrupt:
        print("\nWebSocket disconnected by user")
