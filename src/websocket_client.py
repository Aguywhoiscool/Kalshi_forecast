
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
from src.market_ranker import (
    rank_active_markets,
    series_of,
    MAX_PER_SERIES,
)
from src.data_collector import save_raw_message
from src.orderbook import OrderBookManager
from src.feature_sampler import FeatureSampler
from src.market_activity import MarketActivityTracker



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

TARGET_MARKETS = 10
MARKET_COOLDOWN_SECONDS = 1800  # 30 minutes
REPLACEMENT_CANDIDATES = 50


class ResyncRequired(Exception):
    """Raised when the current order books cannot be trusted."""

class MarketRotationRequired(Exception):
    def __init__(self, rejected_tickers):
        self.rejected_tickers = set(rejected_tickers)
        super().__init__(
            f"Market rotation requested for {len(rejected_tickers)} markets"
        )


async def stream_markets(tickers):
    """Run one WebSocket session with fresh order books."""

    # Create a NEW manager for each connection.
    orderbook_manager = OrderBookManager()
    activity_tracker = MarketActivityTracker(
        window_seconds=300,
        min_deltas_per_minute=30
    )       

    for ticker in tickers:
        activity_tracker.register_market(ticker)
    feature_sampler = FeatureSampler()
    next_sample = time.monotonic() + 1.0

    counts = Counter()
    snapshots = set()

    connected_since = time.monotonic()
    last_report = connected_since
    last_market_review = connected_since

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
                    activity_tracker.record_delta(ticker)

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
            # Review market activity every five minutes.
            if now - last_market_review >= 300:
                print("\n=== MARKET ACTIVITY REVIEW ===")

                rejected = set()

                for ticker in tickers:
                    rate = activity_tracker.get_rate(ticker, now=now)
                    should_rotate = activity_tracker.review_market(
                        ticker, now=now
                    )

                    print(
                        f"{ticker}: "
                        f"{rate:.1f} deltas/min | "
                        f"rotate={'YES' if should_rotate else 'NO'}"
                    )

                    if should_rotate:
                        rejected.add(ticker)

                last_market_review = now

                if rejected:
                    raise MarketRotationRequired(rejected)



async def connect_to_kalshi():
    """Monitor markets and automatically rotate inactive contracts."""

    selected = await asyncio.to_thread(
        rank_active_markets,
        max_markets=TARGET_MARKETS,
    )

    if not selected:
        print("No eligible markets found.")
        return

    # Ticker -> monotonic time when cooldown expires.
    cooldowns = {}

    reconnect_delay = RECONNECT_INITIAL_SECONDS

    while True:
        tickers = [market["ticker"] for market in selected]
        connection_started = time.monotonic()
        rotation_handled = False

        try:
            await stream_markets(tickers)
            print("Stream ended. Reconnecting.")

        except MarketRotationRequired as error:
            rotation_handled = True
            now = time.monotonic()

            rejected = set(error.rejected_tickers)
            rejected &= set(tickers)

            print(
                f"\nROTATION REVIEW: "
                f"{len(rejected)} inactive markets"
            )

            # Remove expired cooldown entries.
            cooldowns = {
                ticker: expiry
                for ticker, expiry in cooldowns.items()
                if expiry > now
            }

            # Keep productive markets.
            keep = [
                market
                for market in selected
                if market["ticker"] not in rejected
            ]

            keep_tickers = {
                market["ticker"] for market in keep
            }

            # Avoid rejected and previously rotated markets.
            blocked = (
                set(cooldowns)
                | rejected
                | keep_tickers
            )

            try:
                candidates = await asyncio.to_thread(
                    rank_active_markets,
                    max_markets=REPLACEMENT_CANDIDATES,
                    excluded_tickers=blocked,
                )

            except Exception as error:
                print(
                    f"Replacement discovery failed: {error}"
                )
                candidates = []

            # Preserve the original series/event diversity
            # constraints when choosing replacements.
            from collections import Counter

            series_counts = Counter(
                series_of(market) for market in keep
            )

            used_events = {
                market.get("event_ticker") or market["ticker"]
                for market in keep
            }

            new_selected = list(keep)
            added = []

            for market in candidates:
                if len(new_selected) >= TARGET_MARKETS:
                    break

                ticker = market["ticker"]
                series = series_of(market)
                event = market.get("event_ticker") or ticker

                if ticker in blocked:
                    continue

                if series_counts[series] >= MAX_PER_SERIES:
                    continue

                if event in used_events:
                    continue

                new_selected.append(market)
                added.append(ticker)

                series_counts[series] += 1
                used_events.add(event)

            # If we cannot fill every slot, retain existing
            # inactive markets temporarily rather than
            # selecting duplicates or unsuitable candidates.
            for market in selected:
                if len(new_selected) >= TARGET_MARKETS:
                    break

                ticker = market["ticker"]

                if ticker in rejected:
                    new_selected.append(market)

            old_tickers = set(tickers)
            new_tickers = {
                market["ticker"] for market in new_selected
            }

            removed = old_tickers - new_tickers

            # Cool down only contracts actually removed.
            for ticker in removed:
                cooldowns[ticker] = (
                    time.monotonic()
                    + MARKET_COOLDOWN_SECONDS
                )

            if removed:
                print("\nMARKETS REMOVED:")
                for ticker in sorted(removed):
                    print(" -", ticker)

                print("\nNEW MARKETS:")
                for ticker in added:
                    print(" +", ticker)

                selected = new_selected

                print(
                    f"\nNow monitoring "
                    f"{len(selected)} markets."
                )
            else:
                print(
                    "No suitable replacements found. "
                    "Keeping existing markets."
                )

            # Return to a fresh WebSocket session.
            reconnect_delay = RECONNECT_INITIAL_SECONDS

        except ResyncRequired as error:
            print(f"\nRESYNC REQUIRED: {error}")

        except (
            websockets.exceptions.ConnectionClosed,
            OSError,
            asyncio.TimeoutError,
        ) as error:
            print(
                f"\nWebSocket interrupted: {error}"
            )

        if not rotation_handled:
            if time.monotonic() - connection_started >= 60:
                reconnect_delay = RECONNECT_INITIAL_SECONDS

        print(
            f"Reconnecting in {reconnect_delay} seconds..."
        )

        await asyncio.sleep(reconnect_delay)

        if not rotation_handled:
            reconnect_delay = min(
                reconnect_delay * 2,
                RECONNECT_MAX_SECONDS,
            )



if __name__ == "__main__":
    try:
        asyncio.run(connect_to_kalshi())

    except KeyboardInterrupt:
        print("\nWebSocket disconnected by user")
