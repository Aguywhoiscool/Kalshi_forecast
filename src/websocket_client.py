
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
from src.live_market_status import get_live_market_status
from src.market_rotation import apply_rotation_fallback
from src.observation_coordinator import ObservationCoordinator
from src.activity_candidate_pool import (
    prioritize_rotation_candidates,
)
from src.spread_quality_tracker import SpreadQualityTracker
from src.quote_quality import classify_quote
from src.one_sided_quote_tracker import OneSidedQuoteTracker

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

# Activity-aware candidate observation.
# Shadow mode observes replacements but does not select them.
ACTIVITY_OBSERVER_ENABLED = (
    os.getenv("ACTIVITY_OBSERVER_ENABLED", "0") == "1"
)
# Controls whether activity-qualified candidates affect rotation.
# Defaults to off, even when the observer is running.
ACTIVITY_ROTATION_ENABLED = (
    os.getenv("ACTIVITY_ROTATION_ENABLED", "0") == "1"
)
class ResyncRequired(Exception):
    """Raised when the current order books cannot be trusted."""


class MarketRotationRequired(Exception):
    def __init__(self, rejected_tickers, reason="inactive"):
        self.rejected_tickers = set(rejected_tickers)
        self.reason = reason

        super().__init__(
            f"Market rotation requested for "
            f"{len(self.rejected_tickers)} markets "
            f"(reason={reason})"
        )



async def stream_markets(tickers):
    """Run one WebSocket session with fresh order books."""

    # Create a NEW manager for each connection.
    orderbook_manager = OrderBookManager()
    activity_tracker = MarketActivityTracker(
        window_seconds=300,
        min_deltas_per_minute=30
    )
    spread_tracker = SpreadQualityTracker(
        max_spread_cents=10.0,
        persistence_seconds=60.0,
        max_update_age_seconds=15.0,
    )   

    for ticker in tickers:
        activity_tracker.register_market(ticker)
    feature_sampler = FeatureSampler()
    next_sample = time.monotonic() + 1.0
    one_sided_tracker = OneSidedQuoteTracker(
    persistence_seconds=60.0
    )
    last_sample_time = None
    sampling_delays = 0
    largest_sampling_gap = 0.0
    counts = Counter()
    snapshots = set()

    connected_since = time.monotonic()
    last_report = connected_since
    last_market_review = connected_since
    last_status_check = connected_since
    STATUS_CHECK_SECONDS = 60
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
                # Track genuine midpoint changes.
                    if prices is not None:
                        activity_tracker.record_midpoint(
                            ticker,
                            prices["midpoint"],
                        )

                        # Order-book spreads are denominated in dollars.
                        # Convert to cents for SpreadQualityTracker.
                        spread = prices.get("spread")

                        spread_tracker.record_spread(
                            ticker,
                            (
                                float(spread) * 100
                                if spread is not None
                                else None
                            ),
                        )
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

                if last_sample_time is not None:
                    sample_gap = now - last_sample_time

                    if sample_gap > 1.5:
                        sampling_delays += 1
                        largest_sampling_gap = max(
                            largest_sampling_gap,
                            sample_gap,
                        )

                        print(
                            f"\nSAMPLING DELAY: "
                            f"{sample_gap:.2f}s between samples | "
                            f"total_delays={sampling_delays} | "
                            f"largest={largest_sampling_gap:.2f}s"
                        )

                last_sample_time = now
                
                # Inspect quote availability once per sampling tick.
                # Do not start a one-sided timer before the first snapshot.
                for ticker in tickers:
                    prices = orderbook_manager.get_prices(ticker)

                    one_sided_tracker.update(
                        ticker=ticker,
                        has_snapshot=ticker in snapshots,
                        prices=prices,
                        now=now,
                    )

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

            
            # Independently check whether monitored markets
            # are still open and sufficiently far from closing.
            if now - last_status_check >= STATUS_CHECK_SECONDS:
                print("\n=== LIVE MARKET STATUS REVIEW ===")

                
                closed_markets = set()

                # Run REST requests concurrently rather than
                # waiting for each market sequentially.
                status_results = await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            get_live_market_status,
                            ticker,
                        )
                        for ticker in tickers
                    ],
                    return_exceptions=True,
                )

                for ticker, status in zip(tickers, status_results):

                    if isinstance(status, BaseException):
                        print(
                            f"{ticker}: STATUS UNKNOWN ({status})"
                        )
                        continue

                    if not status["tradable"]:
                        closed_markets.add(ticker)

                        print(
                            f"{ticker}: NOT TRADABLE | "
                            f"status={status['status']} | "
                            f"close={status['close_time']}"
                        )


                last_status_check = time.monotonic()

                if closed_markets:
                    print(
                        f"Lifecycle rotation requested for "
                        f"{len(closed_markets)} markets"
                    )
                    raise MarketRotationRequired(
                        closed_markets,
                        reason="not_tradable"
                    )

            if now - last_report >= REPORT_SECONDS:

                duration = max(
                    now - connected_since,
                    1,
                )

                quote_quality = {
                    ticker: classify_quote(
                        orderbook_manager.get_prices(ticker)
                    )
                    for ticker in tickers
                }

                valid_books = sum(
                    quality["structurally_valid"]
                    for quality in quote_quality.values()
                )

                two_sided_books = sum(
                    quality["two_sided"]
                    for quality in quote_quality.values()
                )

                training_books = sum(
                    quality["training_quality"]
                    for quality in quote_quality.values()
                )

                print(
                    f"\n=== Activity report: "
                    f"{duration:.0f} seconds ==="
                )

                print(
                    f"Structurally valid books: "
                    f"{valid_books}/{len(tickers)}"
                )

                print(
                    f"Two-sided quoted books: "
                    f"{two_sided_books}/{len(tickers)}"
                )

                print(
                    f"Training-quality books (spread <= 10c): "
                    f"{training_books}/{len(tickers)}"
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
                spread_rejected = set()
                one_sided_rejected = set()

                for ticker in tickers:
                    rate = activity_tracker.get_rate(
                        ticker, now=now
                    )

                    changes = activity_tracker.get_midpoint_changes(
                        ticker, now=now
                    )
                    meaningful_changes = (
                        activity_tracker.get_meaningful_midpoint_changes(
                            ticker, now=now
                        )
                    )

                    should_rotate = activity_tracker.review_market(
                        ticker, now=now
                    )

                    spread_status = spread_tracker.get_status(
                        ticker, now=now
                    )
                    one_sided_status = one_sided_tracker.get_status(ticker)

                    one_sided_rotation = one_sided_status["should_rotate"]

                    print(
                        f"  QUOTE CHECK {ticker}: "
                        f"one_sided={one_sided_status['one_sided']} | "
                        f"seconds={one_sided_status['one_sided_seconds']:.1f} | "
                        f"rotate={'YES' if one_sided_rotation else 'NO'}"
                    )
                    wide_spread_rotation = spread_status["should_rotate"]
                    if (
                        ACTIVITY_ROTATION_ENABLED
                        and wide_spread_rotation
                    ):
                        should_rotate = True
                        spread_rejected.add(ticker)
                    spread_value = spread_status["spread_cents"]

                    spread_text = (
                        f"{spread_value:.1f}c"
                        if spread_value is not None
                        else "unavailable"
                    )

                    print(
                        f"  SPREAD CHECK {ticker}: "
                        f"spread={spread_text} | "
                        f"wide_seconds={spread_status['wide_seconds']:.0f} | "
                        f"stale={spread_status['stale']} | "
                        f"wide_rotation={'YES' if wide_spread_rotation else 'NO'}"
                    )
                    if ACTIVITY_ROTATION_ENABLED and one_sided_rotation:
                        should_rotate = True
                        one_sided_rejected.add(ticker)
                    if ticker in spread_rejected:
                        status = "PERSISTENT WIDE SPREAD"
                    elif should_rotate:
                        if rate >= 30:
                            status = "HIGH DELTAS / NO PRICE MOVEMENT"
                        else:
                            status = "INACTIVE / NO PRICE MOVEMENT"
                    elif rate < 30:
                        status = "LOW DELTAS / PRICE MOVING"
                    else:
                        status = "ACTIVE"

                    print(
                        f"{ticker}: "
                        f"{rate:.1f} deltas/min | "
                        f"{changes} midpoint changes | "
                        f"{meaningful_changes} meaningful changes (>=0.5c) | "
                        f"{status} | "
                        f"rotate={'YES' if should_rotate else 'NO'}"
                    )
                    
                    if should_rotate:
                        rejected.add(ticker)

                last_market_review = now

                if rejected:
                    if spread_rejected:
                        reason = "wide_spread"
                    elif one_sided_rejected:
                        reason = "one_sided"
                    else:
                        reason = "inactive"

                    raise MarketRotationRequired(
                        rejected,
                        reason=reason,
                    )



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
    coordinator = (
        ObservationCoordinator(
            candidate_count=80,
            batch_size=20,
            observation_seconds=120,
        )
        if (
        ACTIVITY_OBSERVER_ENABLED
        or ACTIVITY_ROTATION_ENABLED
        )
        else None
    )
    reconnect_delay = RECONNECT_INITIAL_SECONDS

    while True:
        tickers = [market["ticker"] for market in selected]
        connection_started = time.monotonic()
        rotation_handled = False
        
        if coordinator is not None:

            def get_observer_exclusions():
                now = time.monotonic()

                exclusions = {
                    ticker
                    for ticker, expiry in cooldowns.items()
                    if expiry > now
                }

                exclusions.update(
                    market["ticker"]
                    for market in selected
                )

                return exclusions

            started = coordinator.start_continuous(
                get_excluded_tickers=get_observer_exclusions
            )

            if started:
                print(
                    "Continuous activity observer started."
                )

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
                f"{len(rejected)} markets flagged for rotation "
                f"(reason={error.reason})"
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

            candidates = []
            qualified_tickers = set()

            if ACTIVITY_ROTATION_ENABLED and coordinator is not None:
                # Prefer the already discovered candidate pool.
                shared_candidates = coordinator.get_candidates()
                fresh = coordinator.cache.fresh_observations(
                    [m["ticker"] for m in shared_candidates]
                )

                observations = (
                    fresh.values()
                    if isinstance(fresh, dict)
                    else fresh
                )

                moving_count = sum(
                    obs.get("midpoint_changes", 0) > 0
                    for obs in observations
                )

                print(
                    "\nOBSERVER AVAILABILITY:"
                    f"\n  Shared candidates: {len(shared_candidates)}"
                    f"\n  Fresh observations: {len(fresh)}"
                    f"\n  Moving markets: {moving_count}"
                )
                if shared_candidates:
                    try:
                        priority = prioritize_rotation_candidates(
                            candidates=shared_candidates,
                            cache=coordinator.cache,
                            existing_markets=keep,
                            blocked_tickers=blocked,
                            target_markets=TARGET_MARKETS,
                        )

                        candidates = priority["candidates"]
                        qualified_tickers = priority["qualified_tickers"]

                        print(
                            "\nACTIVITY-AWARE ROTATION:"
                            f"\n  Shared REST candidates: {len(shared_candidates)}"
                            f"\n  Preferred spread candidates: {priority['preferred_count']}"
                            f"\n  Secondary spread candidates: {priority['secondary_count']}"
                            f"\n  Rejected for spread: {priority['spread_rejected_count']}"
                            f"\n  Qualified replacements: {len(qualified_tickers)}"
                            f"\n  Activity shortage: {priority['activity_shortage']}"
                        )

                    except Exception as selection_error:
                        print(
                            "Activity candidate selection failed: "
                            f"{selection_error}"
                        )
                        candidates = []

                if coordinator.last_error:
                    print(
                        "Activity observer error: "
                        f"{coordinator.last_error}"
                    )

            # Existing REST discovery remains the safety fallback.
            if not candidates:
                print(
                    "Using standard REST replacement discovery."
                )

                try:
                    candidates = await asyncio.to_thread(
                        rank_active_markets,
                        max_markets=REPLACEMENT_CANDIDATES,
                        excluded_tickers=blocked,
                    )

                except Exception as discovery_error:
                    print(
                        "Replacement discovery failed: "
                        f"{discovery_error}"
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
                # Revalidate before adding a cached candidate.
                if ACTIVITY_ROTATION_ENABLED:
                    try:
                        live_status = await asyncio.to_thread(
                            get_live_market_status,
                            ticker,
                        )
                    except Exception as status_error:
                        print(
                            f"Skipping {ticker}: "
                            f"live status unavailable ({status_error})"
                        )
                        continue

                    if not live_status["tradable"]:
                        print(
                            f"Skipping {ticker}: no longer tradable"
                        )
                        continue
                    current_spread = live_status.get("spread_cents")

                    if current_spread is None:
                        print(
                            f"Skipping {ticker}: current spread unavailable"
                        )
                        continue

                    if current_spread > 10.0:
                        print(
                            f"Skipping {ticker}: current spread "
                            f"{current_spread:.1f}c exceeds 10c"
                        )
                        continue
                if ticker in qualified_tickers:
                    print(
                        f"ACTIVITY-QUALIFIED REPLACEMENT: {ticker}"
                    )
                elif ACTIVITY_ROTATION_ENABLED:
                    print(
                        f"REST FALLBACK REPLACEMENT: {ticker}"
                    )

                new_selected.append(market)
                added.append(ticker)

                series_counts[series] += 1
                used_events.add(event)

            # If we cannot fill every slot, retain existing
            
            # Only allow inactive (but tradable) markets
            # to remain if replacements are unavailable.
            if error.reason in (
                "inactive",
                "not_tradable",
                "wide_spread",
                "one_sided",
            ):
                
                # Apply tested fallback safety rules.
                new_selected = apply_rotation_fallback(
                    new_selected=new_selected,
                    previous_selected=selected,
                    rejected=rejected,
                    reason=error.reason,
                    target_markets=TARGET_MARKETS,
                )


            else:
                raise ValueError(
                    f"Unknown rotation reason: {error.reason}"
                )


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
                    if ticker in new_tickers:
                        print(" +", ticker)

            else:
                print(
                    "No market replacements were available."
                )

            # Always update the selection after rotation.
            selected = new_selected

            print(
                f"\nNow monitoring {len(selected)} markets."
            )

            # Do not reconnect with an empty market selection.
            if not selected:
                print(
                    "No eligible markets remain. "
                    "Searching for new markets in 60 seconds..."
                )

                await asyncio.sleep(60)

                selected = await asyncio.to_thread(
                    rank_active_markets,
                    max_markets=TARGET_MARKETS,
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
