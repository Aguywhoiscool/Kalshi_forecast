
"""Observe candidate markets without changing the live collector.

Run:
    python -m src.market_observer

Read-only. Does not submit orders or write training CSVs.
"""

import asyncio
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from statistics import median
import websockets
from dotenv import load_dotenv

from src.authentication import get_auth_headers
from src.market_ranker import rank_active_markets
from src.orderbook import OrderBookManager
from src.market_activity import MarketActivityTracker
from src.activity_market_ranker import (
    calculate_activity_score,
    select_active_markets,
)
from src.market_observation_cache import MarketObservationCache
from src.market_probe_batch import MarketProbeBatch
from src.observation_quality import BookCoverageTracker

load_dotenv()

WS_URL = os.getenv(
    "KALSHI_WS_URL",
    "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
)

# Start conservatively before probing 30-50 markets.
CANDIDATE_COUNT = 80
BATCH_SIZE = 20
OBSERVATION_SECONDS = 120
TARGET_MARKETS = 10

OUTPUT_FILE = Path("reports/market_observer/results.json")


async def observe_markets(tickers, duration_seconds=90):
    """Observe live order books on a separate WebSocket connection."""

    manager = OrderBookManager()
    tracker = MarketActivityTracker(
        window_seconds=max(300, duration_seconds + 10)
    )

    for ticker in tickers:
        tracker.register_market(ticker)

    delta_counts = {ticker: 0 for ticker in tickers}
    first_valid_time = {}
    latest_spread = {}
    spread_samples = {ticker: [] for ticker in tickers}
    snapshots = set()

    auth = get_auth_headers("GET", "/trade-api/ws/v2")

    start = time.monotonic()
    coverage_tracker = BookCoverageTracker(
        tickers,
        start_time=start,
    )

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

        print(f"Observing {len(tickers)} candidates...")

        while True:
            remaining = duration_seconds - (
                time.monotonic() - start
            )

            if remaining <= 0:
                break

            try:
                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=min(remaining, 1.0),
                )
            except asyncio.TimeoutError:
                continue

            event = json.loads(raw)
            kind = event.get("type")
            msg = event.get("msg") or {}
            ticker = msg.get("market_ticker")

            if kind == "error":
                raise RuntimeError(
                    f"Observer subscription error: {event}"
                )

            if kind not in (
                "orderbook_snapshot",
                "orderbook_delta",
            ):
                continue

            if ticker not in delta_counts:
                continue

            try:
                prices = manager.process(event)
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Observer order-book error: {exc}"
                ) from exc

            if manager.invalid_subscriptions:
                raise RuntimeError(
                    "Observer detected an order-book sequence gap: "
                    f"{manager.invalid_subscriptions}"
                )

            if kind == "orderbook_snapshot":
                snapshots.add(ticker)

            if kind == "orderbook_delta" and prices is not None:
                delta_counts[ticker] += 1
                tracker.record_delta(ticker)

            if prices is None:
                coverage_tracker.update(
                    ticker,
                    False,
                    now=time.monotonic(),
                )
                tracker.record_midpoint(ticker, None)
                latest_spread.pop(ticker, None)
                continue
            coverage_tracker.update(
                ticker,
                True,
                now=time.monotonic(),
            )
            tracker.record_midpoint(
                ticker, prices["midpoint"]
            )

            if (
                prices["yes_bid"] is not None
                and prices["yes_ask"] is not None
            ):
                latest_spread[ticker] = (
                    Decimal(str(prices["yes_ask"]))
                    - Decimal(str(prices["yes_bid"]))
                ) * 100
                if latest_spread[ticker] > 0:
                    spread_samples[ticker].append(
                        float(latest_spread[ticker])
                    )
                first_valid_time.setdefault(
                    ticker, time.monotonic()
                )

    end = time.monotonic()
    results = {}

    for ticker in tickers:
        valid = ticker in latest_spread
        observed_from = first_valid_time.get(ticker)

        observed_seconds = (
            max(0, end - observed_from)
            if observed_from is not None
            else 0
        )
        coverage = coverage_tracker.summarize(
            ticker,
            end_time=end,
        )

        results[ticker] = {
            "ticker": ticker,
            "valid_seconds": coverage["valid_seconds"],
            "total_seconds": coverage["total_seconds"],
            "valid_coverage": coverage["valid_coverage"],
            "first_valid_delay": coverage["first_valid_delay"],
            "duration_seconds": coverage["valid_seconds"],
            "delta_count": delta_counts[ticker],
            "midpoint_changes": (
                tracker.get_midpoint_changes(ticker, now=end)
            ),
            "meaningful_changes": (
                tracker.get_meaningful_midpoint_changes(
                    ticker, now=end
                )
            ),
            "spread_cents": (
                median(spread_samples[ticker])
                if spread_samples[ticker] else None
            ),
            "last_spread_cents": (
                float(latest_spread[ticker])
                if valid else None
            ),
            "spread_sample_count": len(spread_samples[ticker]),
            "valid_book": valid,
            "snapshot_received": ticker in snapshots,
        }

    return results



async def main():
    cache = MarketObservationCache()
    probe = MarketProbeBatch()

    print("\nDiscovering REST candidates...")

    candidates = await asyncio.to_thread(
        rank_active_markets,
        max_markets=CANDIDATE_COUNT,
    )

    if not candidates:
        print("No eligible candidates found.")
        return

    tickers = [market["ticker"] for market in candidates]

    # Existing observations still within their five-minute TTL.
    cached = cache.fresh_observations(tickers)

    print("REST candidates:", len(candidates))
    print("Recently observed candidates:", len(cached))

    # Select a rotating batch, skipping fresh cache entries.
    batch = probe.next_batch(
        candidates,
        cache,
        batch_size=BATCH_SIZE,
    )

    print("New candidates to observe:", len(batch))

    if batch:
        batch_tickers = [
            market["ticker"] for market in batch
        ]

        new_observations = await observe_markets(
            batch_tickers,
            duration_seconds=OBSERVATION_SECONDS,
        )

        # Cache only after observation completes successfully.
        cache.update(new_observations)
    else:
        print("No uncached candidates available.")

    # Use only observations still within the five-minute TTL.
    observations = cache.fresh_observations(tickers)

    print("\n=== CACHED AND NEW OBSERVATIONS ===")

    ranked = sorted(
        observations.values(),
        key=calculate_activity_score,
        reverse=True,
    )

    for item in ranked:
        score = calculate_activity_score(item)

        print(
            f"{item['ticker']} | "
            f"deltas={item['delta_count']} | "
            f"changes={item['midpoint_changes']} | "
            f"meaningful={item['meaningful_changes']} | "
            f"spread={item['spread_cents']}c | "
            f"score={score:.3f}"
        )

    selected = select_active_markets(
        candidates,
        observations,
        target_markets=TARGET_MARKETS,
    )

    print("\n=== TOP OBSERVED MARKETS ===")

    if not selected:
        print("No markets demonstrated midpoint movement.")

    for market in selected:
        print(market["ticker"])

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "candidate_count": len(candidates),
                "probed_this_run": len(batch),
                "observations": observations,
                "selected_tickers": [
                    market["ticker"] for market in selected
                ],
            },
            file,
            indent=2,
        )

    print("\nSaved:", OUTPUT_FILE)
    print("Main collector was not modified.")

    

if __name__ == "__main__":
    asyncio.run(main())
