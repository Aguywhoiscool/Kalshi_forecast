"""Read-only Kalshi WebSocket feed for in-memory dashboard observations."""

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import websockets

from src.authentication import get_auth_headers
from src.causal_swing_data import history_ends
from src.feature_engineering import extract_features
from src.orderbook import OrderBookManager
from src.v3_research import FEATURES, overlaps


class KalshiFeed:
    def __init__(self, tickers=None, rescan_seconds=600, min_close_minutes=10):
        if rescan_seconds < 1 or min_close_minutes < 3:
            raise ValueError("Positive rescan interval and at least three minutes until close required")
        self.requested = tickers
        self.rescan_seconds = rescan_seconds
        self.min_close_minutes = min_close_minutes
        self.close_times = {}
        self.lock = threading.Lock()
        self.state = {"status": "connecting", "snapshots": 0, "messages": 0,
                      "warmup_samples": 0, "reconnects": 0, "error": None,
                      "rescan_seconds": rescan_seconds, "min_close_minutes": min_close_minutes,
                      "stop_before_close_seconds": 90}
        self.reset()

    def reset(self):
        with self.lock:
            self.manager = OrderBookManager()
            self.histories = {}
            self.session = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
            self.state.update(warmup_samples=0, snapshots=0)

    def event(self, event):
        kind = event.get("type")
        if kind == "subscribed":
            self.reset()  # Subscription IDs are reused; never stitch books/history.
        elif kind in ("orderbook_snapshot", "orderbook_delta"):
            self.manager.process(event)
            if self.manager.invalid_subscriptions:
                raise ValueError("Order-book sequence gap; fresh connection required")
            with self.lock:
                self.state["snapshots"] += int(kind == "orderbook_snapshot")
        elif kind == "error":
            # Keep exchange payload/authentication details out of browser/logs.
            raise ValueError("Kalshi rejected the market-data subscription")
        with self.lock:
            self.state["messages"] += 1
            self.state["last_message_at"] = int(time.time()*1000)

    def sample(self, tickers, stamp=None):
        stamp = time.time_ns() if stamp is None else stamp
        with self.lock:
            for ticker in tickers:
                too_close = stamp/1e9 >= self.close_times.get(ticker, float("inf"))-90
                features = None if too_close else extract_features(self.manager.get_prices(ticker))
                rows = self.histories.setdefault(ticker, deque(maxlen=60))
                if features is None:
                    rows.clear()
                    continue
                if rows and not .5e9 <= stamp-rows[-1][0] <= 1.5e9:
                    rows.clear()
                rows.append((stamp, [features[name] for name in FEATURES]))
            self.state["warmup_samples"] = max((len(h) for h in self.histories.values()), default=0)

    def latest(self, protected):
        items = []
        with self.lock:
            health = dict(self.state)
            if self.state["status"] == "connected":
                for ticker, rows in self.histories.items():
                    if time.time() >= self.close_times.get(ticker, float("inf"))-90:
                        continue
                    if len(rows) != 60 or self.session in protected["sessions"]:
                        continue
                    clocks = np.array([row[0] for row in rows], dtype=np.int64)
                    if overlaps(clocks[0], clocks[-1], protected):
                        continue
                    if history_ends(clocks, np.ones(60, bool)).tolist() != [59]:
                        continue
                    items.append({"ticker": ticker, "session_id": self.session, "input_end": int(clocks[-1]),
                                  "clocks": clocks, "history": np.array([row[1] for row in rows], float)})
        errors = [{"message": health["error"]}] if health.get("error") else []
        return items, errors, health

    async def run(self):
        delay = 2
        while True:
            try:
                with self.lock:
                    self.state.update(status="discovering", error=None)
                if self.requested:
                    tickers = self.requested[:8]
                else:
                    from src.market_ranker import rank_active_markets
                    markets = await asyncio.to_thread(rank_active_markets, max_markets=8, max_pages=10,
                        min_minutes_to_close=self.min_close_minutes)
                    tickers = [m["ticker"] for m in markets]
                    self.close_times = {m["ticker"]: datetime.fromisoformat(
                        m["close_time"].replace("Z", "+00:00")).timestamp() for m in markets}
                if not tickers:
                    raise ValueError("No eligible open markets found")
                auth = await asyncio.to_thread(get_auth_headers, "GET", "/trade-api/ws/v2")
                url = os.getenv("KALSHI_WS_URL", "wss://external-api-ws.kalshi.com/trade-api/ws/v2")
                async with websockets.connect(url, additional_headers=auth, ping_interval=20,
                        ping_timeout=10, open_timeout=15, max_size=8*1024*1024) as ws:
                    self.reset()
                    with self.lock:
                        self.state.update(status="connected", tickers=tickers, error=None,
                                          next_rescan_at=int((time.time()+self.rescan_seconds)*1000))
                    await ws.send(json.dumps({"id": 1, "cmd": "subscribe", "params": {
                        "channels": ["orderbook_delta"], "market_tickers": tickers}}))
                    started = time.monotonic()
                    next_sample, next_ping = started+1, started+15
                    delay = 2
                    while time.monotonic()-started < self.rescan_seconds:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=max(.001, next_sample-time.monotonic()))
                        except asyncio.TimeoutError:
                            raw = None
                        if raw is not None:
                            self.event(json.loads(raw))
                        now = time.monotonic()
                        if now >= next_ping:
                            pong = await ws.ping()
                            await asyncio.wait_for(pong, timeout=5)
                            next_ping = time.monotonic()+15
                        if now >= next_sample:
                            self.sample(tickers)
                            next_sample = time.monotonic()+1
                # Refresh market selection periodically, before expected closes.
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Exception text from auth/network libraries can contain headers.
                message = str(error) if type(error) is ValueError and str(error) in (
                    "No eligible open markets found", "Kalshi rejected the market-data subscription",
                    "Order-book sequence gap; fresh connection required") else f"Live feed interrupted ({type(error).__name__}); retrying"
                with self.lock:
                    self.state.update(status="reconnecting", error=message)
                print(message, flush=True)
            finally:
                self.reset()
                with self.lock:
                    self.state["status"] = "reconnecting"
                    self.state["reconnects"] += 1
            await asyncio.sleep(delay)
            delay = min(30, delay*2)
