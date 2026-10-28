
"""Track rolling market activity for automatic market rotation."""

import time
from collections import defaultdict, deque


class MarketActivityTracker:
    def __init__(self, window_seconds=300, min_deltas_per_minute=60):
        self.window_seconds = window_seconds
        self.min_deltas_per_minute = min_deltas_per_minute

        self.events = defaultdict(deque)
        self.started_at = {}
        self.failed_reviews = defaultdict(int)

    def register_market(self, ticker, now=None):
        now = time.monotonic() if now is None else now
        self.started_at[ticker] = now
        self.events[ticker].clear()
        self.failed_reviews[ticker] = 0

    def record_delta(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            self.register_market(ticker, now)

        self.events[ticker].append(now)
        self._prune(ticker, now)

    def _prune(self, ticker, now):
        cutoff = now - self.window_seconds
        events = self.events[ticker]

        while events and events[0] < cutoff:
            events.popleft()

    def get_rate(self, ticker, now=None):
        now = time.monotonic() if now is None else now
        self._prune(ticker, now)

        if ticker not in self.started_at:
            return 0.0

        elapsed = min(
            self.window_seconds,
            max(now - self.started_at[ticker], 1),
        )

        return len(self.events[ticker]) * 60 / elapsed

    def review_market(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            return False

        # Give each new market time to establish activity.
        if now - self.started_at[ticker] < 300:
            return False

        rate = self.get_rate(ticker, now)

        if rate < self.min_deltas_per_minute:
            self.failed_reviews[ticker] += 1
        else:
            self.failed_reviews[ticker] = 0

        return self.failed_reviews[ticker] >= 1
