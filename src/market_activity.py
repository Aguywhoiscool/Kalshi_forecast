
"""Track rolling order-book activity and midpoint changes."""

import time
from collections import defaultdict, deque
from decimal import Decimal


class MarketActivityTracker:
    def __init__(
        self,
        window_seconds=300,
        min_deltas_per_minute=30,
    ):
        self.window_seconds = window_seconds
        self.min_deltas_per_minute = min_deltas_per_minute

        self.events = defaultdict(deque)
        self.midpoint_events = defaultdict(deque)

        self.started_at = {}
        self.last_midpoint = {}
        self.failed_reviews = defaultdict(int)

    def register_market(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        self.started_at[ticker] = now
        self.events[ticker].clear()
        self.midpoint_events[ticker].clear()
        self.last_midpoint.pop(ticker, None)
        self.failed_reviews[ticker] = 0

    def record_delta(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            self.register_market(ticker, now)

        self.events[ticker].append(now)
        self._prune(ticker, now)

    def record_midpoint(self, ticker, midpoint, now=None):
        """Record a change in the best bid/ask midpoint."""

        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            self.register_market(ticker, now)

        if midpoint is None:
            # A missing quote must not be counted as
            # a price change.
            self.last_midpoint.pop(ticker, None)
            return

        midpoint = Decimal(str(midpoint))
        previous = self.last_midpoint.get(ticker)

        if previous is not None and midpoint != previous:
            self.midpoint_events[ticker].append(now)

        self.last_midpoint[ticker] = midpoint
        self._prune(ticker, now)

    def _prune(self, ticker, now):
        cutoff = now - self.window_seconds

        for events in (
            self.events[ticker],
            self.midpoint_events[ticker],
        ):
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

    def get_midpoint_changes(self, ticker, now=None):
        now = time.monotonic() if now is None else now
        self._prune(ticker, now)

        return len(self.midpoint_events[ticker])

    def review_market(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            return False

        # Five-minute observation period.
        if now - self.started_at[ticker] < 300:
            return False

        rate = self.get_rate(ticker, now)
        changes = self.get_midpoint_changes(ticker, now)

        # Rotate only if both activity signals are low.
        should_rotate = (
            rate < self.min_deltas_per_minute
            and changes == 0
        )

        self.failed_reviews[ticker] = (
            1 if should_rotate else 0
        )

        return should_rotate
