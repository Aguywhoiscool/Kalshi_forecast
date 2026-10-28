
"""Track rolling order-book activity and midpoint changes."""

import time
from collections import defaultdict, deque
from decimal import Decimal
MEANINGFUL_MOVE_THRESHOLD = Decimal("0.005")

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
        self.meaningful_midpoint_events = defaultdict(deque)

        self.started_at = {}
        self.last_midpoint = {}
        self.failed_reviews = defaultdict(int)

    def register_market(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        self.started_at[ticker] = now
        self.events[ticker].clear()
        self.midpoint_events[ticker].clear()
        self.meaningful_midpoint_events[ticker].clear()
        self.last_midpoint.pop(ticker, None)
        self.failed_reviews[ticker] = 0

    def record_delta(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            self.register_market(ticker, now)

        self.events[ticker].append(now)
        self._prune(ticker, now)

    
    def record_midpoint(self, ticker, midpoint, now=None):
        """Track all midpoint changes and meaningful midpoint changes."""

        now = time.monotonic() if now is None else now

        if ticker not in self.started_at:
            self.register_market(ticker, now)

        if midpoint is None:
            self.last_midpoint.pop(ticker, None)
            return

        midpoint = Decimal(str(midpoint))
        previous = self.last_midpoint.get(ticker)

        if previous is not None:
            movement = abs(midpoint - previous)

            # Count every nonzero midpoint change.
            if movement > 0:
                self.midpoint_events[ticker].append(now)

            # Separately count changes of at least half a cent.
            if movement >= MEANINGFUL_MOVE_THRESHOLD:
                self.meaningful_midpoint_events[ticker].append(now)

        self.last_midpoint[ticker] = midpoint
        self._prune(ticker, now)


    def _prune(self, ticker, now):
        cutoff = now - self.window_seconds

        for events in (
            self.events[ticker],
            self.midpoint_events[ticker],
            self.meaningful_midpoint_events[ticker],
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
    def get_meaningful_midpoint_changes(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        self._prune(ticker, now)

        return len(self.meaningful_midpoint_events[ticker])

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
        # Once the market has been monitored for five minutes,
        # rotate it if its midpoint has never changed during
        # the current rolling activity window.
        #
        # A high delta rate alone does not justify keeping a
        # market whose executable quotes never move.
        should_rotate = changes == 0

        self.failed_reviews[ticker] = (
            1 if should_rotate else 0
        )

        return should_rotate
    
