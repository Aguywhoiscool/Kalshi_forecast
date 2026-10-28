
"""Track persistently wide market spreads using monotonic time."""

import math
import time


class SpreadQualityTracker:
    def __init__(
        self,
        max_spread_cents=10.0,
        persistence_seconds=60.0,
        max_update_age_seconds=15.0,
    ):
        self.max_spread_cents = max_spread_cents
        self.persistence_seconds = persistence_seconds
        self.max_update_age_seconds = max_update_age_seconds
        self.states = {}

    def register_market(self, ticker):
        self.states[ticker] = {
            "wide_since": None,
            "last_update": None,
            "last_spread": None,
        }

    def record_spread(self, ticker, spread_cents, now=None):
        """Record a spread on receipt of a fresh order-book update."""
        now = time.monotonic() if now is None else now

        if ticker not in self.states:
            self.register_market(ticker)

        state = self.states[ticker]

        try:
            spread = float(spread_cents)
        except (ValueError, TypeError):
            spread = None

        if (
            spread is None
            or not math.isfinite(spread)
            or spread <= 0
            or spread > 100
        ):
            state["wide_since"] = None
            state["last_spread"] = None
            state["last_update"] = now
            return

        # A prolonged absence of updates breaks continuity.
        last_update = state["last_update"]

        if (
            last_update is not None
            and now - last_update > self.max_update_age_seconds
        ):
            state["wide_since"] = None

        state["last_spread"] = spread
        state["last_update"] = now

        if spread > self.max_spread_cents:
            if state["wide_since"] is None:
                state["wide_since"] = now
        else:
            state["wide_since"] = None

    def get_status(self, ticker, now=None):
        now = time.monotonic() if now is None else now

        state = self.states.get(ticker)

        if state is None:
            return {
                "spread_cents": None,
                "wide_seconds": 0.0,
                "stale": True,
                "should_rotate": False,
            }

        last_update = state["last_update"]

        stale = (
            last_update is None
            or now - last_update > self.max_update_age_seconds
        )

        wide_since = state["wide_since"]

        wide_seconds = (
            max(0.0, now - wide_since)
            if wide_since is not None and not stale
            else 0.0
        )

        return {
            "spread_cents": state["last_spread"],
            "wide_seconds": wide_seconds,
            "stale": stale,
            "should_rotate": (
                not stale
                and wide_since is not None
                and wide_seconds >= self.persistence_seconds
            ),
        }
