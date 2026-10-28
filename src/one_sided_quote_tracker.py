
"""Track markets with persistently one-sided quotes."""

import time


class OneSidedQuoteTracker:
    def __init__(self, persistence_seconds=60.0):
        if persistence_seconds <= 0:
            raise ValueError(
                "persistence_seconds must be positive"
            )

        self.persistence_seconds = persistence_seconds
        self.one_sided_since = {}

    def update(self, ticker, has_snapshot, prices, now=None):
        """Observe a reconstructed book once per sampling interval.

        Do not start the timer before the first snapshot.
        Reset the timer as soon as a two-sided quote returns.
        """
        if now is None:
            now = time.monotonic()

        if not has_snapshot or prices is None:
            self.one_sided_since.pop(ticker, None)
            return self.get_status(ticker, now)

        bid = prices.get("yes_bid")
        ask = prices.get("yes_ask")

        is_one_sided = bid is None or ask is None

        if is_one_sided:
            self.one_sided_since.setdefault(ticker, now)
        else:
            self.one_sided_since.pop(ticker, None)

        return self.get_status(ticker, now)

    def get_status(self, ticker, now=None):
        if now is None:
            now = time.monotonic()

        started = self.one_sided_since.get(ticker)

        seconds = (
            max(0.0, now - started)
            if started is not None
            else 0.0
        )

        return {
            "one_sided": started is not None,
            "one_sided_seconds": seconds,
            "should_rotate": (
                started is not None
                and seconds >= self.persistence_seconds
            ),
        }
