
"""Track valid order-book coverage during market observation."""

import time


class BookCoverageTracker:
    def __init__(self, tickers, start_time=None):
        self.start_time = (
            time.monotonic()
            if start_time is None
            else start_time
        )

        self.states = {
            ticker: {
                "valid": False,
                "last_update": self.start_time,
                "valid_seconds": 0.0,
                "first_valid": None,
            }
            for ticker in tickers
        }

    def update(self, ticker, valid, now=None):
        now = time.monotonic() if now is None else now

        state = self.states[ticker]

        elapsed = max(
            0.0, now - state["last_update"]
        )

        if state["valid"]:
            state["valid_seconds"] += elapsed

        state["valid"] = bool(valid)
        state["last_update"] = now

        if valid and state["first_valid"] is None:
            state["first_valid"] = now

    def summarize(self, ticker, end_time=None):
        end_time = (
            time.monotonic()
            if end_time is None
            else end_time
        )

        state = self.states[ticker]

        valid_seconds = state["valid_seconds"]

        if state["valid"]:
            valid_seconds += max(
                0.0, end_time - state["last_update"]
            )

        total_seconds = max(
            0.0, end_time - self.start_time
        )

        coverage = (
            valid_seconds / total_seconds
            if total_seconds > 0
            else 0.0
        )

        return {
            "valid_seconds": round(valid_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "valid_coverage": round(
                min(1.0, coverage), 4
            ),
            "first_valid_delay": (
                round(
                    state["first_valid"] - self.start_time,
                    3,
                )
                if state["first_valid"] is not None
                else None
            ),
        }
