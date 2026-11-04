"""Run public market-status requests without pausing streaming."""

import asyncio
import time


class BackgroundStatusReview:
    def __init__(self, tickers, fetch_status, interval_seconds=60):
        self.tickers = tuple(tickers)
        self.fetch_status = fetch_status
        self.interval_seconds = interval_seconds
        self.next_review = time.monotonic() + interval_seconds
        self.task = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        # Cancelling to_thread cannot stop an in-flight HTTP request; its
        # existing request timeout still applies. Results are never reused.

    async def _fetch(self):
        started = time.monotonic()
        results = await asyncio.gather(
            *(asyncio.to_thread(self.fetch_status, ticker)
              for ticker in self.tickers),
            return_exceptions=True,
        )
        print(f"Market status review completed in {time.monotonic() - started:.2f}s")
        return results

    def poll(self, now):
        """Return completed results, or schedule one batch without waiting."""
        if self.task is not None:
            if not self.task.done():
                return None
            task, self.task = self.task, None
            self.next_review = now + self.interval_seconds
            return task.result()
        if now >= self.next_review:
            print("\n=== LIVE MARKET STATUS REVIEW (background) ===")
            self.task = asyncio.create_task(self._fetch())
        return None
