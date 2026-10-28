
"""Coordinate independent candidate observation without blocking collection."""

import asyncio
from datetime import datetime, timezone
import time
from src.market_ranker import rank_active_markets
from src.market_observer import observe_markets
from src.market_observation_cache import MarketObservationCache
from src.market_probe_batch import MarketProbeBatch


class ObservationCoordinator:
    def __init__(
        self,
        candidate_count=80,
        batch_size=20,
        observation_seconds=120,
    ):
        self.candidate_count = candidate_count
        self.batch_size = batch_size
        self.observation_seconds = observation_seconds

        self.cache = MarketObservationCache()
        self.probe = MarketProbeBatch()

        self.candidates = []
        self.last_discovery_at = None
        self.task = None
        self.last_error = None

    async def refresh(self, excluded_tickers=None):
        excluded = set(excluded_tickers or [])

        discovery_expired = (
            self.last_discovery_at is None
            or time.monotonic() - self.last_discovery_at >= 360
        )

        if discovery_expired or not self.candidates:
            candidates = await asyncio.to_thread(
                rank_active_markets,
                max_markets=self.candidate_count,
                excluded_tickers=excluded,
            )

            
            self.candidates = list(candidates)

            print(
                f"Observer discovery refreshed: "
                f"{len(candidates)} candidates"
            )
        else:
            candidates = [
                market
                for market in self.candidates
                if market["ticker"] not in excluded
            ]
        if not candidates:
            self.candidates = []
            return

        batch = self.probe.next_batch(
            candidates,
            self.cache,
            batch_size=self.batch_size,
        )

        if batch:
            started = datetime.now(timezone.utc)

            observations = await observe_markets(
                [market["ticker"] for market in batch],
                duration_seconds=self.observation_seconds,
            )

            self.cache.update(
                observations,
                observed_at=started,
            )

            self.probe.commit_batch(candidates, batch)

        self.candidates = candidates
        self.last_discovery_at = time.monotonic()

    def start(self, excluded_tickers=None):
        """Start a new observation task if none is running."""

        if self.task is not None and not self.task.done():
            return False

        self.last_error = None

        self.task = asyncio.create_task(
            self.refresh(
                excluded_tickers=excluded_tickers,
            )
        )

        return True

    async def run_continuous(self, get_excluded_tickers):
        """Continuously observe candidates without blocking the collector."""

        while True:
            try:
                excluded = set(get_excluded_tickers())

                print(
                    "\nOBSERVER: Starting candidate observation cycle"
                )

                await self.refresh(
                    excluded_tickers=excluded
                )

                print(
                    "OBSERVER: Observation cycle completed"
                )

                self.last_error = None

                # Avoid launching batches back-to-back.
                await asyncio.sleep(15)

            except asyncio.CancelledError:
                print("OBSERVER: Shutting down")
                raise

            except Exception as error:
                self.last_error = str(error)

                print(
                    f"OBSERVER ERROR: {error}"
                )

                # Back off before retrying a failed cycle.
                await asyncio.sleep(60)

    def start_continuous(self, get_excluded_tickers):
        """Start one long-lived observer task."""

        if self.task is not None and not self.task.done():
            return False

        self.last_error = None

        self.task = asyncio.create_task(
            self.run_continuous(get_excluded_tickers)
        )

        return True

    async def stop(self):
        """Cancel and await the background observation task."""

        if self.task is None:
            return

        if not self.task.done():
            self.task.cancel()

        try:
            await self.task
        except asyncio.CancelledError:
            pass

        self.task = None

    def check(self):
        """Inspect a completed task and capture its exception."""

        if self.task is None or not self.task.done():
            return

        try:
            self.task.result()
        except asyncio.CancelledError:
            self.last_error = "Observation cancelled"
        except Exception as exc:
            self.last_error = str(exc)

    def get_candidates(self, max_age_seconds=420):
        """Return candidates only while the REST discovery is fresh."""

        self.check()

        if self.last_discovery_at is None:
            return []

        if (
            time.monotonic() - self.last_discovery_at
            > max_age_seconds
        ):
            return []

        return list(self.candidates)