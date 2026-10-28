
"""Persistent cache for short-lived Kalshi market observations."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


CACHE_FILE = Path(
    "reports/market_observer/activity_cache.json"
)

CACHE_TTL_SECONDS = 300


def utc_now():
    return datetime.now(timezone.utc)


class MarketObservationCache:
    def __init__(
        self,
        path=CACHE_FILE,
        ttl_seconds=CACHE_TTL_SECONDS,
    ):
        self.path = Path(path)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.entries = {}
        self.load()

    def load(self):
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.entries = data.get("entries", {})

    def get(self, ticker, now=None):
        now = utc_now() if now is None else now

        entry = self.entries.get(ticker)
        if entry is None:
            return None

        observed_at = datetime.fromisoformat(
            entry["observed_at"]
        )

        age = now - observed_at

        if age < timedelta(0) or age >= self.ttl:
            return None

        return entry["observation"]

    def update(self, observations, now=None, observed_at=None):
        """Store observations using their actual observation timestamp."""

        timestamp = (
            observed_at
            if observed_at is not None
            else (utc_now() if now is None else now)
        )

        for ticker, observation in observations.items():
            self.entries[ticker] = {
                "observed_at": timestamp.isoformat(),
                "observation": observation,
            }

        # Use the supplied timestamp, not the system clock.
        # This keeps historical tests deterministic.
        self.prune(now=timestamp)
        self.save()


    def prune(self, now=None):
        now = utc_now() if now is None else now

        for ticker in list(self.entries):
            if self.get(ticker, now) is None:
                del self.entries[ticker]

    def fresh_observations(self, tickers, now=None):
        result = {}

        for ticker in tickers:
            observation = self.get(ticker, now)
            if observation is not None:
                result[ticker] = observation

        return result

    def save(self):
        self.path.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporary = self.path.with_suffix(".tmp")

        with temporary.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {"entries": self.entries},
                file,
                indent=2,
            )

        temporary.replace(self.path)
