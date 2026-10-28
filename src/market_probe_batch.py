
"""Rotate through candidate markets instead of probing the same ones."""

import json
from pathlib import Path


CURSOR_FILE = Path(
    "reports/market_observer/probe_cursor.json"
)


class MarketProbeBatch:
    def __init__(self, path=CURSOR_FILE):
        self.path = Path(path)
        self.cursor = 0
        self.load()

    def load(self):
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.cursor = max(
            0, int(data.get("cursor", 0))
        )

    def save(self):
        self.path.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporary = self.path.with_suffix(".tmp")

        with temporary.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {"cursor": self.cursor},
                file,
                indent=2,
            )

        temporary.replace(self.path)

    
    def next_batch(
        self,
        candidates,
        cache,
        batch_size=20,
    ):
        """Find uncached candidates without advancing the cursor."""
        if not candidates or batch_size <= 0:
            return []

        count = len(candidates)
        start = self.cursor % count

        selected = []
        seen = set()

        for offset in range(count):
            index = (start + offset) % count
            market = candidates[index]
            ticker = market["ticker"]

            if ticker in seen:
                continue

            seen.add(ticker)

            if cache.get(ticker) is not None:
                continue

            selected.append(market)

            if len(selected) >= batch_size:
                break

        return selected

    def commit_batch(self, candidates, batch):
        """Advance the cursor after a successful observation."""
        if not candidates or not batch:
            return

        last_ticker = batch[-1]["ticker"]

        for index, market in enumerate(candidates):
            if market["ticker"] == last_ticker:
                self.cursor = (index + 1) % len(candidates)
                self.save()
                return

        raise ValueError(
            "Last observed market is no longer in candidate pool"
        )

