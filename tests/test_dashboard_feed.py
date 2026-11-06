import asyncio
import json

import numpy as np
import pytest

from src.dashboard_feed import KalshiFeed


def snapshot(seq=1):
    return {"type": "orderbook_snapshot", "sid": 1, "seq": seq, "msg": {
        "market_ticker": "T", "yes_dollars_fp": [[".40", "2"]], "no_dollars_fp": [[".58", "3"]]}}


def ready():
    feed = KalshiFeed(["T"])
    feed.state["status"] = "connected"
    feed.event(snapshot())
    return feed


def test_unchanged_book_is_valid_between_deltas():
    feed = ready()
    for second in range(60):
        feed.sample(["T"], (second+100)*10**9)
    items, errors, health = feed.latest({"start": 0, "end": 1, "sessions": []})
    assert len(items) == 1 and not errors and health["warmup_samples"] == 60
    np.testing.assert_allclose(items[0]["history"][-1], [.4, .42, .02, .41, 2, 3, -.2])


def test_reused_subscription_id_discards_history_and_sequence():
    feed = ready()
    feed.sample(["T"], 100*10**9)
    feed.event({"type": "subscribed"})
    assert not feed.histories and not feed.manager.books
    feed.event(snapshot(1))
    assert not feed.manager.invalid_subscriptions


def test_gap_forces_resync_and_sampling_pause_cannot_stitch():
    feed = ready()
    with pytest.raises(ValueError, match="sequence gap"):
        feed.event(snapshot(3))
    feed = ready()
    feed.sample(["T"], 100*10**9)
    feed.sample(["T"], 103*10**9)
    assert len(feed.histories["T"]) == 1


def test_only_market_data_subscription_is_sent_and_cleanup_resets(monkeypatch):
    sent = []
    class Socket:
        async def send(self, payload):
            sent.append(json.loads(payload))
        async def recv(self):
            raise asyncio.CancelledError
    class Connection:
        async def __aenter__(self):
            return Socket()
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr("src.dashboard_feed.websockets.connect", lambda *a, **kw: Connection())
    monkeypatch.setattr("src.dashboard_feed.get_auth_headers", lambda *a: {})
    feed = KalshiFeed(["T"])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(feed.run())
    assert sent == [{"id": 1, "cmd": "subscribe", "params": {
        "channels": ["orderbook_delta"], "market_tickers": ["T"]}}]
    assert not feed.histories and not feed.manager.books
