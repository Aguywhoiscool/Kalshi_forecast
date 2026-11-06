import asyncio
import threading
from types import SimpleNamespace

import pytest

from src.background_status_review import BackgroundStatusReview


@pytest.mark.asyncio
async def test_slow_review_does_not_block_or_duplicate_requests():
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow_status(ticker):
        calls.append(ticker)
        started.set()
        assert release.wait(3)
        return {"tradable": True}

    review = BackgroundStatusReview(["TEST"], slow_status)
    review.next_review = 0
    try:
        async with review:
            assert review.poll(0) is None
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(.001)
            assert started.is_set()
            for tick in range(1, 100):
                assert review.poll(tick) is None
                await asyncio.sleep(0)
            assert calls == ["TEST"]
            release.set()
            await asyncio.wait_for(asyncio.shield(review.task), 2)
            assert review.poll(100) == [{"tradable": True}]
            assert review.poll(159) is None
            assert review.task is None
    finally:
        release.set()


@pytest.mark.asyncio
async def test_failures_remain_visible_and_successful_results_survive():
    def fetch(ticker):
        if ticker == "FAILED":
            raise RuntimeError("offline request failed")
        return {"tradable": False}

    async with BackgroundStatusReview(["FAILED", "CLOSED"], fetch) as review:
        review.next_review = 0
        review.poll(0)
        await asyncio.wait_for(asyncio.shield(review.task), 2)
        results = review.poll(1)
        assert isinstance(results[0], RuntimeError)
        assert results[1] == {"tradable": False}


@pytest.mark.asyncio
async def test_session_cleanup_cancels_pending_review():
    cancelled = asyncio.Event()
    review = BackgroundStatusReview([], lambda ticker: None)

    async def pending():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async with review:
        review.task = asyncio.create_task(pending())
        task = review.task
        await asyncio.sleep(0)
    assert cancelled.is_set()
    assert task.cancelled()
    assert review.task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", [True, False])
async def test_stream_samples_during_review_and_handles_closed_market(monkeypatch, slow):
    import src.websocket_client as client

    clock = [0.0]
    samples = []
    started, release = threading.Event(), threading.Event()

    class EndStream(Exception):
        pass

    def fetch(ticker):
        started.set()
        if slow:
            assert release.wait(3)
        return {"tradable": False, "status": "closed", "close_time": None}

    reviews = []

    def make_review(tickers, fetch_status):
        review = BackgroundStatusReview(tickers, fetch_status)
        review.next_review = 0
        reviews.append(review)
        return review

    class FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def send(self, message):
            pass

        async def recv(self):
            await asyncio.sleep(.002)
            clock[0] += 1
            if slow and started.is_set() and len(samples) >= 4:
                assert not release.is_set()
                raise EndStream()
            if clock[0] > 50:
                raise AssertionError("Stream failed to process review results")
            return '{"type":"heartbeat"}'

    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(client, "get_auth_headers", lambda *args: {})
    monkeypatch.setattr(client, "get_live_market_status", fetch)
    monkeypatch.setattr(client, "BackgroundStatusReview", make_review)
    monkeypatch.setattr(client.websockets, "connect", lambda *args, **kw: FakeSocket())
    monkeypatch.setattr(client, "save_raw_message", lambda *args, **kw: None)
    monkeypatch.setattr(client, "FeatureSampler", lambda: SimpleNamespace(
        record=lambda *args: samples.append(clock[0]),
        update_market_metadata=lambda *args: None))
    try:
        expected = EndStream if slow else client.MarketRotationRequired
        with pytest.raises(expected) as caught:
            await asyncio.wait_for(client.stream_markets(["TEST"]), 2)
        if slow:
            assert len(samples) >= 4
        else:
            assert caught.value.rejected_tickers == {"TEST"}
            assert caught.value.reason == "not_tradable"
        assert reviews[0].task is None
    finally:
        release.set()
