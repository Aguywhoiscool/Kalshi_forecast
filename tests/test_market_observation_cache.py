
from datetime import datetime, timedelta, timezone

from src.market_observation_cache import MarketObservationCache
from src.market_probe_batch import MarketProbeBatch


def test_cache_stores_and_recovers_observations(tmp_path):
    path = tmp_path / "cache.json"
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)

    cache = MarketObservationCache(path=path)

    observation = {
        "ticker": "TEST",
        "midpoint_changes": 5,
    }

    cache.update({"TEST": observation}, now=now)

    reloaded = MarketObservationCache(path=path)

    assert reloaded.get(
        "TEST", now=now + timedelta(seconds=60)
    ) == observation


def test_cached_observations_expire(tmp_path):
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)

    cache = MarketObservationCache(
        path=tmp_path / "cache.json",
        ttl_seconds=300,
    )

    cache.update(
        {"TEST": {"midpoint_changes": 0}},
        now=now,
    )

    assert cache.get(
        "TEST", now=now + timedelta(seconds=299)
    ) is not None

    assert cache.get(
        "TEST", now=now + timedelta(seconds=300)
    ) is None


def test_probe_rotates_through_candidates(tmp_path):
    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    probe = MarketProbeBatch(
        path=tmp_path / "cursor.json"
    )

    candidates = [
        {"ticker": f"MARKET-{i}"}
        for i in range(6)
    ]

    first = probe.next_batch(
        candidates, cache, batch_size=2
    )
    probe.commit_batch(candidates, first)
    second = probe.next_batch(
        candidates, cache, batch_size=2
    )

    assert [m["ticker"] for m in first] == [
        "MARKET-0", "MARKET-1"
    ]

    assert [m["ticker"] for m in second] == [
        "MARKET-2", "MARKET-3"
    ]


def test_probe_skips_cached_markets(tmp_path):
    now = datetime.now(timezone.utc)

    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    cache.update(
        {"MARKET-0": {"midpoint_changes": 0}},
        now=now,
    )

    probe = MarketProbeBatch(
        path=tmp_path / "cursor.json"
    )

    candidates = [
        {"ticker": f"MARKET-{i}"}
        for i in range(4)
    ]

    selected = probe.next_batch(
        candidates, cache, batch_size=2
    )

    assert [m["ticker"] for m in selected] == [
        "MARKET-1", "MARKET-2"
    ]


def test_probe_cursor_persists(tmp_path):
    path = tmp_path / "cursor.json"

    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    candidates = [
        {"ticker": f"MARKET-{i}"}
        for i in range(4)
    ]

    probe = MarketProbeBatch(path=path)
    first = probe.next_batch(
        candidates, cache, batch_size=2
    )
    probe.commit_batch(candidates, first)

    reloaded = MarketProbeBatch(path=path)

    selected = reloaded.next_batch(
        candidates, cache, batch_size=2
    )

    assert [m["ticker"] for m in selected] == [
        "MARKET-2", "MARKET-3"
    ]

def test_cache_ages_from_observation_start(tmp_path):
    started = datetime(
        2026, 10, 28, 16, 0,
        tzinfo=timezone.utc,
    )

    cache = MarketObservationCache(
        path=tmp_path / "cache.json",
        ttl_seconds=300,
    )

    cache.update(
        {
            "TEST": {
                "ticker": "TEST",
                "midpoint_changes": 5,
            }
        },
        observed_at=started,
    )

    assert cache.get(
        "TEST",
        now=started + timedelta(seconds=299),
    ) is not None

    assert cache.get(
        "TEST",
        now=started + timedelta(seconds=300),
    ) is None

def test_failed_observation_does_not_advance_cursor(tmp_path):
    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    path = tmp_path / "cursor.json"

    probe = MarketProbeBatch(path=path)

    candidates = [
        {"ticker": f"MARKET-{i}"}
        for i in range(5)
    ]

    first = probe.next_batch(
        candidates,
        cache,
        batch_size=2,
    )

    # Simulate observation failure by not committing.
    restarted = MarketProbeBatch(path=path)

    retry = restarted.next_batch(
        candidates,
        cache,
        batch_size=2,
    )

    assert [m["ticker"] for m in retry] == [
        m["ticker"] for m in first
    ]

    assert restarted.cursor == 0
