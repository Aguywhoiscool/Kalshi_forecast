
from datetime import datetime, timedelta, timezone

from src.activity_candidate_pool import prioritize_rotation_candidates
from src.market_observation_cache import MarketObservationCache


NOW = datetime(2026, 10, 28, 16, 0, tzinfo=timezone.utc)


def market(series, event):
    return {
        "ticker": f"{series}-{event}-YES",
        "event_ticker": f"{series}-{event}",
    }


def observation(ticker, changes=5, coverage=0.99):
    return {
        "ticker": ticker,
        "duration_seconds": 120,
        "delta_count": 100,
        "midpoint_changes": changes,
        "meaningful_changes": changes,
        "spread_cents": 2.0,
        "valid_book": True,
        "valid_coverage": coverage,
    }


def make_cache(tmp_path, observations, at=NOW):
    cache = MarketObservationCache(
        path=tmp_path / "cache.json",
        ttl_seconds=300,
    )
    cache.update(observations, observed_at=at)
    return cache


def test_observed_active_market_gets_priority(tmp_path):
    quiet = market("A", "EVENT1")
    active = market("B", "EVENT2")

    cache = make_cache(tmp_path, {
        quiet["ticker"]: observation(
            quiet["ticker"], changes=0
        ),
        active["ticker"]: observation(
            active["ticker"], changes=12
        ),
    })

    result = prioritize_rotation_candidates(
        candidates=[quiet, active],
        cache=cache,
        existing_markets=[],
        blocked_tickers=set(),
        target_markets=2,
        now=NOW,
    )

    assert result["candidates"][0] == active
    assert result["qualified_count"] == 1
    assert result["candidates"][1] == quiet


def test_blocked_markets_never_return(tmp_path):
    blocked = market("A", "EVENT1")
    available = market("B", "EVENT2")

    cache = make_cache(tmp_path, {})

    result = prioritize_rotation_candidates(
        [blocked, available],
        cache,
        existing_markets=[],
        blocked_tickers={blocked["ticker"]},
        target_markets=1,
        now=NOW,
    )

    assert result["candidates"] == [available]


def test_expired_activity_is_not_qualified(tmp_path):
    stale = market("A", "EVENT1")
    quiet = market("B", "EVENT2")

    cache = make_cache(
        tmp_path,
        {stale["ticker"]: observation(stale["ticker"])},
        at=NOW - timedelta(seconds=301),
    )

    result = prioritize_rotation_candidates(
        [quiet, stale],
        cache,
        existing_markets=[],
        blocked_tickers=set(),
        target_markets=2,
        now=NOW,
    )

    assert result["qualified_count"] == 0
    assert result["candidates"] == [quiet, stale]


def test_no_cache_still_provides_fallback(tmp_path):
    a = market("A", "EVENT1")
    b = market("B", "EVENT2")

    cache = make_cache(tmp_path, {})

    result = prioritize_rotation_candidates(
        [a, b],
        cache,
        existing_markets=[],
        blocked_tickers=set(),
        target_markets=2,
        now=NOW,
    )

    assert result["qualified_count"] == 0
    assert result["candidates"] == [a, b]


def test_existing_event_not_activity_qualified(tmp_path):
    keep = market("A", "EVENT1")
    same_event = {
        "ticker": "A-EVENT1-NO",
        "event_ticker": "A-EVENT1",
    }
    other = market("B", "EVENT2")

    cache = make_cache(tmp_path, {
        same_event["ticker"]: observation(
            same_event["ticker"], changes=100
        ),
        other["ticker"]: observation(
            other["ticker"], changes=5
        ),
    })

    result = prioritize_rotation_candidates(
        [same_event, other],
        cache,
        existing_markets=[keep],
        blocked_tickers={keep["ticker"]},
        target_markets=2,
        now=NOW,
    )

    assert result["qualified_tickers"] == {other["ticker"]}

from src.activity_candidate_pool import spread_quality


def test_spread_quality_thresholds():
    assert spread_quality({
        "spread_cents": 2.0,
        "last_spread_cents": 3.0,
    }) == "preferred"

    assert spread_quality({
        "spread_cents": 4.0,
        "last_spread_cents": 8.0,
    }) == "secondary"

    assert spread_quality({
        "spread_cents": 4.0,
        "last_spread_cents": 44.0,
    }) == "reject"

    assert spread_quality({
        "spread_cents": 19.0,
        "last_spread_cents": 2.0,
    }) == "reject"

    assert spread_quality({}) == "unknown"


def test_wide_spread_market_is_removed(tmp_path):
    wide = market("A", "EVENT1")
    narrow = market("B", "EVENT2")

    wide_obs = observation(wide["ticker"], changes=100)
    wide_obs["spread_cents"] = 44.0
    wide_obs["last_spread_cents"] = 44.0

    narrow_obs = observation(narrow["ticker"], changes=10)
    narrow_obs["spread_cents"] = 2.0
    narrow_obs["last_spread_cents"] = 2.0

    cache = make_cache(tmp_path, {
        wide["ticker"]: wide_obs,
        narrow["ticker"]: narrow_obs,
    })

    result = prioritize_rotation_candidates(
        candidates=[wide, narrow],
        cache=cache,
        existing_markets=[],
        blocked_tickers=set(),
        target_markets=2,
        now=NOW,
    )

    assert wide not in result["candidates"]
    assert narrow in result["candidates"]
    assert result["spread_rejected_count"] == 1
    assert result["qualified_count"] == 1


def test_narrow_spread_precedes_wider_spread(tmp_path):
    wider = market("A", "EVENT1")
    narrow = market("B", "EVENT2")

    wider_obs = observation(wider["ticker"], changes=50)
    wider_obs["spread_cents"] = 8.0
    wider_obs["last_spread_cents"] = 8.0

    narrow_obs = observation(narrow["ticker"], changes=5)
    narrow_obs["spread_cents"] = 2.0
    narrow_obs["last_spread_cents"] = 2.0

    cache = make_cache(tmp_path, {
        wider["ticker"]: wider_obs,
        narrow["ticker"]: narrow_obs,
    })

    result = prioritize_rotation_candidates(
        candidates=[wider, narrow],
        cache=cache,
        existing_markets=[],
        blocked_tickers=set(),
        target_markets=2,
        now=NOW,
    )

    assert result["candidates"][0] == narrow
    assert result["candidates"][1] == wider
    assert result["preferred_count"] == 1
    assert result["secondary_count"] == 1
