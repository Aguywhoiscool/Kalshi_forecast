
from datetime import datetime, timedelta, timezone

from src.activity_market_selection import (
    select_observed_replacements,
)
from src.market_observation_cache import MarketObservationCache


NOW = datetime(2026, 10, 28, 16, 0, tzinfo=timezone.utc)


def market(series, event, outcome):
    return {
        "ticker": f"{series}-{event}-{outcome}",
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


def make_cache(tmp_path, observations, observed_at=NOW):
    cache = MarketObservationCache(
        path=tmp_path / "cache.json",
        ttl_seconds=300,
    )
    cache.update(observations, now=observed_at)
    return cache


def tickers(markets):
    return [m["ticker"] for m in markets]


def test_selects_qualified_market(tmp_path):
    a = market("SERIESA", "EVENT1", "YES")
    b = market("SERIESB", "EVENT2", "YES")

    cache = make_cache(tmp_path, {
        a["ticker"]: observation(a["ticker"], changes=10),
        b["ticker"]: observation(b["ticker"], changes=0),
    })

    result = select_observed_replacements(
        [a, b], cache, target_markets=2, now=NOW
    )

    assert tickers(result["selected"]) == [a["ticker"]]
    assert result["shortage"] == 1
    assert result["target_met"] is False


def test_rejects_low_coverage(tmp_path):
    a = market("SERIESA", "EVENT1", "YES")

    cache = make_cache(tmp_path, {
        a["ticker"]: observation(
            a["ticker"], changes=20, coverage=0.50
        ),
    })

    result = select_observed_replacements(
        [a], cache, target_markets=1, now=NOW
    )

    assert result["selected"] == []
    assert result["shortage"] == 1


def test_expired_observation_cannot_qualify(tmp_path):
    a = market("SERIESA", "EVENT1", "YES")

    cache = make_cache(
        tmp_path,
        {a["ticker"]: observation(a["ticker"])},
        observed_at=NOW - timedelta(seconds=301),
    )

    result = select_observed_replacements(
        [a], cache, target_markets=1, now=NOW
    )

    assert result["selected"] == []
    assert result["fresh_observations"] == 0


def test_preserves_existing_and_event_diversity(tmp_path):
    existing = market("SERIESA", "EVENT1", "YES")
    same_event = market("SERIESA", "EVENT1", "NO")
    different_event = market("SERIESB", "EVENT2", "YES")

    cache = make_cache(tmp_path, {
        same_event["ticker"]: observation(
            same_event["ticker"], changes=100
        ),
        different_event["ticker"]: observation(
            different_event["ticker"], changes=5
        ),
    })

    result = select_observed_replacements(
        [same_event, different_event],
        cache,
        existing_markets=[existing],
        target_markets=2,
        now=NOW,
    )

    assert tickers(result["selected"]) == [
        existing["ticker"],
        different_event["ticker"],
    ]
    assert tickers(result["additions"]) == [
        different_event["ticker"]
    ]
    assert result["target_met"] is True


def test_no_qualified_replacements(tmp_path):
    existing = market("SERIESA", "EVENT1", "YES")
    inactive = market("SERIESB", "EVENT2", "YES")

    cache = make_cache(tmp_path, {
        inactive["ticker"]: observation(
            inactive["ticker"], changes=0
        ),
    })

    result = select_observed_replacements(
        [inactive],
        cache,
        existing_markets=[existing],
        target_markets=3,
        now=NOW,
    )

    assert tickers(result["selected"]) == [
        existing["ticker"]
    ]
    assert result["additions"] == []
    assert result["shortage"] == 2
