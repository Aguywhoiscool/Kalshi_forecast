
from datetime import datetime, timezone

from src.activity_rotation_planner import plan_activity_rotation
from src.market_observation_cache import MarketObservationCache


NOW = datetime(2026, 10, 28, 16, 0, tzinfo=timezone.utc)


def market(series, event):
    return {
        "ticker": f"{series}-{event}-YES",
        "event_ticker": f"{series}-{event}",
    }


def observation(ticker, changes=5):
    return {
        "ticker": ticker,
        "duration_seconds": 120,
        "delta_count": 100,
        "midpoint_changes": changes,
        "meaningful_changes": changes,
        "spread_cents": 2.0,
        "valid_book": True,
        "valid_coverage": 0.99,
    }


def test_prefers_activity_qualified_replacement(tmp_path):
    old = market("SERIESA", "EVENT1")
    new = market("SERIESB", "EVENT2")

    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    cache.update(
        {new["ticker"]: observation(new["ticker"])},
        now=NOW,
    )

    plan = plan_activity_rotation(
        previous_selected=[old],
        rejected_tickers={old["ticker"]},
        candidates=[new],
        cache=cache,
        target_markets=1,
        now=NOW,
    )

    assert [
        m["ticker"] for m in plan["selected"]
    ] == [new["ticker"]]

    assert plan["used_fallback"] is False


def test_reports_activity_shortage(tmp_path):
    old = market("SERIESA", "EVENT1")
    inactive = market("SERIESB", "EVENT2")

    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    cache.update(
        {
            inactive["ticker"]: observation(
                inactive["ticker"], changes=0
            )
        },
        now=NOW,
    )

    plan = plan_activity_rotation(
        previous_selected=[old],
        rejected_tickers={old["ticker"]},
        candidates=[inactive],
        cache=cache,
        target_markets=1,
        now=NOW,
    )

    assert plan["activity_shortage"] == 1
    assert plan["activity_qualified_additions"] == []


def test_keeps_productive_market(tmp_path):
    keep = market("SERIESA", "EVENT1")
    rejected = market("SERIESB", "EVENT2")
    replacement = market("SERIESC", "EVENT3")

    cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )

    cache.update(
        {
            replacement["ticker"]: observation(
                replacement["ticker"]
            )
        },
        now=NOW,
    )

    plan = plan_activity_rotation(
        previous_selected=[keep, rejected],
        rejected_tickers={rejected["ticker"]},
        candidates=[replacement],
        cache=cache,
        target_markets=2,
        now=NOW,
    )

    tickers = {
        m["ticker"] for m in plan["selected"]
    }

    assert keep["ticker"] in tickers
    assert replacement["ticker"] in tickers
    assert rejected["ticker"] not in tickers
