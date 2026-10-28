
from datetime import datetime, timedelta, timezone

from src.market_lifecycle import market_is_tradable


def future_time(minutes):
    return (
        datetime.now(timezone.utc)
        + timedelta(minutes=minutes)
    ).isoformat()


def test_active_market():
    market = {
        "status": "active",
        "close_time": future_time(60),
    }

    assert market_is_tradable(market)


def test_closed_market():
    market = {
        "status": "closed",
        "close_time": future_time(60),
    }

    assert not market_is_tradable(market)


def test_market_closing_soon():
    market = {
        "status": "active",
        "close_time": future_time(5),
    }

    assert not market_is_tradable(market)


def test_missing_close_time():
    market = {"status": "active"}

    assert not market_is_tradable(market)
