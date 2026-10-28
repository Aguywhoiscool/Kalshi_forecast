
from src.live_market_status import get_live_market_status


class FakeResponse:
    def __init__(self, market):
        self.market = market

    def raise_for_status(self):
        pass

    def json(self):
        return {"market": self.market}


def test_current_spread_from_dollar_quotes(monkeypatch):
    market = {
        "status": "active",
        "yes_bid_dollars": "0.05",
        "yes_ask_dollars": "0.49",
    }

    monkeypatch.setattr(
        "src.live_market_status.requests.get",
        lambda *args, **kwargs: FakeResponse(market),
    )

    monkeypatch.setattr(
        "src.live_market_status.market_is_tradable",
        lambda *args, **kwargs: True,
    )

    result = get_live_market_status("TEST-MARKET")

    assert result["yes_bid_cents"] == 5.0
    assert result["yes_ask_cents"] == 49.0
    assert result["spread_cents"] == 44.0
    assert result["tradable"] is True


def test_current_spread_from_integer_cents(monkeypatch):
    market = {
        "status": "active",
        "yes_bid": 43,
        "yes_ask": 45,
    }

    monkeypatch.setattr(
        "src.live_market_status.requests.get",
        lambda *args, **kwargs: FakeResponse(market),
    )

    monkeypatch.setattr(
        "src.live_market_status.market_is_tradable",
        lambda *args, **kwargs: True,
    )

    result = get_live_market_status("TEST-MARKET")

    assert result["spread_cents"] == 2.0


def test_missing_quote_is_not_treated_as_narrow(monkeypatch):
    market = {
        "status": "active",
        "yes_bid": 25,
    }

    monkeypatch.setattr(
        "src.live_market_status.requests.get",
        lambda *args, **kwargs: FakeResponse(market),
    )

    monkeypatch.setattr(
        "src.live_market_status.market_is_tradable",
        lambda *args, **kwargs: True,
    )

    result = get_live_market_status("TEST-MARKET")

    assert result["spread_cents"] is None
