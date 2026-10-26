
from decimal import Decimal

from src.orderbook import OrderBookManager


def test_snapshot_prices():
    manager = OrderBookManager()

    snapshot = {
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {
            "market_ticker": "TEST-MARKET",
            "yes_dollars": [
                ["0.45", "100.00"]
            ],
            "no_dollars": [
                ["0.54", "200.00"]
            ]
        }
    }

    prices = manager.process(snapshot)

    assert prices["yes_bid"] == Decimal("0.45")
    assert prices["yes_ask"] == Decimal("0.46")
    assert prices["spread"] == Decimal("0.01")
    assert prices["midpoint"] == Decimal("0.455")


def test_delta_updates_quantity():
    manager = OrderBookManager()

    manager.process({
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {
            "market_ticker": "TEST-MARKET",
            "yes_dollars": [["0.45", "100.00"]],
            "no_dollars": [["0.54", "200.00"]]
        }
    })

    manager.process({
        "type": "orderbook_delta",
        "sid": 1,
        "seq": 2,
        "msg": {
            "market_ticker": "TEST-MARKET",
            "side": "yes",
            "price_dollars": "0.45",
            "delta_fp": "50.00"
        }
    })

    prices = manager.get_prices("TEST-MARKET")

    assert prices["yes_bid_depth"] == Decimal("150.00")

def test_order_cancellation():
    manager = OrderBookManager()

    manager.process({
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {
            "market_ticker": "TEST",
            "yes_dollars": [["0.45", "100.00"]],
            "no_dollars": [["0.54", "200.00"]]
        }
    })

    # Cancel 40 contracts
    manager.process({
        "type": "orderbook_delta",
        "sid": 1,
        "seq": 2,
        "msg": {
            "market_ticker": "TEST",
            "side": "yes",
            "price_dollars": "0.45",
            "delta_fp": "-40.00"
        }
    })

    prices = manager.get_prices("TEST")

    assert prices["yes_bid_depth"] == Decimal("60.00")


def test_remove_price_level():
    manager = OrderBookManager()

    manager.process({
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {
            "market_ticker": "TEST",
            "yes_dollars": [["0.45", "100.00"]],
            "no_dollars": [["0.54", "200.00"]]
        }
    })

    # Remove all 100 contracts
    manager.process({
        "type": "orderbook_delta",
        "sid": 1,
        "seq": 2,
        "msg": {
            "market_ticker": "TEST",
            "side": "yes",
            "price_dollars": "0.45",
            "delta_fp": "-100.00"
        }
    })

    prices = manager.get_prices("TEST")

    assert prices["yes_bid"] is None
    assert prices["yes_levels"] == 0


def test_sequence_gap_detection():
    manager = OrderBookManager()

    manager.process({
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {
            "market_ticker": "TEST",
            "yes_dollars": [["0.45", "100.00"]],
            "no_dollars": [["0.54", "200.00"]]
        }
    })

    # Skip sequence 2
    result = manager.process({
        "type": "orderbook_delta",
        "sid": 1,
        "seq": 3,
        "msg": {
            "market_ticker": "TEST",
            "side": "yes",
            "price_dollars": "0.45",
            "delta_fp": "50.00"
        }
    })

    assert result is None
    assert 1 in manager.invalid_subscriptions
    assert manager.get_prices("TEST") is None

def test_multiple_markets_same_subscription():
    manager = OrderBookManager()

    # Initial snapshot for Market A
    manager.process({
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {
            "market_ticker": "MARKET-A",
            "yes_dollars": [["0.40", "100.00"]],
            "no_dollars": [["0.58", "200.00"]]
        }
    })

    # Initial snapshot for Market B
    manager.process({
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 2,
        "msg": {
            "market_ticker": "MARKET-B",
            "yes_dollars": [["0.60", "300.00"]],
            "no_dollars": [["0.38", "150.00"]]
        }
    })

    # Update only Market A
    manager.process({
        "type": "orderbook_delta",
        "sid": 1,
        "seq": 3,
        "msg": {
            "market_ticker": "MARKET-A",
            "side": "yes",
            "price_dollars": "0.40",
            "delta_fp": "50.00"
        }
    })

    prices_a = manager.get_prices("MARKET-A")
    prices_b = manager.get_prices("MARKET-B")

    # Market A should change
    assert prices_a["yes_bid_depth"] == Decimal("150.00")
    assert prices_a["yes_bid"] == Decimal("0.40")

    # Market B should remain unchanged
    assert prices_b["yes_bid_depth"] == Decimal("300.00")
    assert prices_b["yes_bid"] == Decimal("0.60")
    assert prices_b["yes_ask"] == Decimal("0.62")

    # Both markets should remain valid
    assert len(manager.books) == 2
    assert manager.invalid_subscriptions == set()

