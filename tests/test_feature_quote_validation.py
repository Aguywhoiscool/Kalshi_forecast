
from decimal import Decimal

from src.feature_engineering import extract_features


def make_prices(bid, ask):
    midpoint = (
        (bid + ask) / 2
        if bid is not None and ask is not None
        else None
    )

    spread = (
        ask - bid
        if bid is not None and ask is not None
        else None
    )

    return {
        "ticker": "TEST",
        "yes_bid": bid,
        "yes_ask": ask,
        "yes_bid_depth": Decimal("100"),
        "yes_ask_depth": Decimal("100"),
        "spread": spread,
        "midpoint": midpoint,
    }


def test_valid_quote_is_recorded():
    prices = make_prices(
        Decimal("0.40"),
        Decimal("0.42"),
    )

    result = extract_features(prices)

    assert result is not None
    assert result["midpoint"] == 0.41


def test_missing_bid_is_skipped():
    prices = make_prices(
        None,
        Decimal("0.42"),
    )

    assert extract_features(prices) is None


def test_crossed_book_is_skipped():
    prices = make_prices(
        Decimal("0.60"),
        Decimal("0.40"),
    )

    assert extract_features(prices) is None


def test_zero_spread_is_skipped():
    prices = make_prices(
        Decimal("0.40"),
        Decimal("0.40"),
    )

    assert extract_features(prices) is None


def test_wide_spread_is_preserved():
    prices = make_prices(
        Decimal("0.05"),
        Decimal("0.49"),
    )

    result = extract_features(prices)

    assert result is not None
    assert result["spread"] == 0.44

