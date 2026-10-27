
from decimal import Decimal

from src.feature_engineering import extract_features


def test_extract_features():
    prices = {
        "ticker": "TEST-MARKET",
        "yes_bid": Decimal("0.45"),
        "yes_ask": Decimal("0.46"),
        "spread": Decimal("0.01"),
        "midpoint": Decimal("0.455"),
        "yes_bid_depth": Decimal("300"),
        "yes_ask_depth": Decimal("100"),
    }

    features = extract_features(prices)

    assert features["ticker"] == "TEST-MARKET"
    assert features["bid"] == 0.45
    assert features["ask"] == 0.46
    assert features["midpoint"] == 0.455
    assert features["spread"] == 0.01
    assert features["bid_depth"] == 300.0
    assert features["ask_depth"] == 100.0
    assert features["imbalance"] == 0.5


def test_missing_ask_returns_none():
    prices = {
        "ticker": "TEST-MARKET",
        "yes_bid": Decimal("0.45"),
        "yes_ask": None,
        "spread": None,
        "midpoint": None,
        "yes_bid_depth": Decimal("100"),
        "yes_ask_depth": None,
    }

    assert extract_features(prices) is None
