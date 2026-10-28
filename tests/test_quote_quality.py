
from decimal import Decimal

from src.quote_quality import classify_quote


def prices(bid, ask, midpoint):
    return {
        "yes_bid": bid,
        "yes_ask": ask,
        "midpoint": midpoint,
    }


def test_narrow_quote_is_training_quality():
    result = classify_quote(
        prices(
            Decimal("0.40"),
            Decimal("0.42"),
            Decimal("0.41"),
        )
    )

    assert result["structurally_valid"] is True
    assert result["two_sided"] is True
    assert result["training_quality"] is True
    assert result["spread_cents"] == 2.0
    assert result["reason"] == "usable"


def test_missing_bid_is_one_sided():
    result = classify_quote(
        prices(None, Decimal("0.04"), None)
    )

    assert result["structurally_valid"] is True
    assert result["two_sided"] is False
    assert result["training_quality"] is False
    assert result["reason"] == "one_sided"


def test_wide_spread_is_not_training_quality():
    result = classify_quote(
        prices(
            Decimal("0.05"),
            Decimal("0.49"),
            Decimal("0.27"),
        )
    )

    assert result["two_sided"] is True
    assert result["training_quality"] is False
    assert result["spread_cents"] == 44.0
    assert result["reason"] == "wide_spread"


def test_invalid_book():
    result = classify_quote(None)

    assert result["structurally_valid"] is False
    assert result["training_quality"] is False
    assert result["reason"] == "invalid_book"


def test_invalid_crossed_book():
    result = classify_quote(
        prices(
            Decimal("0.60"),
            Decimal("0.40"),
            Decimal("0.50"),
        )
    )

    assert result["training_quality"] is False
    assert result["reason"] == "invalid_quote"


def test_inconsistent_midpoint():
    result = classify_quote(
        prices(
            Decimal("0.40"),
            Decimal("0.42"),
            Decimal("0.50"),
        )
    )

    assert result["training_quality"] is False
    assert result["reason"] == "invalid_quote"


def test_nonfinite_quote_is_rejected():
    result = classify_quote(
        prices(
            Decimal("NaN"),
            Decimal("0.42"),
            Decimal("0.41"),
        )
    )

    assert result["training_quality"] is False
    assert result["reason"] == "invalid_quote"


def test_spread_at_limit_is_accepted():
    result = classify_quote(
        prices(
            Decimal("0.40"),
            Decimal("0.50"),
            Decimal("0.45"),
        )
    )

    assert result["training_quality"] is True
    assert result["spread_cents"] == 10.0
