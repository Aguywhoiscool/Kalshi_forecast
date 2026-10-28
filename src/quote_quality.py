
"""Classify order books by their usefulness for model training."""

from decimal import Decimal, InvalidOperation


MAX_TRAINING_SPREAD_CENTS = Decimal("10")


def classify_quote(prices, max_spread_cents=10):
    """Classify a reconstructed Kalshi YES-side order book.

    Prices are expressed in dollars (0 to 1).
    Spread limits are expressed in cents.
    """

    result = {
        "structurally_valid": prices is not None,
        "two_sided": False,
        "training_quality": False,
        "spread_cents": None,
        "reason": "invalid_book",
    }

    if prices is None:
        return result

    bid = prices.get("yes_bid")
    ask = prices.get("yes_ask")
    midpoint = prices.get("midpoint")

    if bid is None or ask is None or midpoint is None:
        result["reason"] = "one_sided"
        return result

    try:
        bid = Decimal(str(bid))
        ask = Decimal(str(ask))
        midpoint = Decimal(str(midpoint))
        maximum = Decimal(str(max_spread_cents))
    except (InvalidOperation, ValueError, TypeError):
        result["reason"] = "invalid_quote"
        return result

    if not all(x.is_finite() for x in (bid, ask, midpoint, maximum)):
        result["reason"] = "invalid_quote"
        return result

    if (
        bid < 0
        or ask > 1
        or bid >= ask
        or midpoint != (bid + ask) / 2
        or maximum <= 0
    ):
        result["reason"] = "invalid_quote"
        return result

    spread_cents = (ask - bid) * 100

    result["two_sided"] = True
    result["spread_cents"] = float(spread_cents)

    if spread_cents > maximum:
        result["reason"] = "wide_spread"
        return result

    result["training_quality"] = True
    result["reason"] = "usable"

    return result
