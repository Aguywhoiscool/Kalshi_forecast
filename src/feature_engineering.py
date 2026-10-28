
from decimal import Decimal


def extract_features(prices):
    """Convert reconstructed order-book prices into numerical features."""

    if prices is None:
        return None

    bid = prices["yes_bid"]
    ask = prices["yes_ask"]
    bid_depth = prices["yes_bid_depth"]
    ask_depth = prices["yes_ask_depth"]

    # A complete two-sided quote is required for these features.
    if any(value is None for value in (
        bid, ask, bid_depth, ask_depth
    )):
        return None

        # Require a valid, strictly positive bid-ask spread.
    if not (
        Decimal("0") <= bid < ask <= Decimal("1")
    ):
        return None

    # Reject invalid or missing midpoint/spread values.
    if (
        prices["midpoint"] is None
        or prices["spread"] is None
        or prices["spread"] <= 0
    ):
        return None

    total_depth = bid_depth + ask_depth

    imbalance = (
        (bid_depth - ask_depth) / total_depth
        if total_depth > 0
        else Decimal("0")
    )

    return {
        "ticker": prices["ticker"],
        "bid": float(bid),
        "ask": float(ask),
        "spread": float(prices["spread"]),
        "midpoint": float(prices["midpoint"]),
        "bid_depth": float(bid_depth),
        "ask_depth": float(ask_depth),
        "imbalance": float(imbalance),
    }
