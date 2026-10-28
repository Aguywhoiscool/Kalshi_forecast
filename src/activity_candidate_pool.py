
"""Spread-aware selection of observed Kalshi market candidates."""

from src.activity_market_selection import select_observed_replacements


PREFERRED_SPREAD_CENTS = 5.0
MAX_SPREAD_CENTS = 10.0


def spread_quality(observation):
    """Classify an observation using median and last recorded spreads."""

    if not observation:
        return "unknown"

    values = []

    for field in ("spread_cents", "last_spread_cents"):
        value = observation.get(field)

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            return "unknown"

        if not (0 < value < float("inf")):
            return "unknown"

        values.append(value)

    if not values:
        return "unknown"

    if max(values) > MAX_SPREAD_CENTS:
        return "reject"

    if max(values) > PREFERRED_SPREAD_CENTS:
        return "secondary"

    return "preferred"


def prioritize_rotation_candidates(
    candidates,
    cache,
    existing_markets,
    blocked_tickers,
    target_markets=10,
    now=None,
):
    """Prefer recently moving markets with reasonable spreads."""

    blocked = set(blocked_tickers or [])

    eligible = []
    rejected_spread = []
    quality_by_ticker = {}

    for market in candidates:
        ticker = market["ticker"]

        if ticker in blocked:
            continue

        observation = cache.get(ticker, now=now)
        quality = spread_quality(observation)

        quality_by_ticker[ticker] = quality

        if quality == "reject":
            rejected_spread.append(ticker)
            continue

        eligible.append(market)

    result = select_observed_replacements(
        candidates=eligible,
        cache=cache,
        existing_markets=existing_markets,
        target_markets=target_markets,
        now=now,
    )

    # The existing selector performs activity and diversity filtering.
    qualified = result["additions"]

    preferred = []
    secondary = []

    for market in qualified:
        quality = quality_by_ticker[market["ticker"]]

        if quality == "preferred":
            preferred.append(market)
        elif quality == "secondary":
            secondary.append(market)

    ordered_qualified = preferred + secondary

    qualified_tickers = {
        market["ticker"] for market in ordered_qualified
    }

    # Preserve original REST ranking for fallback candidates.
    fallback = [
        market
        for market in eligible
        if market["ticker"] not in qualified_tickers
    ]

    return {
        "candidates": ordered_qualified + fallback,
        "qualified_tickers": qualified_tickers,
        "qualified_count": len(ordered_qualified),
        "preferred_count": len(preferred),
        "secondary_count": len(secondary),
        "spread_rejected_count": len(rejected_spread),
        "spread_rejected_tickers": rejected_spread,
        "activity_shortage": max(
            0, result["shortage"] +
            (len(qualified) - len(ordered_qualified)),
        ),
    }
