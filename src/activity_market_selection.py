
"""Safe selection of activity-qualified Kalshi replacement markets.

No networking, subscriptions, or trading actions.
"""

from src.activity_market_ranker import select_active_markets


def select_observed_replacements(
    candidates,
    cache,
    existing_markets=None,
    target_markets=10,
    now=None,
):
    """Choose replacements using only fresh, quality-checked observations.

    Args:
        candidates: REST-eligible market dictionaries.
        cache: MarketObservationCache instance.
        existing_markets: Markets already retained by collector.
        target_markets: Total desired monitored markets.
        now: Optional timezone-aware datetime for deterministic tests.

    Returns:
        Dictionary containing selected markets, additions, and shortages.
    """

    existing = list(existing_markets or [])

    if len(existing) > target_markets:
        raise ValueError(
            "Existing market count exceeds target"
        )

    existing_tickers = {
        market["ticker"] for market in existing
    }

    # Only fetch observations that are still within cache TTL.
    candidate_tickers = [
        market["ticker"]
        for market in candidates
        if market["ticker"] not in existing_tickers
    ]

    observations = cache.fresh_observations(
        candidate_tickers,
        now=now,
    )

    # This existing function applies activity, coverage,
    # movement, and series/event diversity requirements.
    selected = select_active_markets(
        candidates,
        observations,
        target_markets=target_markets,
        existing_markets=existing,
    )

    additions = [
        market for market in selected
        if market["ticker"] not in existing_tickers
    ]

    shortage = max(0, target_markets - len(selected))

    return {
        "selected": selected,
        "additions": additions,
        "shortage": shortage,
        "target_met": shortage == 0,
        "fresh_observations": len(observations),
    }
