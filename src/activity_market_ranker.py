
"""Rank observed Kalshi markets using live activity measurements.

This module performs no API requests or trading actions.
"""

from collections import Counter
import math

from src.market_ranker import series_of, MAX_PER_SERIES


def calculate_activity_score(observation):
    """
    Score one market using observed live activity.

    Expected observation fields:
        ticker
        duration_seconds
        delta_count
        midpoint_changes
        meaningful_changes
        spread_cents
        valid_book
    """

    if not observation.get("valid_book", False):
        return float("-inf")
    coverage = observation.get("valid_coverage", 0.0)

    if coverage < 0.90:
        return float("-inf")

    duration = float(observation.get("duration_seconds", 0))

    if duration <= 0:
        return float("-inf")

    spread = float(observation.get("spread_cents", 100))
    deltas = int(observation.get("delta_count", 0))
    changes = int(observation.get("midpoint_changes", 0))
    meaningful = int(observation.get("meaningful_changes", 0))

    if not math.isfinite(spread) or spread < 0:
        return float("-inf")

    minutes = duration / 60.0

    delta_rate = deltas / minutes
    change_rate = changes / minutes
    meaningful_rate = meaningful / minutes

    # Meaningful movements contribute the most.
    movement_score = 5.0 * math.log1p(meaningful_rate)

    # Any midpoint movement provides secondary evidence.
    midpoint_score = 1.5 * math.log1p(change_rate)

    # Deltas alone should never dominate selection.
    delta_score = 0.25 * math.log1p(delta_rate)

    # Penalize wide quoted spreads.
    spread_penalty = 0.75 * spread

    return (
        movement_score
        + midpoint_score
        + delta_score
        - spread_penalty
    )


def select_active_markets(
    candidates,
    observations,
    target_markets=10,
    existing_markets=None,
):
    """Select observed candidates while preserving diversity.

    candidates: REST market dictionaries
    observations: dictionary keyed by market ticker
    existing_markets: markets already retained by collector
    """

    selected = list(existing_markets or [])

    selected_tickers = {
        market["ticker"] for market in selected
    }

    series_counts = Counter(
        series_of(market) for market in selected
    )

    used_events = {
        market.get("event_ticker") or market["ticker"]
        for market in selected
    }

    ranked = []

    for market in candidates:
        ticker = market["ticker"]

        if ticker in selected_tickers:
            continue

        observation = observations.get(ticker)

        if observation is None:
            continue
        # Do not select a market solely because it receives
        # WebSocket deltas or has a narrow bid-ask spread.
        if observation.get("midpoint_changes", 0) == 0:
            continue
        score = calculate_activity_score(observation)

        if not math.isfinite(score):
            continue

        ranked.append((score, market))

    ranked.sort(key=lambda item: item[0], reverse=True)

    for score, market in ranked:
        if len(selected) >= target_markets:
            break

        ticker = market["ticker"]
        series = series_of(market)
        event = market.get("event_ticker") or ticker

        if ticker in selected_tickers:
            continue

        if series_counts[series] >= MAX_PER_SERIES:
            continue

        if event in used_events:
            continue

        selected.append(market)

        selected_tickers.add(ticker)
        series_counts[series] += 1
        used_events.add(event)

    return selected
