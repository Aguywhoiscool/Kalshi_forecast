
"""Validate whether a Kalshi market is suitable for collection."""

from datetime import datetime, timezone


TRADABLE_STATUSES = {"active"}


def market_is_tradable(market, min_minutes_to_close=10):
    """Return True only for confirmed active markets."""

    if market.get("status") not in TRADABLE_STATUSES:
        return False

    close_time = market.get("close_time")

    if not close_time:
        return False

    try:
        closing = datetime.fromisoformat(
            close_time.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        return False

    if closing.tzinfo is None:
        return False

    now = datetime.now(timezone.utc)

    minutes_remaining = (
        closing - now
    ).total_seconds() / 60

    return minutes_remaining > min_minutes_to_close
