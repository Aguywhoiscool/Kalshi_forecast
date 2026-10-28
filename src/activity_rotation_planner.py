
"""Plan market rotation without modifying the live collector."""

from src.activity_market_selection import select_observed_replacements
from src.market_rotation import apply_rotation_fallback


def plan_activity_rotation(
    previous_selected,
    rejected_tickers,
    candidates,
    cache,
    reason="inactive",
    target_markets=10,
    now=None,
):
    """Plan replacements using fresh activity observations.

    Keep existing productive markets, select active replacements,
    and apply established fallback rules for shortages.
    """

    rejected = set(rejected_tickers)

    keep = [
        market for market in previous_selected
        if market["ticker"] not in rejected
    ]

    blocked = {
        market["ticker"] for market in previous_selected
    }

    eligible_candidates = [
        market for market in candidates
        if market["ticker"] not in blocked
    ]

    result = select_observed_replacements(
        candidates=eligible_candidates,
        cache=cache,
        existing_markets=keep,
        target_markets=target_markets,
        now=now,
    )

    selected = result["selected"]

    # Use the existing production fallback only after
    # activity-qualified replacements have been selected.
    final = apply_rotation_fallback(
        new_selected=selected,
        previous_selected=previous_selected,
        rejected=rejected,
        reason=reason,
        target_markets=target_markets,
    )

    return {
        "selected": final,
        "activity_qualified_additions": result["additions"],
        "activity_shortage": result["shortage"],
        "used_fallback": len(final) > len(selected),
    }
