
"""Safety rules for market rotation."""


def apply_rotation_fallback(
    new_selected,
    previous_selected,
    rejected,
    reason,
    target_markets=10,
):
    """Apply safe fallback rules after replacement discovery.

    new_selected: retained markets plus eligible replacements
    previous_selected: markets monitored before rotation
    rejected: tickers flagged for rotation
    reason: "inactive" or "not_tradable"
    """

    # Markets rejected for lifecycle, spread, or quote-quality
    # reasons must never be restored by fallback.
    if reason in ("not_tradable", "wide_spread", "one_sided"):
        return [
            market
            for market in new_selected
            if market["ticker"] not in rejected
        ]

    if reason != "inactive":
        raise ValueError(f"Unknown rotation reason: {reason}")

    rejected = set(rejected)
    result = list(new_selected)

    # Never allow a rejected market back if Kalshi
    # confirmed that it is nontradable.
    if reason in ("not_tradable", "wide_spread"):
        result = [
            market for market in result
            if market["ticker"] not in rejected
        ]

        return result[:target_markets]

    # Inactive markets may temporarily remain if
    # no suitable replacement is available.
    existing = {m["ticker"] for m in result}

    for market in previous_selected:
        if len(result) >= target_markets:
            break

        ticker = market["ticker"]

        if ticker in rejected and ticker not in existing:
            result.append(market)
            existing.add(ticker)

    return result
