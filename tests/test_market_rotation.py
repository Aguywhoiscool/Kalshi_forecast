
import pytest

from src.market_rotation import apply_rotation_fallback


def market(ticker):
    return {"ticker": ticker}


def test_closed_market_removed_without_replacement():
    """Closed market must not remain when no replacement exists."""

    original = [market("ACTIVE"), market("CLOSED")]

    result = apply_rotation_fallback(
        new_selected=[market("ACTIVE")],
        previous_selected=original,
        rejected={"CLOSED"},
        reason="not_tradable",
        target_markets=2,
    )

    assert len(result) == 1
    assert [m["ticker"] for m in result] == ["ACTIVE"]


def test_closed_market_replaced_when_candidate_exists():
    """An eligible replacement should fill the empty slot."""

    original = [market("ACTIVE"), market("CLOSED")]

    result = apply_rotation_fallback(
        new_selected=[market("ACTIVE"), market("NEW")],
        previous_selected=original,
        rejected={"CLOSED"},
        reason="not_tradable",
        target_markets=2,
    )

    assert [m["ticker"] for m in result] == [
        "ACTIVE",
        "NEW",
    ]


def test_inactive_market_retained_without_replacement():
    """Quiet but tradable markets may remain temporarily."""

    original = [market("ACTIVE"), market("QUIET")]

    result = apply_rotation_fallback(
        new_selected=[market("ACTIVE")],
        previous_selected=original,
        rejected={"QUIET"},
        reason="inactive",
        target_markets=2,
    )

    assert len(result) == 2
    assert result[1]["ticker"] == "QUIET"


def test_nontradable_market_never_reintroduced():
    """Even an accidentally included rejected market is removed."""

    result = apply_rotation_fallback(
        new_selected=[
            market("ACTIVE"),
            market("CLOSED"),
            market("NEW"),
        ],
        previous_selected=[
            market("ACTIVE"),
            market("CLOSED"),
        ],
        rejected={"CLOSED"},
        reason="not_tradable",
        target_markets=3,
    )

    tickers = [m["ticker"] for m in result]

    assert "CLOSED" not in tickers
    assert tickers == ["ACTIVE", "NEW"]


def test_unknown_reason_is_rejected():
    with pytest.raises(ValueError):
        apply_rotation_fallback(
            new_selected=[],
            previous_selected=[],
            rejected=set(),
            reason="unknown",
        )

def test_not_tradable_never_reintroduced():
    from src.market_rotation import apply_rotation_fallback

    previous = [
        {"ticker": "MARKET-A"},
        {"ticker": "MARKET-B"},
    ]

    result = apply_rotation_fallback(
        new_selected=[previous[1]],
        previous_selected=previous,
        rejected={"MARKET-A"},
        reason="not_tradable",
        target_markets=2,
    )

    tickers = {market["ticker"] for market in result}

    assert "MARKET-A" not in tickers
    assert "MARKET-B" in tickers
