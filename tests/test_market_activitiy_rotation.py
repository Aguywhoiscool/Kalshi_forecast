def test_one_sided_rotation_is_accepted():
    from src.market_rotation import apply_rotation_fallback

    previous = [
        {"ticker": "GOOD"},
        {"ticker": "ONE_SIDED"},
    ]

    selected = [
        {"ticker": "GOOD"},
    ]

    result = apply_rotation_fallback(
        new_selected=selected,
        previous_selected=previous,
        rejected={"ONE_SIDED"},
        reason="one_sided",
        target_markets=2,
    )

    assert [market["ticker"] for market in result] == ["GOOD"]