
from src.market_activity import MarketActivityTracker
from src.market_rotation import apply_rotation_fallback

def test_high_deltas_without_movement_rotates():
    tracker = MarketActivityTracker()

    tracker.register_market("TEST", now=0)

    for i in range(400):
        tracker.record_delta("TEST", now=1 + i * 0.5)

    assert tracker.get_rate("TEST", now=301) >= 30
    assert tracker.review_market("TEST", now=301) is True


def test_low_deltas_without_movement_rotates():
    tracker = MarketActivityTracker()

    tracker.register_market("TEST", now=0)
    tracker.record_delta("TEST", now=10)

    assert tracker.review_market("TEST", now=301) is True


def test_moving_market_is_retained():
    tracker = MarketActivityTracker()

    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", 0.50, now=100)
    tracker.record_midpoint("TEST", 0.51, now=200)

    assert tracker.review_market("TEST", now=301) is False


def test_no_rotation_before_five_minutes():
    tracker = MarketActivityTracker()

    tracker.register_market("TEST", now=0)

    assert tracker.review_market("TEST", now=299) is False

def test_wide_spread_market_cannot_return():
    previous = [
        {"ticker": "WIDE"},
        {"ticker": "HEALTHY"},
    ]

    result = apply_rotation_fallback(
        new_selected=[{"ticker": "HEALTHY"}],
        previous_selected=previous,
        rejected={"WIDE"},
        reason="wide_spread",
        target_markets=2,
    )

    assert [m["ticker"] for m in result] == ["HEALTHY"]


def test_inactive_fallback_still_works():
    previous = [
        {"ticker": "INACTIVE"},
        {"ticker": "HEALTHY"},
    ]

    result = apply_rotation_fallback(
        new_selected=[{"ticker": "HEALTHY"}],
        previous_selected=previous,
        rejected={"INACTIVE"},
        reason="inactive",
        target_markets=2,
    )

    assert {m["ticker"] for m in result} == {
        "HEALTHY",
        "INACTIVE",
    }


def test_mixed_rejections_are_not_restored():
    previous = [
        {"ticker": "WIDE"},
        {"ticker": "INACTIVE"},
        {"ticker": "HEALTHY"},
    ]

    result = apply_rotation_fallback(
        new_selected=[{"ticker": "HEALTHY"}],
        previous_selected=previous,
        rejected={"WIDE", "INACTIVE"},
        reason="wide_spread",
        target_markets=3,
    )

    assert [m["ticker"] for m in result] == ["HEALTHY"]