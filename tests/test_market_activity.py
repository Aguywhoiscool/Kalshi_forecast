
from src.market_activity import MarketActivityTracker


def test_inactive_market_rotates():
    tracker = MarketActivityTracker()
    tracker.register_market("TEST", now=0)

    for i in range(10):
        tracker.record_delta("TEST", now=i * 30)

    assert tracker.review_market("TEST", now=299) is False
    assert tracker.review_market("TEST", now=300) is True


def test_active_market_does_not_rotate():
    tracker = MarketActivityTracker()
    tracker.register_market("ACTIVE", now=0)

    for second in range(300):
        tracker.record_delta("ACTIVE", now=second)
        tracker.record_delta("ACTIVE", now=second + 0.1)

    assert tracker.get_rate("ACTIVE", now=300) >= 30
    assert tracker.review_market("ACTIVE", now=300) is False


def test_low_activity_with_price_changes_stays():
    tracker = MarketActivityTracker()
    tracker.register_market("MOVING", now=0)

    tracker.record_midpoint("MOVING", "0.45", now=0)
    tracker.record_midpoint("MOVING", "0.46", now=100)
    tracker.record_midpoint("MOVING", "0.47", now=200)

    assert tracker.get_midpoint_changes("MOVING", now=300) == 2
    assert tracker.review_market("MOVING", now=300) is False


def test_high_deltas_without_price_changes_stays():
    tracker = MarketActivityTracker()
    tracker.register_market("BUSY", now=0)

    tracker.record_midpoint("BUSY", "0.50", now=0)

    for second in range(300):
        tracker.record_delta("BUSY", now=second)
        tracker.record_midpoint("BUSY", "0.50", now=second)

    assert tracker.get_midpoint_changes("BUSY", now=300) == 0
    assert tracker.review_market("BUSY", now=300) is False
