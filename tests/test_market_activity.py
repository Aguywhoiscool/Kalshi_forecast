
from src.market_activity import MarketActivityTracker



def test_market_rotation_threshold():
    tracker = MarketActivityTracker(
        window_seconds=300,
        min_deltas_per_minute=30
    )

    tracker.register_market("TEST", now=0)

    # Only 10 updates during the first five minutes.
    for i in range(10):
        tracker.record_delta("TEST", now=i * 30)

    # Not eligible before five minutes.
    assert tracker.review_market("TEST", now=299) is False

    # Eligible for rotation at five minutes.
    assert tracker.review_market("TEST", now=300) is True



def test_active_market_is_not_rotated():
    tracker = MarketActivityTracker(
        window_seconds=300,
        min_deltas_per_minute=60
    )

    tracker.register_market("ACTIVE", now=0)

    # 100 updates per minute during the last five minutes.
    for second in range(300, 600):
        tracker.record_delta("ACTIVE", now=second)
        tracker.record_delta("ACTIVE", now=second + 0.1)

    assert tracker.review_market("ACTIVE", now=600) is False
    assert tracker.get_rate("ACTIVE", now=600) >= 60
