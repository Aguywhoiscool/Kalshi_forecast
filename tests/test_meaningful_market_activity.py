
from src.market_activity import MarketActivityTracker


def test_small_midpoint_change_is_not_meaningful():
    tracker = MarketActivityTracker()
    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", "0.500", now=1)
    tracker.record_midpoint("TEST", "0.502", now=2)

    assert tracker.get_midpoint_changes("TEST", now=3) == 1
    assert tracker.get_meaningful_midpoint_changes("TEST", now=3) == 0


def test_half_cent_change_is_meaningful():
    tracker = MarketActivityTracker()
    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", "0.500", now=1)
    tracker.record_midpoint("TEST", "0.505", now=2)

    assert tracker.get_midpoint_changes("TEST", now=3) == 1
    assert tracker.get_meaningful_midpoint_changes("TEST", now=3) == 1


def test_multiple_meaningful_changes():
    tracker = MarketActivityTracker()
    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", "0.50", now=1)
    tracker.record_midpoint("TEST", "0.51", now=2)
    tracker.record_midpoint("TEST", "0.52", now=3)
    tracker.record_midpoint("TEST", "0.520", now=4)

    assert tracker.get_midpoint_changes("TEST", now=5) == 2
    assert tracker.get_meaningful_midpoint_changes("TEST", now=5) == 2


def test_old_events_expire():
    tracker = MarketActivityTracker(window_seconds=300)
    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", "0.50", now=1)
    tracker.record_midpoint("TEST", "0.51", now=2)

    assert tracker.get_meaningful_midpoint_changes(
        "TEST", now=10
    ) == 1

    assert tracker.get_meaningful_midpoint_changes(
        "TEST", now=303
    ) == 0


def test_missing_midpoint_does_not_create_movement():
    tracker = MarketActivityTracker()
    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", "0.50", now=1)
    tracker.record_midpoint("TEST", None, now=2)
    tracker.record_midpoint("TEST", "0.60", now=3)

    assert tracker.get_midpoint_changes("TEST", now=4) == 0
    assert tracker.get_meaningful_midpoint_changes(
        "TEST", now=4
    ) == 0


def test_existing_rotation_behavior_is_preserved():
    tracker = MarketActivityTracker()
    tracker.register_market("TEST", now=0)

    tracker.record_midpoint("TEST", "0.500", now=1)
    tracker.record_midpoint("TEST", "0.502", now=200)

    # A small movement still prevents rotation under
    # the current rules, even though it isn't meaningful.
    assert tracker.get_meaningful_midpoint_changes(
        "TEST", now=301
    ) == 0

    assert tracker.review_market("TEST", now=301) is False
