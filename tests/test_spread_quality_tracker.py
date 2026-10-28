
from src.spread_quality_tracker import SpreadQualityTracker


def test_persistent_wide_spread_rotates():
    tracker = SpreadQualityTracker()

    ticker = "TEST-WIDE"

    for second in range(0, 61, 10):
        tracker.record_spread(ticker, 20.0, now=second)

    result = tracker.get_status(ticker, now=60)

    assert result["should_rotate"] is True
    assert result["wide_seconds"] == 60


def test_short_spread_spike_does_not_rotate():
    tracker = SpreadQualityTracker()

    ticker = "TEST-SPIKE"

    tracker.record_spread(ticker, 20.0, now=0)
    tracker.record_spread(ticker, 20.0, now=10)
    tracker.record_spread(ticker, 2.0, now=20)
    tracker.record_spread(ticker, 20.0, now=30)
    tracker.record_spread(ticker, 20.0, now=40)

    assert tracker.get_status(
        ticker, now=40
    )["should_rotate"] is False


def test_stale_spread_does_not_rotate():
    tracker = SpreadQualityTracker()

    ticker = "TEST-STALE"

    tracker.record_spread(ticker, 44.0, now=0)

    result = tracker.get_status(ticker, now=120)

    assert result["stale"] is True
    assert result["should_rotate"] is False


def test_gap_in_updates_resets_duration():
    tracker = SpreadQualityTracker()

    ticker = "TEST-GAP"

    tracker.record_spread(ticker, 25.0, now=0)
    tracker.record_spread(ticker, 25.0, now=10)

    # Fifty seconds without an update breaks continuity.
    tracker.record_spread(ticker, 25.0, now=60)

    result = tracker.get_status(ticker, now=60)

    assert result["wide_seconds"] == 0
    assert result["should_rotate"] is False


def test_narrow_spread_never_rotates():
    tracker = SpreadQualityTracker()

    ticker = "TEST-NARROW"

    for second in range(0, 301, 10):
        tracker.record_spread(ticker, 3.0, now=second)

    assert tracker.get_status(
        ticker, now=300
    )["should_rotate"] is False


def test_invalid_spread_resets_warning():
    tracker = SpreadQualityTracker()

    ticker = "TEST-INVALID"

    tracker.record_spread(ticker, 30.0, now=0)
    tracker.record_spread(ticker, None, now=10)

    assert tracker.get_status(
        ticker, now=10
    )["should_rotate"] is False
