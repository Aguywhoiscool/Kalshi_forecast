
from src.one_sided_quote_tracker import OneSidedQuoteTracker


def book(bid, ask):
    return {
        "yes_bid": bid,
        "yes_ask": ask,
    }


def test_initial_snapshot_not_received():
    tracker = OneSidedQuoteTracker()

    status = tracker.update(
        "TEST", False, None, now=100
    )

    assert status["should_rotate"] is False
    assert status["one_sided"] is False


def test_one_sided_quote_rotates_after_60_seconds():
    tracker = OneSidedQuoteTracker()

    tracker.update(
        "TEST", True, book(None, 0.40), now=100
    )

    status = tracker.update(
        "TEST", True, book(None, 0.40), now=159
    )

    assert status["should_rotate"] is False

    status = tracker.update(
        "TEST", True, book(None, 0.40), now=160
    )

    assert status["should_rotate"] is True
    assert status["one_sided_seconds"] == 60


def test_recovery_resets_timer():
    tracker = OneSidedQuoteTracker()

    tracker.update(
        "TEST", True, book(None, 0.40), now=100
    )

    tracker.update(
        "TEST", True, book(0.30, 0.40), now=150
    )

    tracker.update(
        "TEST", True, book(None, 0.40), now=180
    )

    status = tracker.get_status("TEST", now=200)

    assert status["one_sided_seconds"] == 20
    assert status["should_rotate"] is False


def test_missing_ask_is_one_sided():
    tracker = OneSidedQuoteTracker()

    tracker.update(
        "TEST", True, book(0.30, None), now=0
    )

    assert tracker.get_status(
        "TEST", now=60
    )["should_rotate"] is True


def test_markets_are_independent():
    tracker = OneSidedQuoteTracker()

    tracker.update(
        "A", True, book(None, 0.40), now=100
    )

    tracker.update(
        "B", True, book(0.30, 0.40), now=100
    )

    assert tracker.get_status(
        "A", now=161
    )["should_rotate"] is True

    assert tracker.get_status(
        "B", now=161
    )["should_rotate"] is False


def test_no_snapshot_does_not_start_timer():
    tracker = OneSidedQuoteTracker()

    tracker.update(
        "TEST", False, None, now=100
    )

    assert tracker.get_status(
        "TEST", now=500
    )["should_rotate"] is False
