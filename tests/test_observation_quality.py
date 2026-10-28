
from src.observation_quality import BookCoverageTracker


def test_book_valid_from_beginning():
    tracker = BookCoverageTracker(
        ["TEST"], start_time=0
    )

    tracker.update("TEST", True, now=0)

    result = tracker.summarize("TEST", end_time=100)

    assert result["valid_seconds"] == 100
    assert result["valid_coverage"] == 1.0


def test_book_becomes_valid_late():
    tracker = BookCoverageTracker(
        ["TEST"], start_time=0
    )

    tracker.update("TEST", True, now=40)

    result = tracker.summarize("TEST", end_time=100)

    assert result["valid_seconds"] == 60
    assert result["valid_coverage"] == 0.6
    assert result["first_valid_delay"] == 40


def test_book_becomes_invalid():
    tracker = BookCoverageTracker(
        ["TEST"], start_time=0
    )

    tracker.update("TEST", True, now=10)
    tracker.update("TEST", False, now=60)

    result = tracker.summarize("TEST", end_time=100)

    assert result["valid_seconds"] == 50
    assert result["valid_coverage"] == 0.5


def test_book_never_becomes_valid():
    tracker = BookCoverageTracker(
        ["TEST"], start_time=0
    )

    result = tracker.summarize("TEST", end_time=100)

    assert result["valid_seconds"] == 0
    assert result["valid_coverage"] == 0
    assert result["first_valid_delay"] is None
