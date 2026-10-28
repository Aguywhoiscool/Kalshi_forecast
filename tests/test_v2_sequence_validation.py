
from datetime import datetime, timedelta, timezone


MAX_SPREAD = 0.10
TARGET_TOLERANCE = 0.5


def valid_quote(bid, ask):
    spread = ask - bid

    return (
        0 <= bid < ask <= 1
        and 0 < spread <= MAX_SPREAD
    )


def valid_sequence(times):
    if len(times) != 70:
        return False

    intervals = [
        (current - previous).total_seconds()
        for previous, current in zip(times, times[1:])
    ]

    if not all(0.5 <= dt <= 1.5 for dt in intervals):
        return False

    history = (times[59] - times[0]).total_seconds()
    horizon = (times[69] - times[59]).total_seconds()

    return (
        abs(history - 59) <= 1.5
        and abs(horizon - 10) <= TARGET_TOLERANCE
    )


def make_times():
    start = datetime(
        2026, 10, 28, tzinfo=timezone.utc
    )

    return [
        start + timedelta(seconds=i)
        for i in range(70)
    ]


def test_valid_narrow_spread():
    assert valid_quote(0.40, 0.42)


def test_wide_spread_rejected():
    assert not valid_quote(0.20, 0.40)


def test_complete_sequence():
    assert valid_sequence(make_times())


def test_missing_second_rejected():
    times = make_times()

    for i in range(30, len(times)):
        times[i] += timedelta(seconds=1)

    assert not valid_sequence(times)


def test_incorrect_target_horizon_rejected():
    times = make_times()

    for i in range(60, len(times)):
        times[i] += timedelta(seconds=1)

    assert not valid_sequence(times)
