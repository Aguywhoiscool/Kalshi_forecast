import numpy as np

from src.causal_swing_data import NS, align_future, entry_metadata, history_ends


def test_future_gap_does_not_remove_past_entry_or_30_second_outcome():
    times = np.arange(120, dtype=np.int64) * NS
    valid = np.ones(120, dtype=bool)
    before = history_ends(times, valid)
    times[100:] += 2 * NS  # Gap at 41s AFTER the first signal.
    after = history_ends(times, valid)
    assert 59 in before and 59 in after
    quotes = np.tile([.4, .42, 2., 2.], (120, 1))
    future, clock, available = align_future(times, quotes, valid, [59 * NS], [60])
    assert available[0, 30]
    assert not available[0, 41]
    assert np.isnan(future[0, 41]).all()
    assert clock[0, 41] == -1


def test_recording_end_keeps_entry_and_marks_future_unknown():
    times = np.arange(60, dtype=np.int64) * NS
    assert history_ends(times, np.ones(60, bool)).tolist() == [59]
    quotes = np.tile([.4, .42, 2., 2.], (60, 1))
    future, clocks, valid = align_future(times, quotes, np.ones(60, bool), [59 * NS], [60])
    assert valid[0, 0] and not valid[0, 1:].any()
    assert np.isnan(future[0, 1:]).all()
    assert (clocks[0, 1:] == -1).all()


def test_horizon_eligibility_uses_only_fixed_boundaries():
    times = np.arange(60, dtype=np.int64) * NS
    protected = {"start": 1000 * NS, "end": 1100 * NS}
    ends, meta = entry_metadata("s", "t", "train", times, np.ones(60, bool), (0, 100 * NS), protected)
    assert ends.tolist() == [59]
    assert meta.eligible_30.tolist() == [True]
    assert meta.eligible_60.tolist() == [False]


def test_bad_past_is_rejected_even_when_future_is_available():
    times = np.arange(120, dtype=np.int64) * NS
    valid = np.ones(120, bool)
    valid[20] = False
    assert 59 not in history_ends(times, valid)


def test_small_clock_jitter_does_not_invent_missing_quotes():
    times = np.arange(120, dtype=np.int64) * NS
    times[60:] -= 100_000  # Next sample is 0.1ms before nominal schedule.
    q = np.tile([.4, .42, 1., 1.], (120, 1))
    _, clock, valid = align_future(times, q, np.ones(120, bool), [59 * NS], [60])
    assert valid.all()
    assert clock[0, 1] == times[60]
