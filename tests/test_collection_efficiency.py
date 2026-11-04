import numpy as np

from src.analyze_collection_efficiency import boundary_saving, crossing_candidates, timing_attribution, interval_causes


def test_boundary_saving_handles_short_segments_without_double_counting():
    assert boundary_saving(300, 300) == 69
    assert boundary_saving(10, 10) == 0
    assert boundary_saving(50, 50) == 31
    assert boundary_saving(0, 100) == 0


def test_crossing_candidates_respect_real_timestamp_gap():
    a = np.arange(100, dtype=np.int64) * 1_000_000_000
    b = np.arange(100, 200, dtype=np.int64) * 1_000_000_000
    assert crossing_candidates(a, b) == 69
    assert crossing_candidates(a, b + 5_000_000_000) == 0


def test_timing_attribution_reconciles_valid_and_rejected_windows():
    times = np.arange(200, dtype=np.int64) * 1_000_000_000
    times[100:] += 3_000_000_000
    result = timing_attribution(times)
    assert result['interval_rejected'] == 69
    assert result['timing_valid'] == 62
    assert sum(result.values()) == len(times) - 69


def test_timing_attribution_detects_elapsed_drift_without_bad_intervals():
    times = np.arange(100, dtype=np.int64) * 1_100_000_000
    result = timing_attribution(times)
    assert result['interval_rejected'] == 0
    assert result['horizon_only_rejected'] == 31
    assert result['timing_valid'] == 0


def test_interval_cause_distinguishes_removed_wide_quote_from_missing_sample():
    full = np.arange(200, dtype=np.int64) * 1_000_000_000
    usable = np.delete(full, 100)
    causes = interval_causes(usable, full)
    assert causes['interval_rejected_wide_filter_only'] == 69
    assert causes['interval_rejected_recorded_gap_only'] == 0
    causes = interval_causes(usable, usable)
    assert causes['interval_rejected_recorded_gap_only'] == 69
    assert causes['interval_rejected_wide_filter_only'] == 0
