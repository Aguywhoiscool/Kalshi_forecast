import numpy as np
import pandas as pd
import pytest

from src.v3_research import candidate_ends, overlaps, spaced, targets, valid_rows


def test_windows_reject_internal_gap_and_target_drift():
    times = np.arange(100, dtype=np.int64) * 1_000_000_000
    assert np.array_equal(candidate_ends(times), np.arange(59, 90))
    times[65:] += 2_000_000_000
    assert len(candidate_ends(times)) == 0
    # Every adjacent interval is allowed, but ten steps are not ten seconds.
    assert len(candidate_ends(np.arange(100, dtype=np.int64) * 1_200_000_000)) == 0


def test_protection_excludes_touching_and_spanning_windows():
    assert overlaps(np.array([0, 10, 21]), np.array([10, 30, 40]),
                    {"start": 10, "end": 20}).tolist() == [True, True, False]


def test_spaced_windows_have_no_shared_input_or_target_timestamp():
    frame = pd.DataFrame({"session_id": ["s"] * 3, "ticker": ["t"] * 3,
        "input_start": [0, 69, 70], "input_end": [59, 128, 129],
        "target_end": [69, 138, 139]})
    assert spaced(frame).input_start.tolist() == [0, 70]


def test_holdout_targets_cannot_be_computed():
    with pytest.raises(ValueError, match="holdout"):
        targets(pd.DataFrame({"split": ["holdout"]}), None)


def test_quote_validation_checks_all_features_and_depth_imbalance():
    frame = pd.DataFrame({"bid": [.4] * 3, "ask": [.42] * 3,
        "spread": [.02] * 3, "midpoint": [.41] * 3,
        "bid_depth": [2.] * 3, "ask_depth": [2.] * 3,
        "imbalance": [0., float("inf"), .5],
        "timestamp": pd.to_datetime(["2026-01-01"] * 3, utc=True),
        "session_id": ["s"] * 3, "ticker": ["t"] * 3})
    assert valid_rows(frame).tolist() == [True, False, False]
