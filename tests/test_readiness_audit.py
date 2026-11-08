import numpy as np
import pandas as pd
import pytest

from src.readiness_audit import assess, scan_frame, single_run


def frame():
    stamps = pd.date_range("2026-11-07T05:00:00Z", periods=300, freq="s")
    close = pd.Timestamp("2026-11-07T05:45:00Z")
    return pd.DataFrame({"timestamp": stamps, "session_id": "s", "ticker": "EVENT-YES",
        "bid": .4, "ask": .42, "spread": .02, "midpoint": .41,
        "bid_depth": 2., "ask_depth": 2., "imbalance": 0.,
        "scheduled_close_time": close.isoformat(), "close_time_observed_at": "2026-11-07T04:59:00Z",
        "seconds_to_scheduled_close": (close-stamps).total_seconds()})


def scan(data):
    return scan_frame(data, pd.Timestamp("2026-11-07T04:00:00Z"), {"start": 0, "end": 1, "sessions": []})


def test_counts_spaced_windows_with_verified_full_history():
    result = scan(frame())
    assert result["verified_rows"] == 300
    assert len(result["windows"]) == 3
    assert np.diff([w["time"] for w in result["windows"]]).tolist() == [120*10**9]*2


def test_unknown_or_future_schedule_cannot_enter_history():
    data = frame()
    data.loc[:59, "close_time_observed_at"] = None
    assert scan(data)["windows"][0]["time"] == data.timestamp.iloc[119].value
    data.loc[60:119, "close_time_observed_at"] = "2026-11-07T06:00:00Z"
    result = scan(data)
    assert result["future_knowledge_rows"] == 60
    assert result["windows"][0]["time"] == data.timestamp.iloc[179].value


def test_gap_and_session_boundary_are_not_stitched():
    data = frame().iloc[:100].copy()
    data.loc[:49, "session_id"] = "other"
    assert not scan(data)["windows"]
    data = frame().drop(index=30)
    assert scan(data)["windows"][0]["time"] == frame().timestamp.iloc[90].value


def test_bad_countdown_is_a_violation_and_pilot_screen_is_not_ready():
    data = frame()
    data.loc[20, "seconds_to_scheduled_close"] += 1
    assert scan(data)["countdown_errors"] == 1
    windows = [{"session": "s", "event_proxy": "e", "date": "2026-11-07"}]
    assert assess(windows, 1, [], 0, 1, 1, 1, 1)["status"] == "collecting"
    assert assess(windows, 0, [], 0, 1, 1, 1, 1)["status"] == "ready_for_pilot_review"
    assert assess(windows, 0, [], 4000, 1, 1, 1, 1)["status"] == "collecting"


def test_reserved_sessions_and_periods_are_excluded():
    data = frame()
    result = scan_frame(data, pd.Timestamp("2026-11-07T04:00:00Z"),
        {"start": 0, "end": 1, "sessions": ["s"]})
    assert result["rows"] == 0 and not result["windows"]


def test_duplicate_audit_is_blocked_and_lock_releases(tmp_path):
    path = tmp_path / "audit.lock"
    with single_run(path):
        with pytest.raises(BlockingIOError):
            with single_run(path):
                raise AssertionError("Duplicate audit acquired lock")
    with single_run(path):
        pass
