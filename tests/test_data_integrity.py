import json

import numpy as np
import pandas as pd
import pytest

from src.check_data_integrity import check_raw_message, replay_comparison, row_checks, sampled_lines, timing_counts


def example():
    return pd.DataFrame({"timestamp": ["2026-11-04T16:15:00Z"], "session_id": ["s"],
        "ticker": ["t"], "bid": [.4], "ask": [.42], "spread": [.02], "midpoint": [.41],
        "bid_depth": [3.], "ask_depth": [1.], "imbalance": [.5]})


def test_valid_feature_and_wide_quote_are_not_corruption():
    frame = example()
    _, result, _ = row_checks(frame)
    assert result["invalid_rows"] == 0
    assert result["usable_rows"] == 1
    frame.loc[0, ["ask", "spread", "midpoint"]] = [.6, .2, .5]
    _, result, _ = row_checks(frame)
    assert result["invalid_rows"] == 0
    assert result["wide_rows"] == 1


def test_all_features_identity_and_timezone_are_validated():
    for field, value in [("bid_depth", -1), ("imbalance", .2), ("spread", .03),
                         ("midpoint", np.inf), ("timestamp", "2026-11-04T16:15:00"),
                         ("ticker", None)]:
        frame = example()
        frame[field] = value
        _, result, examples = row_checks(frame)
        assert result["invalid_rows"] == 1, field
        assert len(examples) == 1


def test_duplicates_conflicts_and_original_time_order_are_distinct():
    frame = pd.concat([example()] * 2, ignore_index=True)
    _, counts, _ = row_checks(frame)
    assert counts["duplicate_rows"] == 1
    assert counts["conflicting_duplicate_keys"] == 0
    frame.loc[1, "bid_depth"] = 4
    _, counts, _ = row_checks(frame)
    assert counts["conflicting_duplicate_keys"] == 1
    frame.timestamp = ["2026-11-04T16:15:02Z", "2026-11-04T16:15:00Z"]
    parsed, _, _ = row_checks(frame)
    assert timing_counts(parsed)["backward_market_timestamps"] == 1


def test_empty_recording_is_not_a_malformed_file():
    frame, counts, _ = row_checks(example().iloc[:0])
    assert counts["invalid_rows"] == 0
    assert timing_counts(frame)["market_intervals_over_1_5s"] == 0


def test_sample_offsets_do_not_double_count_small_files(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_bytes(b''.join(json.dumps({"i": i}).encode() + b'\n' for i in range(12)))
    sample = list(sampled_lines(path, head_limit=8, tail_limit=8))
    assert len(sample) == len(set(row[1] for row in sample)) == 12
    assert [json.loads(line)["i"] for _, _, line in sample] == list(range(12))


def test_raw_delta_validation_catches_nonfinite_and_invalid_side():
    message = {"type": "orderbook_delta", "sid": 1, "seq": 1,
        "msg": {"market_ticker": "t", "price_dollars": ".4", "delta_fp": "2", "side": "yes"}}
    check_raw_message(message)
    message["msg"]["side"] = "unknown"
    with pytest.raises(ValueError):
        check_raw_message(message)
    message["msg"].update(side="yes", delta_fp="NaN")
    with pytest.raises(ValueError):
        check_raw_message(message)


def test_equal_raw_and_feature_timestamps_are_order_ambiguous(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    frame = example()
    # This matches the state BEFORE the simultaneous NO-side delta.
    frame.to_csv(tmp_path / "data/processed/features_test.csv", index=False)
    timestamp = pd.Timestamp(frame.timestamp.iloc[0])
    records = [(timestamp - pd.Timedelta(seconds=1), {
        "type": "orderbook_snapshot", "sid": 1, "seq": 1,
        "msg": {"market_ticker": "t", "yes_dollars_fp": [["0.40", "3"]],
                "no_dollars_fp": [["0.58", "1"]]}}),
        (timestamp, {"type": "orderbook_delta", "sid": 1, "seq": 2,
        "msg": {"market_ticker": "t", "side": "no", "price_dollars": "0.58", "delta_fp": "1"}})]
    info = [{"file": "features_test.csv", "rows": 1, "first_timestamp": str(timestamp),
             "last_timestamp": str(timestamp)}]
    counts, discrepancies = replay_comparison(records, tmp_path, info)
    assert counts["ambiguous_same_timestamp_samples"] == 1
    assert counts.get("mismatched_samples", 0) == 0
    assert discrepancies == []
