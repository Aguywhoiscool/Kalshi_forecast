import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.accuracy_data import prepare, walkforward
from src.v3_research import digest, save_json


def setup(tmp_path):
    source, output = tmp_path / "source", tmp_path / "run"
    (source / "snapshot").mkdir(parents=True)
    output.mkdir()
    save_json(source / "input_manifest.json", {"files": []})
    save_json(source / "protected_v2_period.json", {"start": 1, "end": 2, "sessions": []})
    save_json(output / "segments.json", {"source": str(source.resolve()),
        "source_manifest_sha256": digest(source / "input_manifest.json")})
    return source, output


def unlock(output):
    save_json(output / "protocol.json", {"frozen": True})
    save_json(output / "choices.json", {"segments_sha256": digest(output / "segments.json"),
        "protocol_sha256": digest(output / "protocol.json")})
    (output / "choices.sha256").write_text(digest(output / "choices.json"))
    (output / "refits").mkdir()
    (output / "refits/model.json").write_text('{"weight":1}')
    save_json(output / "refit_manifest.json", {"refits/model.json": digest(output / "refits/model.json")})
    save_json(output / "final_started.json", {"choices_sha256": digest(output / "choices.json"),
        "refit_manifest_sha256": digest(output / "refit_manifest.json")})


def test_final_loader_refuses_quarantine_and_weight_tampering(tmp_path, monkeypatch):
    source, output = setup(tmp_path)
    monkeypatch.setattr("src.accuracy_data.pd.read_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Final CSV touched")))
    with pytest.raises(ValueError, match="quarantined"):
        prepare(source, output, "final")
    unlock(output)
    (output / "refits/model.json").write_text('{"weight":2}')
    with pytest.raises(ValueError, match="refit bytes"):
        prepare(source, output, "final")


def test_three_folds_are_session_disjoint_and_have_label_history_embargo():
    records = []
    for session in range(30):
        for signal in (200, 320, 440):
            timestamp = (session*1000+signal)*10**9
            records.append({"session_id": str(session), "input_start": timestamp-179*10**9,
                "input_end": timestamp, "target_end": timestamp+60*10**9})
    meta = pd.DataFrame(records)
    folds = walkforward(meta)
    assert len(folds) == 3
    for fold in folds:
        sets = [set(meta.iloc[fold[k]].session_id) for k in ("fit", "calibration", "validation")]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        assert meta.iloc[fold["calibration"]].input_start.min() >= meta.iloc[fold["fit"]].target_end.max()+70*10**9
        assert meta.iloc[fold["validation"]].input_start.min() >= meta.iloc[np.r_[fold["fit"],fold["calibration"]]].target_end.max()+70*10**9


def test_actual_final_cohort_gate_precedes_future_quote_alignment(tmp_path, monkeypatch):
    source, output = setup(tmp_path)
    stamps = pd.date_range("2026-11-09T05:00:00Z", periods=241, freq="s")
    close = stamps[0]+pd.Timedelta(hours=1)
    frame = pd.DataFrame({"timestamp": stamps, "session_id": "s", "ticker": "EVENT-YES", "bid": .4,
        "ask": .42, "spread": .02, "midpoint": .41, "bid_depth": 2., "ask_depth": 2., "imbalance": 0.,
        "scheduled_close_time": close, "close_time_observed_at": stamps[0]-pd.Timedelta(seconds=1),
        "seconds_to_scheduled_close": (close-stamps).total_seconds()})
    path = source / "snapshot/features_s.csv"
    frame.to_csv(path, index=False)
    save_json(source / "input_manifest.json", {"files": [{"name": path.name, "sha256": digest(path)}]})
    pd.DataFrame({"session_id": ["s"]}).to_parquet(output / "final_sessions.parquet", index=False)
    save_json(output / "segments.json", {"source": str(source.resolve()),
        "source_manifest_sha256": digest(source / "input_manifest.json"),
        "final_metadata_sha256": digest(output / "final_sessions.parquet"),
        "final_cutoff_ns": stamps[0].value, "final_end_ns": stamps[-1].value})
    unlock(output)
    monkeypatch.setattr("src.accuracy_data.align_future", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Final outcomes opened before20event gate")))
    assert prepare(source, output, "final") is None
    assert json.loads((output / "final_gate_unavailable.json").read_text())["candidate_event_proxies"] == 1
    assert not (output / "final_labels_opened.json").exists()
    assert not (output / "final").exists()


def development_fixture(tmp_path, mixed=False, missing_future=False):
    source, output = setup(tmp_path)
    stamps = pd.date_range("2026-11-09T05:00:00Z", periods=301, freq="s")
    close = stamps[0]+pd.Timedelta(hours=1)
    frame = pd.DataFrame({"timestamp": stamps, "session_id": "s", "ticker": "EVENT-YES", "bid": .4,
        "ask": .42, "spread": .02, "midpoint": .41, "bid_depth": 2., "ask_depth": 2., "imbalance": 0.,
        "scheduled_close_time": close, "close_time_observed_at": stamps[0]-pd.Timedelta(seconds=1),
        "seconds_to_scheduled_close": (close-stamps).total_seconds()})
    if mixed:
        frame.loc[100, "session_id"] = "other"
    if missing_future:
        frame = frame.drop(index=239)
    path = source / "snapshot/features_s.csv"
    frame.to_csv(path, index=False)
    save_json(source / "input_manifest.json", {"files": [{"name": path.name, "sha256": digest(path)}]})
    pd.DataFrame({"session_id": ["s"]}).to_parquet(output / "development_sessions.parquet", index=False)
    save_json(output / "segments.json", {"source": str(source.resolve()),
        "source_manifest_sha256": digest(source / "input_manifest.json"),
        "development_metadata_sha256": digest(output / "development_sessions.parquet"),
        "development_end_ns": (stamps[0]+pd.Timedelta(seconds=1000)).value})
    return source, output, stamps


def test_actual_materializer_rejects_mixed_sessions(tmp_path):
    source, output, _ = development_fixture(tmp_path, mixed=True)
    with pytest.raises(ValueError, match="Filename/session"):
        prepare(source, output, "development")


def test_future_gap_and_recording_end_never_remove_past_eligible_signal(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source, output, stamps = development_fixture(first)
    normal = prepare(source, output, "development")
    regular = pd.read_parquet(normal / "metadata.parquet")
    assert regular.input_end.tolist() == [stamps[179].value, stamps[299].value]
    assert not np.load(normal / "future_valid.npy")[-1, 60]
    source2, output2, _ = development_fixture(second, missing_future=True)
    missing = prepare(source2, output2, "development")
    reduced = pd.read_parquet(missing / "metadata.parquet")
    assert stamps[179].value in reduced.input_end.tolist()
    assert not np.load(missing / "future_valid.npy")[0, 60]
