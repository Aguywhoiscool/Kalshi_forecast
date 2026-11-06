import json

import numpy as np
import pandas as pd
import pytest

from src.model_benchmark import summarize
from src.verify_swing_inputs import verify_derived_inputs, verify_exit_times
from src.v3_research import FEATURES, digest


@pytest.fixture
def legacy_run(tmp_path):
    source, run = tmp_path / "source", tmp_path / "run"
    (source / "snapshot").mkdir(parents=True)
    run.mkdir()
    session = "2026-11-01_00-00-00_000000"
    times = pd.date_range("2026-11-01", periods=120, freq="s", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "session_id": session, "ticker": "TEST",
        "bid": .4, "ask": .42, "spread": .02, "midpoint": .41,
        "bid_depth": 3., "ask_depth": 1., "imbalance": .5})
    filename = f"features_{session}.csv"
    frame.to_csv(source / "snapshot" / filename, index=False)
    (source / "input_manifest.json").write_text(json.dumps({"files": [
        {"name": filename, "sha256": digest(source / "snapshot" / filename)}]}))
    ns = times.as_unit("ns").asi8
    pd.DataFrame({"session_id": [session], "ticker": ["TEST"], "input_start": [ns[0]],
        "input_end": [ns[59]], "target_end": [ns[119]]}).to_parquet(run / "sample_metadata.parquet", index=False)
    history = frame[FEATURES].to_numpy(dtype=float)[:60][None]
    np.save(run / "history.npy", history)
    np.save(run / "summary_features.npy", summarize(history))
    np.save(run / "future_quotes.npy", frame[["bid", "ask", "bid_depth", "ask_depth"]].to_numpy(dtype=float)[59:120][None])
    np.save(run / "future_times.npy", ns[59:120][None])
    (run / "protocol.json").write_text(json.dumps({"source": str(source),
        "sample_metadata_sha256": digest(run / "sample_metadata.parquet")}))
    return run


def test_source_reconstruction_accepts_original_arrays(legacy_run):
    assert verify_derived_inputs(legacy_run)["source_reconstructed_examples"] == 1


@pytest.mark.parametrize("name", ["history", "summary_features", "future_quotes", "future_times"])
def test_each_altered_derived_array_is_rejected(legacy_run, name):
    path = legacy_run / f"{name}.npy"
    array = np.load(path)
    array.flat[0] += 86400 * 10**9 if name == "future_times" else 1
    np.save(path, array)
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        verify_derived_inputs(legacy_run)


def test_saved_trade_clock_must_match_observation():
    times = np.arange(61, dtype=np.int64)[None] * 10**9
    trades = pd.DataFrame({"exit_index": [30], "exit_time_ns": [30 * 10**9]})
    verify_exit_times(trades, times, np.array([0]))
    trades.loc[0, "exit_time_ns"] += 86400 * 10**9
    with pytest.raises(ValueError, match="exit timestamps"):
        verify_exit_times(trades, times, np.array([0]))
