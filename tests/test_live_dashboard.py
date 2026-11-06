import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.live_dashboard import Dashboard, ROOT, corrected, latest_histories, read_tail
from src.model_benchmark import summarize


def make_feed(tmp_path, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    session = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    times = pd.date_range(end=now, periods=60, freq="s")
    frame = pd.DataFrame({"timestamp": times.astype(str), "session_id": session, "ticker": "T",
        "bid": .4, "ask": .42, "spread": .02, "midpoint": .41,
        "bid_depth": 2., "ask_depth": 2., "imbalance": 0.})
    path = tmp_path / f"features_{session}.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_partial_csv_row_is_not_used_and_gaps_are_not_stitched(tmp_path):
    path, frame = make_feed(tmp_path)
    with path.open("a") as stream:
        stream.write("2026-11-06T12:00")
    assert len(read_tail(path)) == 60
    protected = {"start": 0, "end": 1, "sessions": []}
    items, _ = latest_histories(tmp_path, protected)
    assert len(items) == 1
    frame = frame.drop(index=30)
    frame.to_csv(path, index=False)
    assert latest_histories(tmp_path, protected)[0] == []


def test_reserved_history_is_never_exposed(tmp_path):
    _, frame = make_feed(tmp_path)
    protected = {"start": 0, "end": 1, "sessions": [frame.session_id.iloc[0]]}
    assert latest_histories(tmp_path, protected)[0] == []
    protected["sessions"] = []
    t = pd.Timestamp(frame.timestamp.iloc[-1]).value
    protected["additional_periods"] = [{"start": t-10**9, "end": t+10**9}]
    assert latest_histories(tmp_path, protected)[0] == []


def test_saved_tabular_predictions_match_live_correction():
    run = ROOT / "artifacts/swing_returns_all_20261106"
    if not run.exists():
        import pytest
        pytest.skip("Local model artifact unavailable")
    x = summarize(np.load(run / "history.npy", mmap_mode="r")[:2])
    metrics = json.loads((run / "metrics.json").read_text())
    for horizon in (30, 60):
        for name in ("mean", "ridge", "sgd", "decision_tree", "random_forest", "gradient_boosting"):
            model = joblib.load(run / "models" / f"{name}_{horizon}.joblib")
            raw = model.predict(x)
            correction = np.array(metrics[f"{name}_{horizon}"]["calibration_correction_cents"])
            np.testing.assert_allclose(corrected(raw, correction), np.clip(np.clip(raw, -104, 96)+correction, -104, 96))


def test_stale_feed_does_not_generate_live_predictions(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_feed(data, pd.Timestamp.now(tz="UTC")-pd.Timedelta(hours=1))
    source = tmp_path / "source"
    source.mkdir()
    (source / "protected_v2_period.json").write_text(json.dumps({"start": 0, "end": 1, "sessions": []}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol.json").write_text(json.dumps({"source": str(source)}))
    class Fake:
        def __init__(self, run):
            pass
        def predict(self, item):
            raise AssertionError("Stale data must not be inferred as live")
    dashboard = Dashboard(run, data, Fake)
    dashboard.update()
    snapshot = dashboard.snapshot()
    assert snapshot["status"] == "stale"
    assert snapshot["markets"][0]["predictions"] == []
    assert snapshot["timeline"] == []
