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


def make_dashboard(tmp_path, predictor_factory=None):
    data = tmp_path / "data"
    data.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "protected_v2_period.json").write_text(json.dumps({"start": 0, "end": 1, "sessions": []}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol.json").write_text(json.dumps({"source": str(source)}))
    return Dashboard(run, data, predictor_factory or (lambda run: object()))


def add_forecast(dashboard, forecast_id="f1"):
    history = np.tile([.4, .42, .02, .41, 2., 2., 0.], (60, 1))
    dashboard.predictor = object()
    dashboard.forecasts[forecast_id] = {"ticker": "T", "session_id": "S", "input_end": 123,
        "history": history, "predictions": []}
    return history


def fake_explanation(calls, fail_gru=True):
    def explain(predictor, item, model, horizon, run, include_all=False, tree_index=0):
        calls.append((model, horizon))
        if fail_gru and model == "gru":
            raise AssertionError("prediction/explanation parity failed")
        details = {"kind": model}
        if model == "monte_carlo":
            details = {"kind": "monte_carlo", "seed": "1", "path_count": 2048,
                "preview_indices": list(range(48)), "preview_paths": [[i/100]*(horizon+1) for i in range(48)],
                "preview_returns": [[i, -i] for i in range(48)], "preview_draw_indices": [[0]*horizon]*48,
                "historical_increments": [0.]*59, "histogram": {"counts": [1], "edges": [0, 1]},
                "convergence_yes": [float(i) for i in range(2048)], "scenario_percentiles": [[0, 0]]*3,
                "limitations": "Bootstrap scenarios"}
        return {"model": model, "horizon": horizon, "details": details, "training": {"available": False},
            "calculation": {"final": [1., 2.], "signal": "WAIT"}}
    return explain


def route(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


def test_windows_returns_compacted_explanations_with_isolated_errors(tmp_path, monkeypatch):
    import src.prediction_explanations as explanations
    from src.live_dashboard import create_app
    calls = []
    monkeypatch.setattr(explanations, "explanation", fake_explanation(calls))
    dashboard = make_dashboard(tmp_path)
    history = add_forecast(dashboard)
    windows = route(create_app(run=dashboard.run, dashboard=dashboard), "/api/windows/{forecast_id}")
    payload = json.loads(windows("f1", "60").body)
    assert set(payload["models"]) == {"mean", "ridge", "sgd", "decision_tree", "random_forest",
                                      "gradient_boosting", "monte_carlo"}
    assert payload["errors"] == {"gru": "AssertionError: prediction/explanation parity failed"}
    assert payload["input"]["features"] == history.tolist()
    assert len(payload["input"]["summary_values"]) == len(payload["input"]["summary_names"]) == 50
    assert payload["models"]["ridge"] == {"calculation": {"final": [1., 2.], "signal": "WAIT"},
                                          "details": {"kind": "ridge"}}
    mc = payload["models"]["monte_carlo"]["details"]
    assert "historical_increments" not in mc and "preview_draw_indices" not in mc
    assert len(mc["preview_paths"]) == len(mc["preview_indices"]) == len(mc["preview_returns"]) == 32
    assert mc["preview_indices"][0] == 0 and mc["preview_indices"][-1] == 47
    assert [p[0] for p in mc["preview_paths"]] == [i/100 for i in mc["preview_indices"]]
    assert len(mc["preview_paths"][0]) == 61
    assert len(mc["convergence_yes"]) == len(mc["convergence_paths"]) == 128
    assert mc["convergence_paths"][-1] == 2048 and mc["convergence_yes"][-1] == 2047.
    assert mc["histogram"] == {"counts": [1], "edges": [0, 1]} and mc["path_count"] == 2048
    # Payloads with errors are not cached: a transient failure is retried.
    assert len(calls) == 8
    assert json.loads(windows("f1", "60").body) == payload
    assert len(calls) == 16


def test_cached_windows_still_honor_expiry_and_artifact_changes(tmp_path, monkeypatch):
    import pytest
    from fastapi import HTTPException
    import src.prediction_explanations as explanations
    from src.live_dashboard import create_app
    calls = []
    monkeypatch.setattr(explanations, "explanation", fake_explanation(calls, fail_gru=False))
    dashboard = make_dashboard(tmp_path)
    add_forecast(dashboard)
    windows = route(create_app(run=dashboard.run, dashboard=dashboard), "/api/windows/{forecast_id}")
    payload = json.loads(windows("f1", "30").body)
    assert payload["errors"] == {} and len(payload["models"]) == 8
    assert json.loads(windows("f1", "30").body) == payload
    assert len(calls) == 8  # Served from cache.
    (dashboard.run / "metrics.json").write_text("{}")  # Artifacts change after caching.
    with pytest.raises(HTTPException) as error:
        windows("f1", "30")
    assert error.value.status_code == 400
    (dashboard.run / "metrics.json").unlink()
    assert json.loads(windows("f1", "30").body) == payload
    dashboard.forecasts.clear()  # Evicted from the 240-forecast store.
    with pytest.raises(HTTPException) as error:
        windows("f1", "30")
    assert error.value.status_code == 410


def test_windows_rejects_bad_horizon_and_expired_forecast(tmp_path, monkeypatch):
    import pytest
    from fastapi import HTTPException
    import src.prediction_explanations as explanations
    from src.live_dashboard import create_app
    monkeypatch.setattr(explanations, "explanation", fake_explanation([]))
    dashboard = make_dashboard(tmp_path)
    add_forecast(dashboard)
    windows = route(create_app(run=dashboard.run, dashboard=dashboard), "/api/windows/{forecast_id}")
    for horizon in ("45", "abc"):
        with pytest.raises(HTTPException) as error:
            windows("f1", horizon)
        assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        windows("missing", "30")
    assert error.value.status_code == 410


def test_windows_cache_is_bounded(tmp_path, monkeypatch):
    import src.prediction_explanations as explanations
    monkeypatch.setattr(explanations, "explanation", fake_explanation([], fail_gru=False))
    dashboard = make_dashboard(tmp_path)
    for index in range(40):
        add_forecast(dashboard, f"f{index}")
        dashboard.windows(f"f{index}", 30)
    assert len(dashboard.windows_cache) == 32
    assert ("f0", 30) not in dashboard.windows_cache and ("f39", 30) in dashboard.windows_cache


def test_windows_cache_evicts_least_recently_used(tmp_path, monkeypatch):
    import src.prediction_explanations as explanations
    monkeypatch.setattr(explanations, "explanation", fake_explanation([], fail_gru=False))
    dashboard = make_dashboard(tmp_path)
    for index in range(32):
        add_forecast(dashboard, f"f{index}")
        dashboard.windows(f"f{index}", 30)
    dashboard.windows("f0", 30)  # Cache hit refreshes recency.
    add_forecast(dashboard, "f32")
    dashboard.windows("f32", 30)
    assert ("f0", 30) in dashboard.windows_cache
    assert ("f1", 30) not in dashboard.windows_cache
    assert len(dashboard.windows_cache) == 32


def test_scoreboard_endpoint_caches_success_and_reports_unavailable(tmp_path, monkeypatch):
    import pytest
    from fastapi import HTTPException
    import src.live_dashboard as live
    dashboard = make_dashboard(tmp_path)
    scoreboard = route(live.create_app(run=dashboard.run, dashboard=dashboard), "/api/scoreboard")
    def missing(run):
        raise FileNotFoundError("targets_30.npz")
    monkeypatch.setattr(live, "build_scoreboard", missing)
    with pytest.raises(HTTPException) as error:
        scoreboard()
    assert error.value.status_code == 503 and "targets_30.npz" in error.value.detail
    calls = []
    def build(run):
        calls.append(run)
        return {"run": run.name, "horizons": {}}
    monkeypatch.setattr(live, "build_scoreboard", build)
    expected = {"run": "run", "horizons": {}, "run_id": dashboard.run_id}
    assert json.loads(scoreboard().body) == expected
    assert json.loads(scoreboard().body) == expected
    assert len(calls) == 1
    (dashboard.run / "metrics.json").write_text("{}")  # Artifacts changed: refuse instead of serving stale numbers.
    with pytest.raises(HTTPException) as error:
        scoreboard()
    assert error.value.status_code == 503 and "changed" in error.value.detail


def test_update_predicts_and_stores_c_ordered_history(tmp_path):
    # summarize() reductions differ at ~1e-16 between memory layouts, enough to flip tree splits.
    seen = []
    class Fake:
        def __init__(self, run):
            pass
        def predict(self, item):
            seen.append(item["history"].flags["C_CONTIGUOUS"])
            return [{"model": "mean", "label": "Mean", "horizon": 30, "yes": -1., "no": -1., "signal": "WAIT"}]
    dashboard = make_dashboard(tmp_path, Fake)
    make_feed(dashboard.directory)
    dashboard.update()
    assert seen == [True]
    stored = next(iter(dashboard.forecasts.values()))
    assert stored["history"].flags["C_CONTIGUOUS"]
