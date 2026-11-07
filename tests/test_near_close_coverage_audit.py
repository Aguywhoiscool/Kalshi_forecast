import json

import numpy as np
import pandas as pd
import pytest

from src.audit_near_close_coverage import classify, download, load_cache


def test_time_bins_have_exact_boundaries_and_missing_remains_unknown():
    assert classify([-1, 0, 599, 600, 1799, 1800, 3599, 3600, np.nan]).tolist() == [
        "after_close", "under_10m", "under_10m", "10_to_30m", "10_to_30m",
        "30_to_60m", "30_to_60m", "over_60m", "unknown"]


def test_cache_recovers_only_final_interrupted_record(tmp_path):
    path = tmp_path / "schedules.jsonl"
    good = json.dumps({"ticker": "T", "close_time": "2026-11-06T12:00:00Z"})+"\n"
    path.write_text(good+'{"ticker":')
    assert set(load_cache(path)) == {"T"}
    assert path.read_text() == good
    path.write_text(good+"broken\n"+good)
    with pytest.raises(ValueError, match="corruption"):
        load_cache(path)


def test_resume_uses_cached_schedules_but_retries_failures(tmp_path, monkeypatch):
    path = tmp_path / "schedules.jsonl"
    path.write_text(json.dumps({"ticker": "CACHED", "close_time": "2026-11-06T12:00:00Z"})+"\n"+
                    json.dumps({"ticker": "RETRY", "error": "Timeout"})+"\n")
    calls = []
    def fetch(session, ticker, base):
        calls.append(ticker)
        return {"ticker": ticker, "close_time": "2026-11-06T12:00:00Z", "asof_verified": False}
    monkeypatch.setattr("src.audit_near_close_coverage.fetch_schedule", fetch)
    result = download(["CACHED", "RETRY", "NEW"], tmp_path, None, lambda *a, **k: None,
                      workers=2, rate=1000000)
    assert set(calls) == {"RETRY", "NEW"}
    assert set(result) == {"CACHED", "RETRY", "NEW"}
    assert all(r.get("close_time") for r in result.values())
    with pytest.raises(ValueError, match="Offline audit requires"):
        download(["MISSING"], tmp_path, None, lambda *a, **k: None, offline=True)


def test_subgroup_report_preserves_uncertain_position_blocking(tmp_path):
    from src.audit_near_close_coverage import summarize_predictions
    run, output = tmp_path / "run", tmp_path / "output"
    (run / "trades").mkdir(parents=True)
    output.mkdir()
    meta = pd.DataFrame({"session_id": ["s", "s"], "ticker": ["T", "T"],
        "input_end": [100, 200], "split": ["val", "val"], "time_bin": ["10_to_30m", "under_10m"]})
    meta.to_parquet(output / "window_schedule_diagnostics.parquet", index=False)
    models = ("mean", "ridge", "sgd", "decision_tree", "random_forest", "gradient_boosting", "gru", "monte_carlo", "no_trade")
    for horizon in (30, 60):
        meta.to_parquet(run / f"validation_{horizon}.parquet", index=False)
        np.savez(run / f"targets_{horizon}.npz", lower=np.zeros((2, 2)))
        for model in models:
            if model != "no_trade":
                np.savez(run / f"{model}_{horizon}_predictions.npz", validation=np.tile([2., -1.], (2, 1)))
            trades = meta[["session_id", "ticker", "input_end"]].copy()
            trades["status"] = ["missing_exit_quote", "blocked_uncertain_or_pending"]
            trades["net_lower_cents"] = [-50., 0.]
            trades["net_upper_cents"] = [50., 0.]
            if model == "no_trade":
                trades = trades.iloc[:0]
            trades.to_parquet(run / "trades" / f"{model}_{horizon}_margin1.parquet", index=False)
    rows = summarize_predictions(run, output)
    selected = next(r for r in rows if r["model"] == "gru" and r["horizon"] == 30 and r["time_bin"] == "under_10m")
    assert selected["attempts"] == 0 and selected["net_lower_total_cents"] == 0
