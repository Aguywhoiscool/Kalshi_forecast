import numpy as np
import pandas as pd
from src.timing_comparison import timing_history, summarize_schema, non_near_fit


def group():
    stamps = pd.date_range("2026-11-07T05:00:00Z", periods=60, freq="s")
    close = pd.Timestamp("2026-11-07T05:45:00Z")
    return pd.DataFrame({"timestamp": stamps, "scheduled_close_time": close,
        "close_time_observed_at": pd.Timestamp("2026-11-07T04:59:00Z"),
        "seconds_to_scheduled_close": (close-stamps).total_seconds()})


def test_timing_uses_each_observations_own_knowledge_and_countdown():
    data = group()
    result = timing_history(data, np.array([59]))
    np.testing.assert_allclose(result, [2700-np.arange(60)])
    data.loc[20, "close_time_observed_at"] = pd.Timestamp("2026-11-07T06:00:00Z")
    assert np.isnan(timing_history(data, np.array([59]))[0, 20])
    data = group()
    data.loc[10, "seconds_to_scheduled_close"] += 1
    assert np.isnan(timing_history(data, np.array([59]))[0, 10])


def test_old_missing_metadata_is_not_filled_retrospectively():
    assert np.isnan(timing_history(group().drop(columns="scheduled_close_time"), np.array([59]))).all()


def test_seven_feature_summary_is_preserved_and_eighth_only_appends_timing():
    from src.model_benchmark import summarize
    history = np.ones((2, 60, 7))
    extended = np.concatenate([history, np.tile(np.arange(60)[None, :, None], (2, 1, 1))], axis=2)
    result = summarize_schema(extended)
    np.testing.assert_array_equal(result[:, :50], summarize(history))
    np.testing.assert_allclose(result[:, 50], 59)
    assert result.shape == (2, 56)


def test_fit_filter_uses_exact_ten_and_sixty_minute_boundaries():
    fit = np.array([0, 1, 2, 3, 4])
    remaining = np.array([599., 600., 600.01, 3600., 3600.01, 1000., 1500.])
    np.testing.assert_array_equal(non_near_fit(fit, remaining), [0, 1, 4])
