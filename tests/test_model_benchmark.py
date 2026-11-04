import json

import numpy as np
import pandas as pd
import pytest

from src.model_benchmark import (FEATURE_NAMES, IDENTITY, classification_metrics,
    execution_rows, make_models, materialize, monte_carlo_forecast, select_samples, simulate_validation, summarize,
    validate_splits, verify_gru, write_outputs)
from src.movement_labels import LABEL_POLICY, meaningful_movement


def test_features_only_read_history_and_preserve_known_momentum(tmp_path):
    data = np.ones((80, 7), dtype=float)
    data[:, 3] = np.arange(80) / 1000

    class Cache:
        def get(self, shard):
            return data

    meta = pd.DataFrame({"shard": [0], "end": [59]})
    before = materialize(meta, Cache(), tmp_path, "before").copy()
    data[60:] = 123456  # A future change must not alter any input feature.
    after = materialize(meta, Cache(), tmp_path, "after")
    np.testing.assert_array_equal(before, after)
    assert before.shape == (1, 50)
    assert before[0, FEATURE_NAMES.index("midpoint_momentum_10s")] == pytest.approx(.01)
    assert before[0, FEATURE_NAMES.index("midpoint_volatility_59s")] == pytest.approx(0, abs=1e-12)


def test_feature_extraction_rejects_invalid_shape_and_nonfinite():
    with pytest.raises(ValueError):
        summarize(np.zeros((2, 70, 7)))
    data = np.zeros((1, 60, 7))
    data[0, 3, 1] = np.inf
    with pytest.raises(ValueError):
        summarize(data)


def test_train_selection_matches_stride_and_cap_without_using_holdout():
    frame = pd.DataFrame({"split": ["train"] * 12 + ["val", "holdout"],
        "session_id": ["s"] * 14, "ticker": ["t"] * 14,
        "input_start": np.arange(14) * 100, "target_end": np.arange(14) * 100 + 69,
        "input_end": np.arange(14) * 100 + 59})
    train, val = select_samples(frame, {"train_stride": 2, "max_train_windows": 3})
    assert train.input_start.tolist() == [0, 400, 1000]
    assert val.split.tolist() == ["val"]


def test_split_checks_detect_reserved_period_and_session_leakage():
    meta = pd.DataFrame({"split": ["train", "val", "holdout"],
        "session_id": ["a", "b", "c"], "input_start": np.array([0, 200, 400]) * 10**9,
        "target_end": np.array([69, 269, 469]) * 10**9})
    protected = {"start": 1000 * 10**9, "end": 1100 * 10**9, "sessions": []}
    validate_splits(meta, protected)
    with pytest.raises(ValueError, match="Reserved v2 period"):
        validate_splits(meta, {**protected, "start": 50 * 10**9})
    meta.loc[1, "session_id"] = "a"
    with pytest.raises(ValueError, match="Session crosses"):
        validate_splits(meta, protected)


def test_execution_charges_both_spreads_and_no_branch_uses_true_direction():
    quotes = np.array([[.4, .42, .45, .47], [.4, .42, .35, .37]])
    result = execution_rows("test", [.9, .1], [.9, .1], quotes)
    assert result[0]["signals"] == 1
    assert result[0]["mean_gross_cents"] == pytest.approx(3)
    assert result[-1]["mean_gross_cents"] == pytest.approx(3)
    flat = execution_rows("test", [.9], [.1], [[.4, .42, .4, .42]])
    assert flat[-1]["mean_gross_cents"] == pytest.approx(-2)


def test_class_imbalance_does_not_turn_prior_into_skill():
    metrics = classification_metrics(np.array([0] * 9 + [1]), np.full(10, .1))
    assert metrics["accuracy"] == .9
    assert metrics["average_precision"] == .1
    assert metrics["balanced_accuracy"] == .5
    assert metrics["roc_auc"] == .5
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0, 2])


def test_gru_reuse_rejects_reordered_validation_even_with_same_row_count(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"label_policy": LABEL_POLICY}))
    frame = pd.DataFrame({key: [1, 2] for key in IDENTITY})
    saved = frame.iloc[::-1].copy()
    saved["change_cents"] = [1., -1.]
    saved["p_movement"] = [.8, .8]
    saved["p_up"] = [.9, .1]
    saved.to_parquet(tmp_path / "validation_predictions.parquet", index=False)
    with pytest.raises(AssertionError):
        verify_gru(tmp_path, frame, np.array([1., -1.]), np.array([1., -1.]))


def test_threshold_roundoff_matches_exact_decimal_quote_changes():
    from decimal import Decimal
    pairs = [(.405, .41), (.10, .105), (.41, .405), (.105, .10)]
    floating = np.array([(end - start) * 100 for start, end in pairs])
    exact = np.array([float((Decimal(str(end)) - Decimal(str(start))) * 100)
                      for start, end in pairs])
    assert (np.abs(floating) < .5).any()
    np.testing.assert_array_equal(meaningful_movement(floating), np.abs(exact) >= .5)
    assert not meaningful_movement([.499999, -.499999, 0]).any()


def test_legacy_gru_cannot_be_reused_with_corrected_labels(tmp_path):
    (tmp_path / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="legacy labels"):
        verify_gru(tmp_path, None, None, None)


def test_tied_signal_scores_have_order_invariant_expected_pnl():
    quotes = np.array([[.4, .42, .50, .52], [.4, .42, .30, .32]])
    first = execution_rows("prior", [.2, .2], [.9, .9], quotes)
    reverse = execution_rows("prior", [.2, .2], [.9, .9], quotes[::-1])
    assert first == reverse
    assert first[0]["mean_gross_cents"] == pytest.approx(-2.)
    assert first[0]["cutoff_tie_size"] == 2


def test_execution_rejects_broadcasting_and_bad_quotes():
    with pytest.raises(ValueError, match="shapes"):
        execution_rows("test", [.2], [.4], [[.4, .42, .4, .42]] * 2)
    with pytest.raises(ValueError, match="quotes"):
        execution_rows("test", [.2], [.4], [[.5, .4, .4, .42]])


def test_models_do_not_randomly_split_time_series_for_early_stopping():
    models = make_models()
    assert models["gradient_boosting"].early_stopping is False
    assert models["sgd"].steps[-1][1].early_stopping is False
    assert models["random_forest"].n_jobs == 2


def test_monte_carlo_flat_history_and_no_moving_path_fallback():
    result = monte_carlo_forecast(np.full(60, .4), paths=100, seed=42)
    assert result["simulated_moving_paths"] == 0
    assert result["movement"] == pytest.approx(.5 / 101)
    assert result["direction"] == .5
    assert result["terminal_p05"] == result["terminal_p95"] == .4


def test_monte_carlo_known_trend_and_bounded_paths():
    result = monte_carlo_forecast(np.linspace(.3, .359, 60), paths=100)
    assert result["movement"] == pytest.approx(100.5 / 101)
    assert result["direction"] == pytest.approx(100.5 / 101)
    assert result["mean_terminal_midpoint"] == pytest.approx(.369)
    bounded = monte_carlo_forecast(np.linspace(.41, 1., 60), paths=100)
    assert bounded["terminal_p95"] == 1
    assert bounded["simulated_moving_paths"] == 0


def test_monte_carlo_reproducible_per_identity_and_ignores_future():
    data = np.zeros((80, 7))
    data[:60, 3] = .4 + np.sin(np.arange(60)) * .01

    class Cache:
        def get(self, shard):
            return data

    meta = pd.DataFrame({"session_id": ["s", "s"], "ticker": ["A", "B"],
                         "input_end": [59, 59], "shard": [0, 0], "end": [59, 59]})
    result = simulate_validation(meta, Cache())
    data[60:] = np.inf
    reordered = simulate_validation(meta.iloc[::-1], Cache()).iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(result, reordered)
    assert result.movement.between(0, 1).all()
    assert result.direction.between(0, 1).all()


def test_monte_carlo_rejects_future_sized_or_invalid_histories():
    with pytest.raises(ValueError):
        monte_carlo_forecast(np.full(70, .4))
    with pytest.raises(ValueError):
        monte_carlo_forecast(np.full(60, 1.1))
    with pytest.raises(ValueError):
        monte_carlo_forecast(np.full(60, .4), paths=0)


def test_report_exports_actual_metrics_and_escapes_embedded_html(tmp_path):
    metrics = {"gru": {task: {**classification_metrics([0, 1], [.1, .9]),
                "training_examples": 10, "fit_seconds": None}
                for task in ("movement", "direction")}}
    execution = execution_rows("gru", [.9], [.9], [[.4, .42, .4, .42]])
    protocol = {"training_examples": 10, "moving_training_examples": 4,
                "validation_examples": 2, "source": "</script><script>alert(1)</script>"}
    write_outputs(tmp_path, metrics, execution, protocol)
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "</script><script>alert(1)" not in page
    assert "const DATA=" in page
    assert "GRU neural network" in (tmp_path / "REPORT.md").read_text()
    assert json.loads((tmp_path / "metrics.json").read_text())["gru"]["movement"]["roc_auc"] == 1
