import numpy as np
import pandas as pd
import pytest

from src.model_scoreboard import SCOREBOARD_MODELS, build_scoreboard


YES_MODELS = ("mean", "ridge", "sgd", "decision_tree")


def make_run(tmp_path, forecasts=None):
    """Two validation sessions; hand-computable YES/NO outcomes at 30s."""
    run = tmp_path / "run"
    run.mkdir()
    meta = pd.DataFrame({"session_id": ["T", "A", "A", "A", "B"], "ticker": ["X", "X", "Y", "Z", "X"],
        "split": ["train", "val", "val", "val", "val"], "input_start": [0, 1, 2, 3, 4],
        "input_end": [10, 11, 12, 13, 14], "eligible_30": [True, True, True, False, True],
        "eligible_60": [True, True, True, True, True], "forecast_max_horizon": 60, "target_end": 99})
    meta.to_parquet(run / "sample_metadata.parquet", index=False)
    for horizon in (30, 60):
        rows = meta.loc[(meta.split == "val") & meta[f"eligible_{horizon}"]]
        rows.to_parquet(run / f"validation_{horizon}.parquet", index=False)
        # Row 2's NO side is an unknown entry: interval [-104, 96], excluded from closed means.
        lower = np.array([[50, 50], [2, -6], [-10, -104], [0, 0], [-4, -8]], float)
        upper = np.array([[50, 50], [2, -6], [-10, 96], [0, 0], [-4, -8]], float)
        status = np.array([[0, 0], [0, 0], [0, 2], [0, 0], [0, 0]], np.int8)
        np.savez_compressed(run / f"targets_{horizon}.npz", lower=lower, upper=upper, status=status)
        for name in SCOREBOARD_MODELS:
            if forecasts is not None:
                value = forecasts[name]
            elif name == "mean":
                value = [0., 0.]  # Tie chooses YES.
            else:
                value = [5., -5.] if name in YES_MODELS else [-5., 5.]
            np.savez_compressed(run / f"{name}_{horizon}_predictions.npz",
                validation=np.tile(value, (len(rows), 1)))
    return run


def by_model(board, horizon="30"):
    return {item["model"]: item for item in board["horizons"][horizon]["models"]}


def test_hand_computed_closed_interval_and_relative_values(tmp_path):
    board = build_scoreboard(make_run(tmp_path), resamples=200)
    h30 = board["horizons"]["30"]
    assert (h30["n"], h30["sessions"]) == (3, 2)
    assert board["horizons"]["60"]["n"] == 4
    assert h30["average_closed_mean_cents"] == pytest.approx(-5.5)
    models = by_model(board)
    for name in YES_MODELS:
        item = models[name]
        assert item["closed_mean_cents"] == pytest.approx(-4)  # (2 - 10 - 4) / 3
        assert (item["lower_mean_cents"], item["upper_mean_cents"]) == pytest.approx((-4, -4))
        assert item["known_fraction"] == 1 and item["yes_fraction"] == 1
        assert item["relative_cents"] == pytest.approx(1.5)
        # Resamples AA, AB, BB give relative 1, 1.5, 2.
        assert item["relative_ci_cents"] == pytest.approx([1, 2])
        assert item["within_noise"] is False
    for name in set(SCOREBOARD_MODELS) - set(YES_MODELS):
        item = models[name]
        assert item["closed_mean_cents"] == pytest.approx(-7)  # Unknown row excluded: (-6 - 8) / 2
        assert item["lower_mean_cents"] == pytest.approx(-118/3)  # Unknown row included as an interval.
        assert item["upper_mean_cents"] == pytest.approx(82/3)
        assert item["known_fraction"] == pytest.approx(2/3) and item["yes_fraction"] == 0
        assert item["relative_cents"] == pytest.approx(-1.5)
        assert item["relative_ci_cents"] == pytest.approx([-2, -1])
    ranked = h30["models"]
    assert [item["rank"] for item in ranked] == list(range(1, 9))
    assert [item["model"] for item in ranked[:4]] == sorted(YES_MODELS)
    assert board["notes"][0].startswith("Every model's closed-outcome average per hypothetical trade is negative")
    assert "not a complete-population result" in board["notes"][0]
    assert board["bootstrap"] == {"resamples": 200, "unit": "session_id", "seed": 20261109, "interval": [2.5, 97.5]}


def test_identical_models_are_within_noise(tmp_path):
    board = build_scoreboard(make_run(tmp_path, {name: [5., -5.] for name in SCOREBOARD_MODELS}), resamples=50)
    for item in board["horizons"]["30"]["models"]:
        assert item["relative_cents"] == pytest.approx(0)
        assert item["relative_ci_cents"] == pytest.approx([0, 0])
        assert item["within_noise"] is True


def test_validation_misalignment_fails_loudly(tmp_path):
    run = make_run(tmp_path)
    rows = pd.read_parquet(run / "validation_30.parquet")
    rows.iloc[::-1].to_parquet(run / "validation_30.parquet", index=False)
    with pytest.raises(ValueError, match="align"):
        build_scoreboard(run, resamples=10)


def test_forecast_length_mismatch_fails_loudly(tmp_path):
    run = make_run(tmp_path)
    np.savez_compressed(run / "ridge_30_predictions.npz", validation=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        build_scoreboard(run, resamples=10)


def test_bootstrap_is_deterministic_for_a_seed(tmp_path):
    run = make_run(tmp_path, {name: [i - 3.5, 0.] for i, name in enumerate(SCOREBOARD_MODELS)})
    assert build_scoreboard(run, resamples=100, seed=7) == build_scoreboard(run, resamples=100, seed=7)


def test_non_finite_values_raise_value_error(tmp_path):
    run = make_run(tmp_path)
    targets = dict(np.load(run / "targets_30.npz"))
    targets["lower"][1, 0] = np.nan  # Closed YES outcome becomes non-finite.
    np.savez_compressed(run / "targets_30.npz", **targets)
    with pytest.raises(ValueError, match="Non-finite"):
        build_scoreboard(run, resamples=10)
