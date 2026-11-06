import numpy as np
import pandas as pd
import pytest

from src.swing_strategy_experiment import (action_mask, entry_choice, fixed_labels,
    inner_split, simulate_monte_carlo, strategy_rows, trade_outcomes)


def quotes(n=1):
    values = np.tile([.4, .42, 10., 10.], (n, 61, 1))
    return values.astype(float)


def test_fixed_returns_use_delayed_entry_and_complementary_no_prices():
    values = quotes(2)
    values[:, 1, :2] = [.45, .47]
    values[0, 30, :2] = [.52, .54]
    values[1, 30, :2] = [.30, .32]
    outcome = trade_outcomes(values, [1, 2], 30)
    np.testing.assert_allclose(outcome["gross_quote_cents"], [5, 13])
    np.testing.assert_array_equal(fixed_labels(values, 30), [1, 2])


def test_bracket_exits_at_first_crossing_not_later_peak():
    values = quotes()
    values[0, 3, :2] = [.52, .54]
    values[0, 6, :2] = [.8, .82]
    result = trade_outcomes(values, [1], 30, rule="bracket")
    assert result["exit_index"][0] == 3
    assert result["gross_quote_cents"][0] == pytest.approx(10)
    assert trade_outcomes(values, [1], 30)["gross_quote_cents"][0] == pytest.approx(-2)


def test_stop_loss_overshoot_uses_observed_price_not_fictitious_stop_fill():
    values = quotes()
    values[0, 2, :2] = [.2, .22]
    result = trade_outcomes(values, [1], 60, rule="bracket")
    assert result["exit_index"][0] == 2
    assert result["gross_quote_cents"][0] == pytest.approx(-22)


def test_low_exit_depth_remains_unresolved_with_conservative_value():
    values = quotes()
    values[0, 30, :2] = [.9, .92]
    values[0, 30, 2] = .5
    result = trade_outcomes(values, [1], 30)
    assert result["unresolved_exit"][0]
    assert result["gross_quote_cents"][0] == pytest.approx(48)
    assert result["gross_conservative_cents"][0] == pytest.approx(-42)
    assert fixed_labels(values, 30)[0] == 0


def test_entry_decision_uses_current_capacity_and_skips_ties():
    hist = np.ones((3, 60, 7))
    probability = np.array([[.1, .8, .1], [.1, .8, .1], [.2, .4, .4]])
    hist[1, -1, 5] = .5
    side, score = entry_choice(probability, hist)
    assert side.tolist()[:2] == [1, 2]
    assert score.tolist() == [.8, .1, -1]
    assert action_mask(score, "p50", 0).tolist() == [True, False, False]
    assert not action_mask([.01] * 3, "calibration_top1pct", .01).any()


def test_calibration_has_disjoint_sessions_and_temporal_purge():
    meta = pd.DataFrame({"split": ["train"] * 10 + ["val"],
        "session_id": [str(i) for i in range(11)],
        "input_start": np.arange(11) * 300 * 10**9,
        "input_end": (np.arange(11) * 300 + 59) * 10**9,
        "target_end": (np.arange(11) * 300 + 119) * 10**9})
    fit, cal, val = inner_split(meta)
    assert set(meta.loc[fit, "session_id"]).isdisjoint(meta.loc[cal, "session_id"])
    assert meta.loc[cal, "input_start"].min() >= meta.loc[fit, "target_end"].max() + 70 * 10**9
    assert meta.loc[val, "split"].eq("val").all()


def test_no_trade_is_zero_and_unfilled_entry_does_not_pay_cost(tmp_path):
    (tmp_path / "trades").mkdir()
    meta = pd.DataFrame({"session_id": ["s"], "ticker": ["t"], "input_end": [59]})
    hist = np.ones((1, 60, 7))
    times = np.arange(61, dtype=np.int64)[None] * 10**9
    empty = strategy_rows("none", 30, np.array([1]), np.array([-1.]), 1., meta, hist, quotes(), times, tmp_path)
    assert all(row["trades"] == 0 and row["total_net_scenario_cents"] == 0 for row in empty)
    values = quotes()
    values[0, 1, 3] = .1
    result = strategy_rows("unfilled", 30, np.array([1]), np.array([.9]), .5,
                           meta, hist, values, times, tmp_path)
    delayed = [row for row in result if row["entry_delay_observations"] == 1]
    assert all(row["trades"] == 0 and row["unfilled_entry_attempts"] == 1 for row in delayed)


def test_monte_carlo_only_uses_past_and_preserves_probabilities():
    hist = np.zeros((1, 60, 7))
    hist[:, :, 0] = .4
    hist[:, :, 1] = .42
    hist[:, :, 3] = .41
    meta = pd.DataFrame({"session_id": ["s"], "ticker": ["t"], "input_end": [59]})
    result = simulate_monte_carlo(hist, meta, paths=100)
    for horizon in (30, 60):
        assert result[horizon].sum() == pytest.approx(1)
        assert result[horizon][0, 0] > .98


def test_inputs_cannot_choose_hindsight_direction_or_invalid_horizons():
    with pytest.raises(ValueError):
        trade_outcomes(quotes(), [0], 30)
    with pytest.raises(ValueError):
        trade_outcomes(quotes(), [1], 61)
