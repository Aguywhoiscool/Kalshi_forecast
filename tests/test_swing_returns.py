import numpy as np
import pandas as pd

from src.swing_return_experiment import evaluate, expected_entry, return_bounds


def fixture(n=1):
    return np.tile([.4, .42, 2., 2.], (n, 61, 1)).astype(float), np.ones((n, 61), bool)


def test_known_returns_are_net_and_complementary():
    future, known = fixture()
    future[0, 30, :2] = [.52, .54]
    lower, upper, status = return_bounds(future, known, 30)
    np.testing.assert_allclose(lower, [[6, -18]])
    np.testing.assert_array_equal(lower, upper)
    assert (status == 0).all()


def test_missing_future_is_bounded_not_dropped_or_a_claimed_fill():
    future, known = fixture()
    known[0, 30] = False
    future[0, 30] = np.nan
    lower, upper, status = return_bounds(future, known, 30)
    np.testing.assert_allclose(lower, [[-46, -64]])
    np.testing.assert_allclose(upper, [[54, 36]])
    assert (status == 3).all()
    known[0, 1] = False
    future[0, 1] = np.nan
    lower, upper, status = return_bounds(future, known, 30)
    assert (lower == -104).all() and (upper == 96).all() and (status == 2).all()


def test_known_capacity_rejection_costs_zero():
    future, known = fixture()
    future[0, 1, 3] = .5
    lower, upper, status = return_bounds(future, known, 60)
    assert lower[0, 0] == upper[0, 0] == 0
    assert status[0, 0] == 1


def test_high_win_probability_cannot_override_negative_expected_return():
    history = np.ones((1, 60, 7))
    expected_net = .9*5 + .1*(-90) - 4
    _, _, selected = expected_entry([[expected_net, -2]], history)
    assert not selected[0]
    # Positive expected value can be selected even if P(win) is below 50%.
    expected_net = .2*100 + .8*(-2) - 4
    _, _, selected = expected_entry([[expected_net, -2]], history)
    assert selected[0]


def test_missing_exit_keeps_first_signal_and_blocks_later_uncertain_position():
    history = np.ones((2, 60, 7))
    meta = pd.DataFrame({"session_id": ["s", "s"], "ticker": ["T", "T"],
                         "input_end": [0, 120*10**9]})
    future, known = fixture(2)
    known[0, 30] = False
    future[0, 30] = np.nan
    trades, result = evaluate("test", 30, np.array([[5., -2.], [5., -2.]]), history,
                              meta, future, known, 1)
    assert len(trades) == 2 and result["attempts"] == 1 and result["blocked"] == 1
    assert result["unresolved_exits"] == 1
    assert trades.status.tolist() == ["missing_exit_quote", "blocked_uncertain_or_pending"]
    assert result["net_lower_total_cents"] == -46
    np.testing.assert_allclose(result["net_upper_total_cents"], 54)
