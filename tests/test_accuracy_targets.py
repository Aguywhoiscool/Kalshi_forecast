import numpy as np
import pytest
from src.accuracy_targets import interval_loss_gradient, correction, AccuracySGD, error_metrics, event_bootstrap


def test_interval_huber_gradient_matches_nonlinear_finite_differences():
    prediction = np.array([[-20., -2.], [3., 30.]])
    lower = np.array([[-4., -4.], [-4., -4.]])
    upper = np.array([[4., 4.], [4., 4.]])
    loss, gradient = interval_loss_gradient(prediction, lower, upper)
    numerical = np.zeros_like(prediction)
    for index in np.ndindex(prediction.shape):
        plus, minus = prediction.copy(), prediction.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical[index] = (interval_loss_gradient(plus, lower, upper)[0].sum()-interval_loss_gradient(minus, lower, upper)[0].sum())/2e-5
    np.testing.assert_allclose(gradient, numerical, atol=1e-8)
    np.testing.assert_array_equal(gradient[0, 1], 0)
    np.testing.assert_array_equal(gradient[1, 0], 0)
    assert loss[0, 0] > 0 and loss[1, 1] > 0


def test_unknown_intervals_do_not_become_zero_or_lower_point_labels():
    prediction = np.array([[0., 5.]])
    lower, upper = np.array([[-104., -60.]]), np.array([[96., 40.]])
    loss, gradient = interval_loss_gradient(prediction, lower, upper)
    np.testing.assert_array_equal(loss, 0)
    np.testing.assert_array_equal(gradient, 0)
    np.testing.assert_array_equal(correction(prediction, lower, upper, "interval"), [0, 0])
    assert (correction(prediction, lower, upper, "conservative") < 0).all()


def test_observed_metric_reports_unknown_coverage_without_dropping_candidates():
    pred = np.array([[2., 3.], [50., -50.]])
    lower = np.array([[0., 1.], [-104., -104.]])
    upper = np.array([[0., 1.], [96., 96.]])
    status = np.array([[0, 1], [2, 3]])
    result = error_metrics(pred, lower, upper, status)
    assert result["observed_mae"] == 2
    assert result["observed_sides"] == 2 and result["total_sides"] == 4
    assert result["closed_sides"] == 1 and result["rejected_sides"] == 1


def test_event_bootstrap_uses_group_resampling_and_reproducible_paired_improvement():
    lower, status = np.zeros((4, 2)), np.zeros((4, 2), int)
    groups = ["A", "A", "B", "C"]
    result = event_bootstrap(np.ones((4, 2)), np.ones((4, 2))*2, lower, status, groups)
    assert result["groups"] == 3 and not result["passes_both"]
    assert result["passes_five_percent"] and result["passes_positive_ci"]
    assert not result["passes_observed_group_screen"]
    assert result["improvement_cents_95ci"] == [1, 1]
    assert result["relative_observed_mae_improvement"] == .5


def test_invalid_objectives_and_huber_scale_rejected():
    with pytest.raises(ValueError):
        interval_loss_gradient([0.], [0.], [1.], delta=0)
    with pytest.raises(ValueError):
        correction(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2)), "typo")
    with pytest.raises(ValueError):
        AccuracySGD(objective="typo").fit(np.ones((2, 2)), np.zeros((2, 4)))


def test_actual_multioutput_sgd_reduction_does_not_turn_unknown_bounds_into_labels():
    x = np.eye(2)
    bounds = np.array([[10., -10., 10., -10.], [-100., -100., 100., 100.]])
    model = AccuracySGD(learning_rate=.1, alpha=0, objective="interval", epochs=1).fit(x, bounds)
    np.testing.assert_allclose(model.coef_, [[.25, 0.], [-.25, 0.]])
    np.testing.assert_allclose(model.intercept_, [.25, -.25])
