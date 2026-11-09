import numpy as np
from src.model_traces import TracedSGD, simulate_mc


def history():
    values = np.tile([.4, .42, .02, .41, 2., 3., -.2], (60, 1))
    values[:, :2] += np.linspace(-.01, 0, 60)[:, None]
    values[:, 3] += np.linspace(-.01, 0, 60)
    return values


def test_mc_trace_reconstructs_real_prediction_and_draws():
    ident = {"session_id": "s", "ticker": "T", "input_end": 123}
    h = history()
    trace = simulate_mc(h, ident, paths=32, trace=True)
    plain = simulate_mc(h, ident, paths=32)
    assert trace["predictions"] == plain["predictions"]
    paths = np.asarray(trace["paths"])
    draws = np.asarray(trace["sampled_indices"])
    for step in range(60):
        expected = np.clip(paths[:, step]+np.asarray(trace["historical_increments"])[draws[:, step]], .01, .99)
        np.testing.assert_allclose(paths[:, step+1], expected)
    for horizon in (30, 60):
        np.testing.assert_allclose(np.mean(trace["per_path_returns"][horizon], axis=0), trace["predictions"][horizon])


def test_sgd_records_actual_updates_and_reduces_known_quadratic_loss():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 3))
    y = np.column_stack([x[:, 0]*2+1, x[:, 1]*-3])
    model = TracedSGD(epochs=100, learning_rate=.1, batch_size=256).fit(x, y)
    for step in model.trace_["steps"]:
        np.testing.assert_allclose(np.asarray(step["after"]), np.asarray(step["before"])-step["learning_rate"]*np.asarray(step["gradient"]))
    assert np.mean((model.predict(x)-y)**2) < .01
    np.testing.assert_allclose(model.predict(x), x@np.asarray(model.trace_["final_coefficients"]).T+model.trace_["final_intercepts"])


def test_logged_gradient_matches_finite_difference_of_logged_objective():
    x = np.array([[1., 2.], [-1., 3.], [2., -1.]])
    y = np.array([[2., -1.], [3., 4.], [-2., 1.]])
    model = TracedSGD(epochs=1, batch_size=3).fit(x, y)
    step = model.trace_["steps"][0]
    weights = np.zeros((2, 2))
    def objective(w):
        return np.sum((x@w.T-y)**2)/(2*len(x))+.001*np.sum(w**2)/2
    np.testing.assert_allclose(step["loss"], objective(weights))
    numerical = np.zeros_like(weights)
    for side in range(2):
        for feature in range(2):
            plus, minus = weights.copy(), weights.copy()
            plus[side, feature] += 1e-5
            minus[side, feature] -= 1e-5
            numerical[side, feature] = (objective(plus)-objective(minus))/2e-5
    np.testing.assert_allclose(step["gradient"], numerical, atol=1e-8)
