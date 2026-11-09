"""Partially identified return objectives; unknown outcomes are never filled."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def interval_loss_gradient(prediction, lower, upper, delta=5.):
    prediction, lower, upper = map(lambda v: np.asarray(v, float), (prediction, lower, upper))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be positive and finite")
    if prediction.shape != lower.shape or lower.shape != upper.shape or np.any(lower > upper) or not all(np.isfinite(v).all() for v in (prediction, lower, upper)):
        raise ValueError("Invalid prediction/interval shapes")
    residual = prediction-np.clip(prediction, lower, upper)
    absolute = np.abs(residual)
    loss = np.where(absolute <= delta, .5*residual**2, delta*(absolute-.5*delta))
    return loss, np.clip(residual, -delta, delta)


def correction(raw, lower, upper, objective):
    if objective not in ("conservative", "interval"):
        raise ValueError("Unknown return objective")
    raw = np.asarray(raw, float)
    target = lower if objective == "conservative" else np.clip(raw, lower, upper)
    return (target-raw).mean(axis=0)


def error_metrics(prediction, lower, upper, status):
    observed = np.isin(status, [0, 1])
    closed = status == 0
    errors = np.abs(prediction-lower)
    outside = np.abs(prediction-np.clip(prediction, lower, upper))
    return {"observed_mae": float(errors[observed].mean()) if observed.any() else None,
        "closed_mae": float(errors[closed].mean()) if closed.any() else None,
        "observed_sides": int(observed.sum()), "closed_sides": int(closed.sum()),
        "rejected_sides": int((status == 1).sum()), "total_sides": int(status.size),
        "observed_coverage": float(observed.mean()), "full_lower_bound_mae": float(errors.mean()),
        "interval_distance_mae": float(outside.mean()), "unknown_sides": int((status >= 2).sum())}


class AccuracySGD(RegressorMixin, BaseEstimator):
    def __init__(self, learning_rate=.001, alpha=.001, objective="conservative", epochs=20, seed=42, delta=5.):
        self.learning_rate, self.alpha, self.objective = learning_rate, alpha, objective
        self.epochs, self.seed, self.delta = epochs, seed, delta

    def fit(self, x, bounds):
        if self.objective not in ("conservative", "interval"):
            raise ValueError("Unknown return objective")
        x, bounds = np.asarray(x, float), np.asarray(bounds, float)
        lower, upper = bounds[:, :2], bounds[:, 2:]
        self.coef_, self.intercept_ = np.zeros((2, x.shape[1])), np.zeros(2)
        self.n_features_in_ = x.shape[1]
        rng, step = np.random.default_rng(self.seed), 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), 256):
                ids = order[start:start+256]
                prediction = self.predict(x[ids])
                residual = prediction-lower[ids] if self.objective == "conservative" else interval_loss_gradient(prediction, lower[ids], upper[ids], self.delta)[1]
                gradient = residual.T@x[ids]/len(ids)+self.alpha*self.coef_
                rate = self.learning_rate/(1+step*.001)**.5
                self.coef_ -= rate*gradient
                self.intercept_ -= rate*residual.mean(axis=0)
                step += 1
        if not np.isfinite(self.coef_).all():
            raise ValueError("Nonfinite SGD fit")
        return self

    def predict(self, x):
        return np.asarray(x)@self.coef_.T+self.intercept_


def event_bootstrap(candidate, baseline, lower, status, groups, repetitions=1000, seed=42):
    observed = np.isin(status, [0, 1])
    sums = {}
    for index, group in enumerate(groups):
        mask = observed[index]
        if not mask.any():
            continue
        a = np.abs(candidate[index]-lower[index])[mask].sum()
        b = np.abs(baseline[index]-lower[index])[mask].sum()
        row = sums.setdefault(str(group), np.zeros(3))
        row += [a, b, mask.sum()]
    values = np.array(list(sums.values()))
    if len(values) < 2:
        return {"groups": len(values), "observed_event_groups": len(values), "ci_available": False,
            "passes_five_percent": False, "passes_positive_ci": False, "passes_observed_group_screen": False,
            "passes_both": False, "repetitions": repetitions, "reason": "Fewer than two observed event-proxy groups"}
    rng = np.random.default_rng(seed)
    improvements = []
    for _ in range(repetitions):
        sample = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        improvements.append((sample[1]-sample[0])/sample[2])
    overall = values.sum(axis=0)
    relative = (overall[1]-overall[0])/overall[1] if overall[1] > 0 else 0.
    low, high = np.quantile(improvements, [.025, .975])
    return {"groups": len(values), "ci_available": True, "repetitions": repetitions,
        "relative_observed_mae_improvement": float(relative), "improvement_cents_95ci": [float(low), float(high)],
        "passes_five_percent": bool(relative >= .05), "passes_positive_ci": bool(low > 0),
        "observed_event_groups": len(values), "passes_observed_group_screen": bool(len(values) >= 20),
        "passes_both": bool(relative >= .05 and low > 0 and len(values) >= 20),
        "limitation": "Observed-side event-proxy bootstrap; unknown intervals and proxy/event dependence limit identification"}
