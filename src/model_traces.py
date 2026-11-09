"""Reproducible calculations shared by forecasting and explanation views."""
import hashlib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TracedSGD(RegressorMixin, BaseEstimator):
    """Versioned mini-batch SGD with actual parameter/gradient observations."""
    def __init__(self, epochs=20, learning_rate=.001, alpha=.001, batch_size=256, seed=42):
        self.epochs, self.learning_rate, self.alpha = epochs, learning_rate, alpha
        self.batch_size, self.seed = batch_size, seed

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.coef_, self.intercept_ = np.zeros((2, x.shape[1])), np.zeros(2)
        self.n_features_in_ = x.shape[1]
        self.trace_ = {"algorithm": "mini_batch_sgd_v1", "optimizer": "SGD", "projection": [0, 1],
            "projection_note": "Actual selected coefficients; a projection of the higher-dimensional training trajectory", "steps": []}
        rng, step = np.random.default_rng(self.seed), 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), self.batch_size):
                ids = order[start:start+self.batch_size]
                residual = self.predict(x[ids])-y[ids]
                gradient = residual.T @ x[ids]/len(ids)+self.alpha*self.coef_
                bias_gradient = residual.mean(axis=0)
                rate = self.learning_rate/(1+step*.001)**.5
                before = self.coef_[:, :2].copy()
                self.coef_ -= rate*gradient
                self.intercept_ -= rate*bias_gradient
                self.trace_["steps"].append({"step": step, "epoch": epoch+1,
                    "loss": float(np.sum(residual**2)/(2*len(ids))+self.alpha*np.sum((self.coef_+rate*gradient)**2)/2),
                    "learning_rate": rate, "before": before.tolist(), "after": self.coef_[:, :2].tolist(),
                    "gradient": gradient[:, :2].tolist(), "gradient_norm": float(np.linalg.norm(gradient))})
                step += 1
        self.trace_["final_coefficients"] = self.coef_.tolist()
        self.trace_["final_intercepts"] = self.intercept_.tolist()
        return self

    def predict(self, x):
        return np.asarray(x) @ self.coef_.T+self.intercept_


def simulate_mc(history, identity, paths=2048, trace=False, context=None):
    history = np.asarray(history, dtype=float)
    if history.shape != (60, 7) or not np.isfinite(history).all() or paths < 1:
        raise ValueError("Expected finite 60x7 history and positive path count")
    spread = history[-1, 1]-history[-1, 0]
    if context is None:
        increment_history = history
        method_version = "bootstrap60_v1"
    else:
        context = np.asarray(context, dtype=float)
        if context.shape != (180,7) or not np.isfinite(context).all():
            raise ValueError("Long Monte Carlo context requires finite 180x7 observed history")
        if not np.array_equal(context[-60:],history):
            raise ValueError("Long Monte Carlo context must end in the exact baseline history")
        from src.causal_features import _history
        _history(context[None],180)
        increment_history = context
        method_version = "bootstrap180_v1"
    increments = np.diff(increment_history[:, 3])
    seed = int.from_bytes(hashlib.sha256(
        f"return42|{identity['session_id']}|{identity['ticker']}|{identity['input_end']}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    mid = np.full(paths, history[-1, 3])
    predictions, per_path = {}, {}
    trajectory = np.empty((paths, 61)) if trace else None
    draws = np.empty((paths, 60), dtype=int) if trace else None
    if trace:
        trajectory[:, 0] = mid
    for step in range(1, 61):
        indices = rng.choice(len(increments), size=paths)
        mid = np.clip(mid+increments[indices], spread/2, 1-spread/2)
        if trace:
            trajectory[:, step], draws[:, step-1] = mid, indices
        if step == 1:
            entry_yes, entry_no = mid+spread/2, 1-(mid-spread/2)
        if step in (30, 60):
            yes = ((mid-spread/2)-entry_yes)*100-4
            no = ((1-mid-spread/2)-entry_no)*100-4
            predictions[step] = [float(yes.mean()), float(no.mean())]
            if trace:
                per_path[step] = np.column_stack([yes, no]).tolist()
    result = {"predictions": predictions}
    if trace:
        result.update(seed=str(seed), sampled_indices=draws.tolist(), paths=trajectory.tolist(),
            historical_increments=increments.tolist(), per_path_returns=per_path,
            spread=float(spread), path_count=paths, cost_buffer_cents=4,
            method_version=method_version,context_observations=len(increment_history),
            context_validation_note="Caller validates continuous timestamps and session/ticker identity; kernel validates quote values and exact suffix" if context is not None else "Original 60-observation model",
            method="Bootstrap past midpoint increments; constant spread; bounded prices; no future inputs")
    return result
