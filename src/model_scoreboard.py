"""Read-only relative leaderboard over saved validation forecasts.

Uses only artifacts already written by the model run: exploratory validation
rows, their conservative return targets and each model's corrected forecast.
Never reads reserved outcomes, writes files or places orders.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.swing_return_experiment import MODEL_NAMES


SCOREBOARD_MODELS = ("mean", "ridge", "sgd", "decision_tree", "random_forest",
                     "gradient_boosting", "gru", "monte_carlo")
KEYS = ["session_id", "ticker", "input_end"]


def validation_rows(run, horizon):
    """Indices into the run-wide sample arrays for this horizon's validation rows."""
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    idx = np.flatnonzero((meta["split"] == "val").to_numpy() & meta[f"eligible_{horizon}"].to_numpy(bool))
    saved = pd.read_parquet(run / f"validation_{horizon}.parquet")
    expected = meta.iloc[idx][KEYS].reset_index(drop=True)
    if len(saved) != len(expected) or not expected.equals(saved[KEYS].reset_index(drop=True)):
        raise ValueError(f"Validation rows for {horizon}s do not align with sample metadata")
    return idx, expected["session_id"].to_numpy()


def horizon_scoreboard(run, horizon, resamples, seed):
    idx, sessions = validation_rows(run, horizon)
    targets = np.load(run / f"targets_{horizon}.npz")
    lower, upper, status = (targets[k][idx] for k in ("lower", "upper", "status"))
    n = len(idx)
    rows = np.arange(n)
    codes, session_names = pd.factorize(sessions)
    count = len(session_names)
    per_model, session_sums, session_counts = {}, [], []
    for name in SCOREBOARD_MODELS:
        forecast = np.load(run / f"{name}_{horizon}_predictions.npz")["validation"]
        if forecast.shape != (n, 2):
            raise ValueError(f"{name} {horizon}s forecasts have shape {forecast.shape}, expected {(n, 2)}")
        side = np.asarray(forecast).argmax(axis=1)  # Ties choose YES (index 0).
        chosen_lower, chosen_upper = lower[rows, side], upper[rows, side]
        closed = status[rows, side] == 0
        if not closed.any():
            raise ValueError(f"{name} {horizon}s has no closed outcomes")
        per_model[name] = {"closed_mean_cents": float(chosen_lower[closed].mean()),
            "lower_mean_cents": float(chosen_lower.mean()), "upper_mean_cents": float(chosen_upper.mean()),
            "known_fraction": float(closed.mean()), "yes_fraction": float((side == 0).mean())}
        session_sums.append(np.bincount(codes, weights=np.where(closed, chosen_lower, 0.), minlength=count))
        session_counts.append(np.bincount(codes, weights=closed.astype(float), minlength=count))
    average = float(np.mean([v["closed_mean_cents"] for v in per_model.values()]))

    # Cluster bootstrap: draw sessions with replacement; weights[b, s] = times session s drawn.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, count, size=(resamples, count))
    weights = np.bincount((draws + np.arange(resamples)[:, None]*count).ravel(),
                          minlength=resamples*count).reshape(resamples, count).astype(float)
    sums, counts = weights @ np.stack(session_sums, axis=1), weights @ np.stack(session_counts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums/np.where(counts > 0, counts, 1), np.nan)
    relative = means - means.mean(axis=1, keepdims=True)
    intervals = np.nanpercentile(relative, [2.5, 97.5], axis=0)

    values = [average, *intervals.ravel(), *(v for item in per_model.values() for v in item.values())]
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite scoreboard value at {horizon}s; population too small for the session bootstrap")
    models = []
    for j, name in enumerate(SCOREBOARD_MODELS):
        lo, hi = float(intervals[0, j]), float(intervals[1, j])
        models.append({"model": name, "label": MODEL_NAMES[name], **per_model[name],
            "relative_cents": per_model[name]["closed_mean_cents"]-average,
            "relative_ci_cents": [lo, hi], "within_noise": bool(lo <= 0 <= hi)})
    models.sort(key=lambda item: (-item["relative_cents"], item["model"]))
    for rank, item in enumerate(models, 1):
        item["rank"] = rank
    return {"n": int(n), "sessions": int(count), "average_closed_mean_cents": average, "models": models}


def build_scoreboard(run: Path, resamples=1000, seed=20261109) -> dict:
    run = Path(run)
    horizons = {str(h): horizon_scoreboard(run, h, resamples, seed) for h in (30, 60)}
    notes = []
    if all(m["closed_mean_cents"] < 0 for h in horizons.values() for m in h["models"]):
        notes.append("Every model's closed-outcome average per hypothetical trade is negative: none shows a profit on known outcomes. "
                     "Unknown outcomes remain intervals, so this is not a complete-population result.")
    else:
        notes.append("Some closed-outcome averages are not negative; unknown outcomes and costs still prevent a profit claim.")
    notes += ["Relative = model's closed-outcome average minus the cross-model average; it ranks models against each other, not against zero.",
        "Intervals resample whole sessions; when an interval contains 0 the model is not distinguishable from the average.",
        "Closed means use only outcomes with a known entry and exit; the lower/upper means bound every row including unknown outcomes.",
        "Exploratory validation rows were reused during development, so these rankings may be optimistic."]
    return {"run": run.name,
        "population": "Exploratory validation split (repeatedly used in development; not a fresh test)",
        "policy": "Always take the side with the higher forecast; one contract; 4c cost buffer included",
        "bootstrap": {"resamples": int(resamples), "unit": "session_id", "seed": int(seed), "interval": [2.5, 97.5]},
        "horizons": horizons, "notes": notes}
