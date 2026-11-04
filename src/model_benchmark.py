"""Offline model comparison on the frozen v3 development split.

Run: python -m src.model_benchmark --source-run artifacts/MY_CORRECTED_RUN
Never imports a market client, reads holdout outcomes, or submits orders.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                            balanced_accuracy_score, brier_score_loss,
                            log_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.v3_research import FEATURES, ShardCache, digest, overlaps, save_json, spaced, targets
from src.movement_labels import LABEL_POLICY, meaningful_movement


IDENTITY = ["session_id", "ticker", "shard", "end", "input_start", "input_end", "target_end"]
MODEL_INFO = {
    "prior": ("Class-prior baseline", "Predicts the training class frequency; no market features."),
    "logistic": ("Logistic regression", "A linear probability model, fitted with L-BFGS."),
    "sgd": ("Logistic regression · SGD", "A linear model trained with stochastic gradient descent."),
    "decision_tree": ("Decision tree", "One interpretable tree of feature thresholds."),
    "random_forest": ("Random forest", "An ensemble of randomized decision trees."),
    "gradient_boosting": ("Gradient boosting", "Histogram-based trees that successively correct errors."),
    "gru": ("GRU neural network", "Existing two-layer recurrent model using the full 60 × 7 sequence."),
    "monte_carlo": ("Monte Carlo bootstrap", "Simulates 2,048 bounded 10-second paths from each observed history; no labels fitted."),
}
STAT_NAMES = ("last", "mean", "std", "min", "max", "change")
FEATURE_NAMES = [f"{name}_{stat}" for stat in STAT_NAMES for name in FEATURES] + [
    *(f"midpoint_momentum_{lag}s" for lag in (1, 5, 10, 30, 59)),
    *(f"midpoint_volatility_{lag}s" for lag in (10, 30, 59)),
]


def summarize(history):
    """50 causal features. Input ends now; no future quotes are accepted."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (60, 7):
        raise ValueError("Expected (N, 60, 7) input history")
    if not np.isfinite(history).all():
        raise ValueError("Non-finite input history")
    mid = history[:, :, 3]
    blocks = [history[:, -1], history.mean(axis=1), history.std(axis=1),
              history.min(axis=1), history.max(axis=1), history[:, -1] - history[:, 0]]
    blocks += [(mid[:, -1] - mid[:, -lag - 1])[:, None] for lag in (1, 5, 10, 30, 59)]
    blocks += [np.diff(mid[:, -lag - 1:], axis=1).std(axis=1)[:, None] for lag in (10, 30, 59)]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def select_samples(meta, config):
    if (config["train_stride"] < 1 or config["max_train_windows"] < 1
            or not 0 < config.get("train_fraction", 1.) <= 1):
        raise ValueError("Invalid training sample configuration")
    training = meta.loc[meta.split == "train"].reset_index(drop=True)
    training = training.iloc[::config["train_stride"]].reset_index(drop=True)
    if len(training) > config["max_train_windows"]:
        indices = np.linspace(0, len(training) - 1, config["max_train_windows"], dtype=int)
        training = training.iloc[indices].reset_index(drop=True)
    fraction = config.get("train_fraction", 1.)
    training = training.iloc[:max(1, int(len(training) * fraction))].reset_index(drop=True)
    validation = spaced(meta.loc[meta.split == "val"].reset_index(drop=True))
    if training.empty or validation.empty:
        raise ValueError("Empty training or validation split")
    return training, validation


def validate_splits(meta, protected):
    if not set(meta.split.unique()) <= {"train", "val", "holdout"}:
        raise ValueError("Unknown split")
    for name in ("train", "val", "holdout"):
        if not (meta.split == name).any():
            raise ValueError("Missing split: " + name)
    if meta.groupby("session_id").split.nunique().max() > 1:
        raise ValueError("Session crosses split boundary")
    if overlaps(meta.input_start.to_numpy(), meta.target_end.to_numpy(), protected).any():
        raise ValueError("Reserved v2 period overlap")
    if meta.session_id.astype(str).isin(protected["sessions"]).any():
        raise ValueError("Reserved v2 session overlap")
    for left, right in (("train", "val"), ("val", "holdout")):
        if meta.loc[meta.split == right, "input_start"].min() < (
            meta.loc[meta.split == left, "target_end"].max() + 70 * 1_000_000_000
        ):
            raise ValueError("Chronological embargo violated")


def materialize(meta, cache, output, label):
    matrix = np.lib.format.open_memmap(output / f"{label}_features.npy", mode="w+",
                                     dtype=np.float32, shape=(len(meta), len(FEATURE_NAMES)))
    for shard, positions in meta.groupby("shard", sort=False).indices.items():
        data = cache.get(shard)
        for start in range(0, len(positions), 1024):
            selected = positions[start:start + 1024]
            ends = meta.iloc[selected].end.to_numpy(dtype=np.int64)
            history = data[ends[:, None] + np.arange(-59, 1)]
            matrix[selected] = summarize(history)
    matrix.flush()
    return matrix


def make_models():
    return {
        "prior": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=1., max_iter=500, random_state=42)),
        "sgd": make_pipeline(StandardScaler(), SGDClassifier(
            loss="log_loss", alpha=.001, max_iter=100, tol=1e-4,
            early_stopping=False, average=True, random_state=42)),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=100, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=64, max_depth=12,
            min_samples_leaf=50, max_features="sqrt", n_jobs=2, random_state=42),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=100,
            max_leaf_nodes=15, learning_rate=.1, l2_regularization=1.,
            early_stopping=False, random_state=42),
    }


def classification_metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes required for evaluation")
    if probability.shape != y.shape or not np.isfinite(probability).all():
        raise ValueError("Invalid prediction shape or values")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probability out of bounds")
    predicted = probability >= .5
    return {"average_precision": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "brier_score": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "positive_rate": float(y.mean()), "validation_examples": len(y)}


def execution_rows(model, movement_probability, direction_probability, quotes):
    quotes = np.asarray(quotes, dtype=float)
    score = np.asarray(movement_probability, dtype=float)
    direction = np.asarray(direction_probability, dtype=float)
    if (quotes.ndim != 2 or quotes.shape[1] != 4 or len(quotes) == 0
            or score.shape != (len(quotes),) or direction.shape != score.shape):
        raise ValueError("Execution input lengths or shapes differ")
    if (not np.isfinite(quotes).all() or not np.isfinite(score).all()
            or not np.isfinite(direction).all() or ((direction < 0) | (direction > 1)).any()):
        raise ValueError("Non-finite quotes/scores or invalid direction probabilities")
    entry_bid, entry_ask, exit_bid, exit_ask = quotes.T
    if not ((entry_bid >= 0) & (entry_bid < entry_ask) & (entry_ask <= 1)
            & (exit_bid >= 0) & (exit_bid < exit_ask) & (exit_ask <= 1)).all():
        raise ValueError("Invalid execution quotes")
    up = direction >= .5
    pnl = 100 * np.where(up, exit_bid - entry_ask, entry_bid - exit_ask)
    order = np.argsort(-score, kind="stable")
    rows = []
    for fraction in (.01, .05, .1, .2, 1.):
        count = max(1, int(len(pnl) * fraction))
        cutoff = score[order[count - 1]]
        above, tied = score > cutoff, score == cutoff
        remaining = count - int(above.sum())
        # Average over all possible uniform choices at the score cutoff.
        # Fractional weights are expected selection counts, not executed trades.
        weights = above.astype(float) + tied * (remaining / int(tied.sum()))
        rows.append({"model": model, "fraction": fraction, "signals": count,
                     "mean_gross_cents": float(np.dot(weights, pnl) / count),
                     "sum_gross_cents": float(np.dot(weights, pnl)),
                     "win_rate": float(np.dot(weights, pnl > 1e-8) / count),
                     "cutoff_score": float(cutoff), "cutoff_tie_size": int(tied.sum()),
                     "tie_policy": "Uniform expected selection at tied cutoff; retrospective ranking"})
    return rows


def quote_frame(meta, cache):
    quotes = np.empty((len(meta), 4))
    for shard, positions in meta.groupby("shard", sort=False).indices.items():
        ends = meta.iloc[positions].end.to_numpy(dtype=np.int64)
        data = cache.get(shard)
        quotes[positions, :2] = data[ends, :2]
        quotes[positions, 2:] = data[ends + 10, :2]
    return quotes


def monte_carlo_forecast(midpoints, paths=2048, seed=42):
    """IID bootstrap of 59 past one-second changes, clipping each step to [0,1].

    This is a stochastic-process baseline, not a supervised learning model.
    Jeffreys smoothing prevents impossible certainty from finite simulation;
    conditional direction is 0.5 when no simulated meaningful move occurs.
    """
    midpoints = np.asarray(midpoints, dtype=np.float64)
    if midpoints.shape != (60,) or not np.isfinite(midpoints).all():
        raise ValueError("Monte Carlo requires exactly 60 finite historical midpoints")
    if ((midpoints < 0) | (midpoints > 1)).any() or paths < 1:
        raise ValueError("Invalid midpoint range or simulation count")
    rng = np.random.default_rng(seed)
    increments = np.diff(midpoints)
    simulated = np.full(paths, midpoints[-1], dtype=np.float64)
    for _ in range(10):
        simulated = np.clip(simulated + rng.choice(increments, size=paths), 0., 1.)
    change_cents = (simulated - midpoints[-1]) * 100
    moving = meaningful_movement(change_cents)
    count = int(moving.sum())
    return {"movement": (count + .5) / (paths + 1),
            "direction": (int((change_cents[moving] > 0).sum()) + .5) / (count + 1),
            "simulated_moving_paths": count,
            "mean_terminal_midpoint": float(simulated.mean()),
            "terminal_p05": float(np.quantile(simulated, .05)),
            "terminal_p95": float(np.quantile(simulated, .95))}


def simulate_validation(meta, cache, progress=None):
    rows = []
    for position, row in enumerate(meta.itertuples()):
        identity = f"42|{row.session_id}|{row.ticker}|{row.input_end}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
        history = cache.get(row.shard)[row.end - 59:row.end + 1, 3]
        rows.append(monte_carlo_forecast(history, seed=seed))
        if progress is not None and (position + 1) % 1000 == 0:
            progress("monte_carlo", windows=position + 1, total=len(meta))
    return pd.DataFrame(rows)


def verify_gru(source, validation, changes, training_changes):
    config = json.loads((source / "config.json").read_text())
    if config.get("label_policy") != LABEL_POLICY:
        raise ValueError("GRU checkpoint used legacy labels; retrain in a NEW run before benchmarking")
    saved = pd.read_parquet(source / "validation_predictions.parquet")
    pd.testing.assert_frame_equal(validation[IDENTITY], saved[IDENTITY], check_dtype=False)
    if not np.array_equal(changes, saved.change_cents.to_numpy()):
        raise ValueError("Saved GRU target alignment mismatch")
    moving_train = meaningful_movement(training_changes)
    expected = {"movement": len(training_changes), "direction": int(moving_train.sum())}
    original = json.loads((source / "metrics.json").read_text())
    for task, count in expected.items():
        if original[task]["training_windows"] != count:
            raise ValueError("GRU training population mismatch: " + task)
    return {"movement": saved.p_movement.to_numpy(), "direction": saved.p_up.to_numpy()}


def write_outputs(output, metrics, execution, protocol):
    save_json(output / "metrics.json", metrics)
    pd.DataFrame([{"model": model, "task": task, **values}
                  for model, tasks in metrics.items() for task, values in tasks.items()]
                 ).to_csv(output / "leaderboard.csv", index=False)
    pd.DataFrame(execution).to_csv(output / "execution.csv", index=False)
    payload = {"metrics": metrics, "execution": execution, "protocol": protocol,
               "models": {key: {"name": name, "description": description}
                          for key, (name, description) in MODEL_INFO.items()}}
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kalshi Model Lab</title><style>
body{font:16px system-ui;background:#0e1726;color:#e4eaf3;margin:0;padding:36px;max-width:1200px;margin:auto}
h1{font-size:36px;margin-bottom:8px}p{line-height:1.65;color:#b9c7db}label{display:inline-block;margin:12px 24px 18px 0}
select{padding:9px;background:#1b2b43;color:white;border:1px solid #456080;border-radius:6px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:14px 10px;text-align:right;border-bottom:1px solid #2c3a50}
th:first-child,td:first-child{text-align:left}th{color:#9bb9d9}small{display:block;color:#9fafc4;max-width:350px;margin-top:5px}
.notice{background:#192941;border-left:4px solid #71b8ff;padding:16px;border-radius:4px}.scroll{overflow-x:auto}
.negative{color:#ffb3a8}.positive{color:#8fddb1}button{cursor:pointer}details{margin-top:24px}a{color:#8fc8ff}
</style><h1>Kalshi Model Lab</h1><p>Compare how different learning methods forecast the same Kalshi order books.</p>
<p class="notice">Development validation only. Prediction skill does not establish profitability.
Execution is a hypothetical buy at ask / sell at bid; fees, slippage, latency and market impact are excluded.
Original v2 test and v3 holdout outcomes remain untouched.</p>
<label>Prediction task <select id="task"><option value="movement">Will price move at least 0.5¢?</option>
<option value="direction">Up or down, given a meaningful move?</option></select></label>
<label>Signals for gross simulation <select id="fraction"><option value="0.01">Top 1%</option><option value="0.05">Top 5%</option>
<option value="0.1" selected>Top 10%</option><option value="0.2">Top 20%</option><option value="1">All</option></select></label>
<p id="population"></p><div class="scroll"><table><thead><tr><th>Model</th><th>Average precision ↑</th>
<th>ROC-AUC</th><th>Accuracy</th><th>Balanced accuracy</th><th>Brier ↓</th><th>Fit time</th><th>Gross ¢ / contract</th>
</tr></thead><tbody id="results"></tbody></table></div>
<p>Ranking uses average precision (higher is better). Brier score measures probability error (lower is better).
The simulation always uses both movement and direction models, regardless of the selected task.
Class-prior accuracy can look high when most prices do not move; compare average precision and balanced accuracy.</p>
<p>Top fractions rank the entire validation period retrospectively, not a deployable online threshold.
Ties at the cutoff use the expected result of uniform selection, avoiding arbitrary chronological tie-breaking.</p>
<details open><summary>How to read this experiment</summary><p id="method"></p>
<p>The GRU reads the full 60-observation sequence. Other methods read 50 fixed summaries of the same history;
this compares practical pipelines, not architecture alone. Scaling is fitted on training data only.
The GRU checkpoint was selected using this validation period; other methods use fixed settings.
These results are exploratory and need replication on future dates. No hyperparameter search or significance claim.</p>
<p>Direction is scored only on examples with actual movement ≥0.5¢, but execution selects signals using predictions alone.
Validation windows do not overlap within a market/session; related markets may remain correlated.</p>
<p>Gradient descent is an optimizer: the SGD row is logistic regression trained with stochastic gradient descent.
Gradient boosting is a separate tree-ensemble method. No model submits trades.</p></details>
<details open><summary>Monte Carlo assumptions</summary><p>Monte Carlo is a stochastic baseline, not a supervised learner.
For each validation history it independently resamples the 59 observed one-second midpoint changes to simulate
2,048 paths of 10 steps, clipping prices to [0,1] at every step. It estimates movement probability and direction
conditional on a meaningful simulated move. A fixed seed per example makes results reproducible; Jeffreys smoothing
avoids probabilities of exactly zero or one. It fits no training labels and assumes recent increments are
representative and independent. It does not model queues, trades or confirmed fills. Simulation ranges are
model-based ranges, not validated confidence intervals for performance.</p></details>
<details><summary>Reproducibility and exports</summary><p><a href="leaderboard.csv">Prediction CSV</a> ·
<a href="execution.csv">Execution CSV</a> · <a href="protocol.json">Protocol</a> · <a href="metrics.json">Metrics JSON</a></p>
<pre id="protocol" style="white-space:pre-wrap"></pre></details>
<script>const DATA=__DATA__;
const el=id=>document.getElementById(id);const fmt=x=>Number(x).toFixed(3);
function render(){const task=el('task').value,fraction=Number(el('fraction').value);
const models=Object.keys(DATA.metrics).sort((a,b)=>DATA.metrics[b][task].average_precision-DATA.metrics[a][task].average_precision);
el('results').replaceChildren();for(const id of models){const m=DATA.metrics[id][task];
const e=DATA.execution.find(x=>x.model===id&&x.fraction===fraction);const tr=document.createElement('tr');
const title=document.createElement('td');title.textContent=DATA.models[id].name;const desc=document.createElement('small');
desc.textContent=DATA.models[id].description;title.append(desc);tr.append(title);
for(const value of [fmt(m.average_precision),fmt(m.roc_auc),(100*m.accuracy).toFixed(1)+'%',
(100*m.balanced_accuracy).toFixed(1)+'%',fmt(m.brier_score),m.fit_seconds===null?(id==='monte_carlo'?'No fitting':'Reused'):m.fit_seconds.toFixed(1)+'s',e?e.mean_gross_cents.toFixed(3):'—']){
const td=document.createElement('td');td.textContent=value;tr.append(td)}
if(e)tr.lastChild.className=e.mean_gross_cents<0?'negative':'positive';el('results').append(tr)}
const first=DATA.metrics[models[0]][task];el('population').textContent=first.validation_examples.toLocaleString()+
' shared validation examples · positive-class prevalence '+(100*first.positive_rate).toFixed(1)+'% · forecast horizon 10 seconds';}
el('task').onchange=render;el('fraction').onchange=render;
el('method').textContent=DATA.protocol.training_examples.toLocaleString()+' shared sampled training windows for supervised methods; '+
DATA.protocol.moving_training_examples.toLocaleString()+' moving windows for direction. '+DATA.protocol.validation_examples.toLocaleString()+
' shared spaced validation windows. No reserved holdout scoring.';
el('protocol').textContent=JSON.stringify(DATA.protocol,null,2);render();</script></html>'''
    (output / "index.html").write_text(page.replace("__DATA__", encoded), encoding="utf-8")
    lines = ["# Kalshi Model Lab", "", "Exploratory development-validation benchmark. No live trades.", "",
             "| Model | Movement AP | Direction accuracy | Top-10% gross cents/contract |",
             "|---|---:|---:|---:|"]
    for model in sorted(metrics, key=lambda key: -metrics[key]["movement"]["average_precision"]):
        gross = next(row["mean_gross_cents"] for row in execution
                     if row["model"] == model and row["fraction"] == .1)
        lines.append(f"| {MODEL_INFO[model][0]} | {metrics[model]['movement']['average_precision']:.4f} | "
                     f"{metrics[model]['direction']['accuracy']:.2%} | {gross:+.4f} |")
    lines += ["", "Costs exclude fees, latency, slippage and capacity constraints. Taker fills are hypothetical.",
              "Supervised models share rows; GRU sees sequences, tabular learners use 50 summaries; Monte Carlo uses historical midpoint increments.",
              "GRU used this validation for checkpoint selection. These are not untouched test results.",
              "A single seed and fixed settings are used; no significance or profitability claim.",
              "Signal fractions are retrospective ranking diagnostics; cutoff ties use uniform expected selection.",
              "Monte Carlo is an IID historical-increment bootstrap: 2,048 clipped paths, 10 steps, no label fitting.",
              "Holdout outcomes were not evaluated. Open index.html for interactive results."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing results are preserved")
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from source research run")
    root = Path.cwd().resolve()
    if root not in output.parents or root / "data" in output.parents:
        raise ValueError("Write results inside project and outside synced data/")
    config = json.loads((source / "config.json").read_text())
    protected = json.loads((source / "protected_v2_period.json").read_text())
    meta = pd.read_parquet(source / "windows.parquet")
    validate_splits(meta, protected)
    training, validation = select_samples(meta, config)
    cache = ShardCache(source)
    train_change, val_change = targets(training, cache), targets(validation, cache)
    gru = verify_gru(source, validation, val_change, train_change)
    output.mkdir(parents=True)

    def progress(stage, **values):
        state = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **values}
        save_json(output / "status.json", state)
        print(json.dumps(state), flush=True)

    try:
        progress("features", training_examples=len(training), validation_examples=len(validation))
        train_x = materialize(training, cache, output, "train")
        val_x = materialize(validation, cache, output, "validation")
        training[IDENTITY].to_parquet(output / "training_indices.parquet", index=False)
        validation[IDENTITY].to_parquet(output / "validation_indices.parquet", index=False)
        moving_train, moving_val = meaningful_movement(train_change), meaningful_movement(val_change)
        for labels in (moving_train, moving_val, train_change[moving_train] > 0, val_change[moving_val] > 0):
            if len(np.unique(labels)) != 2:
                raise ValueError("Both classes required for each task")
        protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source), "source_config": config, "seed": 42,
            "training_examples": len(training), "moving_training_examples": int(moving_train.sum()),
            "validation_examples": len(validation), "moving_validation_examples": int(moving_val.sum()),
            "feature_names": FEATURE_NAMES, "feature_schema": "history_summary_v1",
            "label_policy": LABEL_POLICY,
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "selection": "Exact source GRU train sample and spaced validation; no additional tuning",
            "holdout": "Outcomes never read; metadata inspected only for split checks",
            "representations": "GRU:60x7 sequence; tabular learners:50 causal summaries; Monte Carlo:59 midpoint increments",
            "execution": "Gross hypothetical taker; fees/slippage/latency/capacity excluded",
            "caveats": "Exploratory validation; GRU checkpoint selected on this validation; correlated events; single seed",
            "monte_carlo": {"paths": 2048, "steps": 10, "seed": "42 plus stable example identity hash",
                "assumption": "IID bootstrap of the 59 historical one-second midpoint increments; clip each step to [0,1]",
                "training": "No labels fitted; parameters estimated from each available input history only",
                "probability_smoothing": "Jeffreys: (positive_count+0.5)/(total_count+1)"},
            "source_hashes": {name: digest(source / name) for name in (
                "config.json", "input_manifest.json", "windows.parquet", "protected_v2_period.json",
                "metrics.json", "validation_predictions.parquet", "direction/best.keras", "movement/best.keras")},
            "sample_hashes": {name: digest(output / name) for name in (
                "training_indices.parquet", "validation_indices.parquet")}}
        save_json(output / "protocol.json", protocol)
        quotes = quote_frame(validation, cache)
        metrics, execution = {}, []
        predictions = validation[IDENTITY].copy()
        predictions["change_cents"] = val_change
        model_dir = output / "models"
        model_dir.mkdir()
        for name in [*make_models(), "gru", "monte_carlo"]:
            model_metrics, probabilities = {}, {}
            if name == "monte_carlo":
                started = time.monotonic()
                simulation = simulate_validation(validation, cache, progress)
                simulation_seconds = time.monotonic() - started
                simulation.to_parquet(output / "monte_carlo_simulation.parquet", index=False)
            for task in ("movement", "direction"):
                progress("fitting", model=name, task=task)
                mask_train = moving_train if task == "direction" else np.ones(len(training), dtype=bool)
                mask_val = moving_val if task == "direction" else np.ones(len(validation), dtype=bool)
                train_y = (train_change[mask_train] > 0 if task == "direction" else moving_train).astype(int)
                val_y = (val_change[mask_val] > 0 if task == "direction" else moving_val).astype(int)
                started = time.monotonic()
                if name == "gru":
                    probability = gru[task]
                    fit_seconds = None
                elif name == "monte_carlo":
                    probability = simulation[task].to_numpy()
                    fit_seconds = None
                else:
                    model = make_models()[name]
                    with threadpool_limits(limits=2):
                        model.fit(train_x[mask_train], train_y)
                        fit_seconds = time.monotonic() - started
                        probability = model.predict_proba(val_x)[:, list(model.classes_).index(1)]
                    joblib.dump(model, model_dir / f"{name}_{task}.joblib")
                    save_json(model_dir / f"{name}_{task}_parameters.json",
                              {key: str(value) for key, value in model.get_params().items()})
                    importance = getattr(model, "feature_importances_", None)
                    if importance is not None:
                        pd.DataFrame({"feature": FEATURE_NAMES, "importance": importance}).sort_values(
                            "importance", ascending=False).to_csv(model_dir / f"{name}_{task}_importance.csv", index=False)
                probabilities[task] = probability
                model_metrics[task] = {**classification_metrics(val_y, probability[mask_val]),
                    "training_examples": 0 if name == "monte_carlo" else len(train_y), "fit_seconds": fit_seconds,
                    "simulation_seconds": simulation_seconds if name == "monte_carlo" else None,
                    "training": ("Historical-window simulation; no labels fitted" if name == "monte_carlo" else
                                 "Reused saved checkpoint" if name == "gru" else "Fitted in this benchmark")}
                predictions[f"{name}_{task}"] = probability
            metrics[name] = model_metrics
            execution.extend(execution_rows(name, probabilities["movement"], probabilities["direction"], quotes))
            write_outputs(output, metrics, execution, protocol)
            predictions.to_parquet(output / "validation_predictions.parquet", index=False)
            progress("model_complete", model=name, models_completed=len(metrics), models_total=len(MODEL_INFO))
        progress("complete", models_completed=len(metrics), report=str(output / "index.html"))
    except BaseException as error:
        progress("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / (
        "model_benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    if args.source_run is None:
        latest = Path("artifacts/model_lab_latest.json")
        if not latest.exists():
            parser.error("No completed corrected run yet; first run python -m src.rerun_model_lab")
        args.source_run = Path(json.loads(latest.read_text())["run_dir"])
    run_benchmark(args.source_run, args.output)


if __name__ == "__main__":
    main()
