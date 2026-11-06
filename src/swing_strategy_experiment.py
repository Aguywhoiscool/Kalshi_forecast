"""Offline, all-spread large-return forecasting with predeclared exits.

Requires the frozen development opportunity cohort. Never submits orders.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.analyze_swing_opportunities import common_ends
from src.model_benchmark import MODEL_INFO, make_models, summarize, validate_splits
from src.v3_research import FEATURES, digest, overlaps, save_json, valid_rows


EPS = 1e-9
HORIZONS = (30, 60)
LABEL_GROSS_CENTS = 5.
PRIMARY_COST_CENTS = 4.
TAKE_PROFIT_GROSS_CENTS = 10.
STOP_LOSS_GROSS_CENTS = 5.


def trade_outcomes(future, side, horizon, delay=1, rule="fixed"):
    """One-contract quotes; low exit depth is unresolved, never silently dropped.

    Columns: bid, ask, bid_depth, ask_depth. Signals precede the entry observation.
    Bracket thresholds are first sampled crossings, not hindsight-optimal exits.
    """
    future = np.asarray(future, dtype=np.float64)
    side = np.asarray(side, dtype=int)
    if future.ndim != 3 or future.shape[2] != 4 or future.shape[1] <= horizon:
        raise ValueError("Future quotes do not cover the declared horizon")
    if side.shape != (len(future),) or not np.isin(side, [1, 2]).all():
        raise ValueError("Side must be YES=1 or NO=2 for each example")
    if not 0 <= delay < horizon or rule not in ("fixed", "bracket"):
        raise ValueError("Invalid entry delay or exit rule")
    n = len(future)
    index = np.arange(n)
    yes = side == 1
    entry = np.where(yes, future[:, delay, 1], 1 - future[:, delay, 0])
    entry_depth = np.where(yes, future[:, delay, 3], future[:, delay, 2])
    liquidation = np.where(yes[:, None], future[:, :, 0], 1 - future[:, :, 1])
    pnl = (liquidation - entry[:, None]) * 100
    exits = np.full(n, horizon, dtype=int)
    if rule == "bracket":
        path = pnl[:, delay + 1:horizon + 1]
        hits = (path >= TAKE_PROFIT_GROSS_CENTS - EPS) | (path <= -STOP_LOSS_GROSS_CENTS + EPS)
        any_hit = hits.any(axis=1)
        exits[any_hit] = delay + 1 + hits[any_hit].argmax(axis=1)
    exit_depth = np.where(yes, future[index, exits, 2], future[index, exits, 3])
    gross_quote = pnl[index, exits]
    # No full depth at the intended exit: don't invent a completed fill. The
    # conservative scenario writes the remaining contract value to zero.
    unresolved = exit_depth < 1
    conservative = np.where(unresolved, -entry * 100, gross_quote)
    return {"entry_price": entry, "entry_depth": entry_depth,
            "exit_depth": exit_depth, "exit_index": exits,
            "gross_quote_cents": gross_quote, "gross_conservative_cents": conservative,
            "unresolved_exit": unresolved, "entry_fillable": entry_depth >= 1}


def fixed_labels(future, horizon):
    yes = trade_outcomes(future, np.ones(len(future), int), horizon)
    no = trade_outcomes(future, np.full(len(future), 2, int), horizon)
    y = np.zeros(len(future), dtype=np.int32)
    wins_yes = ((yes["gross_quote_cents"] >= LABEL_GROSS_CENTS - EPS)
                & yes["entry_fillable"] & ~yes["unresolved_exit"])
    wins_no = ((no["gross_quote_cents"] >= LABEL_GROSS_CENTS - EPS)
               & no["entry_fillable"] & ~no["unresolved_exit"])
    if (wins_yes & wins_no).any():
        raise ValueError("Both complementary sides cannot have positive fixed-exit returns")
    y[wins_yes], y[wins_no] = 1, 2
    return y


def inner_split(meta):
    """Keep calibration inside original training, with disjoint sessions/purge."""
    training = meta.loc[meta.split == "train"]
    sessions = training.groupby("session_id").agg(start=("input_start", "min"),
        count=("input_end", "size")).sort_values("start")
    cut = int(np.searchsorted(sessions["count"].cumsum().to_numpy(), len(training) * .8)) + 1
    if not 0 < cut < len(sessions):
        raise ValueError("Insufficient sessions for chronological calibration")
    fit_sessions = set(sessions.index[:cut])
    fit = training.index[training.session_id.isin(fit_sessions)].to_numpy()
    boundary = int(meta.loc[fit, "target_end"].max()) + 70 * 10**9
    calibration = training.index[(~training.session_id.isin(fit_sessions))
                                 & (training.input_start >= boundary)].to_numpy()
    validation = meta.index[meta.split == "val"].to_numpy()
    if min(len(fit), len(calibration), len(validation)) == 0:
        raise ValueError("Empty fit/calibration/validation population")
    return fit, calibration, validation


def entry_choice(probabilities, histories):
    """Choose direction and score using predictions/current depth only."""
    probabilities = np.asarray(probabilities, dtype=float)
    yes = probabilities[:, 1].copy()
    no = probabilities[:, 2].copy()
    yes[histories[:, -1, 5] < 1] = -1
    no[histories[:, -1, 4] < 1] = -1
    side = np.where(yes > no, 1, 2)
    score = np.maximum(yes, no)
    score[yes == no] = -1  # Equal predictions are not a directional signal.
    return side, score


def action_mask(score, rule, threshold):
    score = np.asarray(score)
    if rule == "p50":
        return score >= .5
    if rule == "calibration_top1pct":
        return score > threshold  # Never select tied constant-prior forecasts.
    raise ValueError("Unknown entry rule")


def event_metrics(labels, probability):
    result = {}
    for side, name in ((1, "yes"), (2, "no")):
        truth = labels == side
        p = probability[:, side]
        result[name] = {"positive_examples": int(truth.sum()), "examples": len(truth),
            "prevalence": float(truth.mean()), "brier_score": float(brier_score_loss(truth, p)),
            "average_precision": float(average_precision_score(truth, p)) if truth.any() else None,
            "roc_auc": float(roc_auc_score(truth, p)) if len(np.unique(truth)) == 2 else None}
    return result


def prepare(source, diagnostics, output, progress):
    full_meta = pd.read_parquet(source / "windows.parquet")
    protected = json.loads((source / "protected_v2_period.json").read_text())
    validate_splits(full_meta, protected)
    diagnostic_protocol = json.loads((diagnostics / "protocol.json").read_text())
    if Path(diagnostic_protocol["source"]).resolve() != source:
        raise ValueError("Diagnostic cohort and source run differ")
    meta = pd.read_parquet(diagnostics / "development_diagnostics.parquet")
    meta = meta.loc[meta.horizon_seconds == 60, ["split", "session_id", "ticker", "input_start", "input_end"]]
    meta = meta.sort_values(["input_end", "ticker"]).reset_index(drop=True)
    if not meta.split.isin(["train", "val"]).all():
        raise ValueError("Refusing reserved outcome access")
    if meta.groupby("session_id").split.nunique().max() > 1:
        raise ValueError("Session crosses development partitions")
    holdout_sessions = set(full_meta.loc[full_meta.split == "holdout", "session_id"])
    if meta.session_id.isin(holdout_sessions).any():
        raise ValueError("Reserved v3 holdout session")
    if meta.session_id.isin(protected["sessions"]).any():
        raise ValueError("Reserved v2 session")
    bounds = {split: (int(group.input_start.min()), int(group.target_end.max()))
              for split, group in full_meta.groupby("split") if split in ("train", "val")}
    hist = np.lib.format.open_memmap(output / "history.npy", mode="w+", dtype=np.float64,
                                   shape=(len(meta), 60, 7))
    future = np.lib.format.open_memmap(output / "future_quotes.npy", mode="w+", dtype=np.float64,
                                     shape=(len(meta), 61, 4))
    future_times = np.lib.format.open_memmap(output / "future_times.npy", mode="w+", dtype=np.int64,
                                          shape=(len(meta), 61))
    tabular = np.lib.format.open_memmap(output / "summary_features.npy", mode="w+", dtype=np.float32,
                                       shape=(len(meta), 50))
    manifest = json.loads((source / "input_manifest.json").read_text())
    files = {item["name"]: item for item in manifest["files"]}
    provenance = []
    for number, (session, selected) in enumerate(meta.groupby("session_id", sort=False), 1):
        name = f"features_{session}.csv"
        item = files[name]
        path = source / "snapshot" / name
        if digest(path) != item["sha256"]:
            raise ValueError("Snapshot hash mismatch: " + name)
        provenance.append({"file": name, "sha256": item["sha256"]})
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        for col in FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.loc[valid_rows(frame)].drop_duplicates(["session_id", "ticker", "timestamp"])
        for ticker, rows in selected.groupby("ticker", sort=False):
            group = frame.loc[(frame.session_id.astype(str) == str(session)) & (frame.ticker == ticker)].sort_values("timestamp")
            times = group.timestamp.array.as_unit("ns").asi8
            ends = np.searchsorted(times, rows.input_end.to_numpy())
            if not np.isin(ends, common_ends(times)).all():
                raise ValueError("Non-contiguous or incomplete history/forecast")
            np.testing.assert_array_equal(times[ends - 59], rows.input_start)
            np.testing.assert_array_equal(times[ends], rows.input_end)
            target = times[ends + 60]
            if overlaps(times[ends - 59], target, protected).any():
                raise ValueError("Reserved time overlap")
            for split, part in rows.groupby("split"):
                first, last = bounds[split]
                positions = rows.index.get_indexer(part.index)
                if (part.input_start < first).any() or (target[positions] > last).any():
                    raise ValueError("Original development boundaries violated")
            history = group[FEATURES].to_numpy(dtype=float)[ends[:, None] + np.arange(-59, 1)]
            hist[rows.index] = history
            tabular[rows.index] = summarize(history)
            future[rows.index] = group[["bid", "ask", "bid_depth", "ask_depth"]].to_numpy(dtype=float)[
                ends[:, None] + np.arange(61)]
            future_times[rows.index] = times[ends[:, None] + np.arange(61)]
        if number % 100 == 0:
            progress("dataset", sessions=number)
    for array in (hist, future, future_times, tabular):
        array.flush()
    meta["target_end"] = future_times[:, 60]
    for _, group in meta.groupby(["session_id", "ticker"]):
        if (group.input_start.to_numpy()[1:] <= group.target_end.to_numpy()[:-1]).any():
            raise ValueError("Overlapping strategy positions/history windows")
    meta.to_parquet(output / "sample_metadata.parquet", index=False)
    save_json(output / "source_files.json", provenance)
    return meta, hist, future, future_times, tabular


def simulate_monte_carlo(history, meta, paths=2048):
    """Forecast large returns with locally bootstrapped paths and constant spread.

    This predicts hypothetical future quotes; no future observed values enter it.
    Depth at future times is not simulated, so the capacity-aware label is harder
    than this simplified baseline's assumptions.
    """
    result = {h: np.empty((len(meta), 3)) for h in HORIZONS}
    for position, row in enumerate(meta.itertuples()):
        past = history[position]
        increments = np.diff(past[:, 3])
        spread = past[-1, 1] - past[-1, 0]
        mid = np.full(paths, past[-1, 3])
        identity = f"swing42|{row.session_id}|{row.ticker}|{row.input_end}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        entry_yes = entry_no = None
        for step in range(1, 61):
            mid = np.clip(mid + rng.choice(increments, size=paths), spread / 2, 1 - spread / 2)
            if step == 1:
                entry_yes, entry_no = mid + spread / 2, 1 - (mid - spread / 2)
            if step in HORIZONS:
                yes = (mid - spread / 2 - entry_yes) * 100 >= LABEL_GROSS_CENTS - EPS
                no = ((1 - (mid + spread / 2)) - entry_no) * 100 >= LABEL_GROSS_CENTS - EPS
                counts = np.array([int((~yes & ~no).sum()), int(yes.sum()), int(no.sum())])
                result[step][position] = (counts + .5) / (paths + 1.5)
    return result


def neural_probabilities(hist, labels, fit, calibration, validation, output, horizon, progress, epochs):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    directory = output / f"gru_{horizon}s"
    directory.mkdir()
    scaler = StandardScaler()
    for start in range(0, len(fit), 256):
        raw = hist[fit[start:start + 256]]
        scaler.partial_fit(raw.reshape(-1, 7))
    joblib.dump(scaler, directory / "scaler.joblib")

    class Batches(tf.keras.utils.Sequence):
        def __init__(self, positions, with_labels, shuffle=False):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.positions, self.with_labels, self.shuffle = positions.copy(), with_labels, shuffle
            self.rng = np.random.default_rng(42)
        def __len__(self):
            return (len(self.positions) + 255) // 256
        def __getitem__(self, index):
            positions = self.positions[index * 256:(index + 1) * 256]
            raw = hist[positions]
            x = scaler.transform(raw.reshape(-1, 7)).reshape(raw.shape).astype(np.float32)
            return (x, labels[positions]) if self.with_labels else x
        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.positions)

    model = tf.keras.Sequential([tf.keras.layers.Input((60, 7)),
        tf.keras.layers.GRU(64, return_sequences=True), tf.keras.layers.Dropout(.2),
        tf.keras.layers.GRU(32), tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(3, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="sparse_categorical_crossentropy")
    progress("training_gru", horizon=horizon, training=len(fit), calibration=len(calibration))
    model.fit(Batches(fit, True, True), validation_data=Batches(calibration, True), epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            tf.keras.callbacks.ModelCheckpoint(str(directory / "best.keras"), monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(str(directory / "epochs.csv"))])
    model = tf.keras.models.load_model(directory / "best.keras")
    result = (model.predict(Batches(calibration, False), verbose=0),
              model.predict(Batches(validation, False), verbose=0))
    tf.keras.backend.clear_session()
    return result


def summarize_trades(trades, context):
    filled = trades.loc[trades.entry_fillable]
    net = filled.net_conservative_cents.to_numpy()
    grouped = filled.groupby("exit_time_ns").net_conservative_cents.sum().sort_index()
    cumulative = np.r_[0., grouped.cumsum().to_numpy()]
    drawdown = float((np.maximum.accumulate(cumulative) - cumulative).max())
    return {**context, "signals": len(trades), "trades": len(filled),
        "unfilled_entry_attempts": int((~trades.entry_fillable).sum()),
        "unresolved_exits": int(filled.unresolved_exit.sum()),
        "total_net_scenario_cents": float(net.sum()),
        "mean_net_scenario_cents": float(net.mean()) if len(net) else None,
        "win_rate": float((net > EPS).mean()) if len(net) else None,
        "worst_trade_cents": float(net.min()) if len(net) else None,
        "maximum_realized_drawdown_cents": drawdown}


def strategy_rows(model, horizon, side, score, threshold, meta, histories, future, future_times, output):
    summaries = []
    for entry_rule in ("p50", "calibration_top1pct"):
        selected = np.flatnonzero(action_mask(score, entry_rule, threshold))
        for delay in (0, 1):
            for policy in ("fixed", "bracket"):
                outcome = trade_outcomes(future[selected], side[selected], horizon, delay, policy)
                for cost in (0., 2., 4., 5.):
                    trade = meta.iloc[selected][["session_id", "ticker", "input_end"]].reset_index(drop=True)
                    for key, value in outcome.items():
                        trade[key] = value
                    trade["side"] = side[selected]
                    trade["score"] = score[selected]
                    trade["exit_time_ns"] = future_times[selected, outcome["exit_index"]]
                    trade["net_conservative_cents"] = np.where(trade.entry_fillable,
                        trade.gross_conservative_cents - cost, 0.)
                    trade["net_quoted_cents"] = np.where(trade.entry_fillable, trade.gross_quote_cents - cost, 0.)
                    context = {"model": model, "horizon_seconds": horizon, "entry_rule": entry_rule,
                        "entry_delay_observations": delay, "exit_policy": policy, "cost_buffer_cents": cost}
                    summaries.append(summarize_trades(trade, context))
                    if delay == 1 and cost == PRIMARY_COST_CENTS:
                        trade = trade.assign(**context)
                        name = f"{model}_{horizon}_{entry_rule}_{policy}.parquet"
                        trade.to_parquet(output / "trades" / name, index=False)
    return summaries


def write_report(output, metrics, summaries, protocol):
    save_json(output / "classification_metrics.json", metrics)
    results = pd.DataFrame(summaries)
    results.to_csv(output / "strategy_results.csv", index=False)
    primary = results.loc[(results.entry_delay_observations == 1) & (results.cost_buffer_cents == 4)
        & (results.exit_policy == "fixed") & (results.entry_rule == "p50")]
    rows = ["# Large-swing strategy experiment", "", "Offline exploratory development validation; no orders.",
        "Primary rule: forecast >=5c gross return with probability >=0.5; one observation entry delay;",
        "fixed 30s/60s exit after the signal; additional 4c round-trip cost scenario.",
        "All valid spreads retained; no future width filter. No hindsight direction/peak selection.", "",
        "| Model | Horizon | Trades | Unresolved exits | Total scenario cents | Mean scenario cents |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        mean = "—" if pd.isna(row.mean_net_scenario_cents) else f"{row.mean_net_scenario_cents:+.3f}"
        rows.append(f"| {MODEL_INFO.get(row.model, (row.model,))[0]} | {row.horizon_seconds}s | {row.trades} | {row.unresolved_exits} | {row.total_net_scenario_cents:+.3f} | {mean} |")
    audit_path = output / "audit.json"
    audit_record = json.loads(audit_path.read_text()) if audit_path.exists() else None
    rows += ["", "## Interpret carefully", "- Cost buffers are scenarios, not actual market-specific fees.",
        "- Entry/exit fills at displayed quotes are hypothetical. One-contract top-depth checks are included.",
        "- Low exit depth is unresolved: the primary conservative scenario values that position at zero; it is not a claimed realized fill.",
        "- Missing/one-sided future observations are censored by complete-quote cohort eligibility.",
        "- The top-1% entry alternative uses a threshold fixed from chronological calibration inside original training, never validation ranks.",
        "- Bracket sensitivity uses first sampled +10c take-profit / -5c stop-loss crossings, then time exit; overshoots use the observed price.",
        "- Both exit policies and all cost/latency scenarios were fixed before fitting. Do not cherry-pick the best validation variant.",
        "- No-trade is valid. Zero trades do not demonstrate profitability or prove that no learnable signal exists.",
        "- Spacing separates completed modeled positions; unresolved exits may persist and prevent a deployable portfolio interpretation. Related markets remain correlated.",
        "- Original reserved v2 test and v3 holdout outcomes remain untouched. Single seed, exploratory validation only."]
    if audit_record is not None:
        rows += ["", "## Independent audit", "Status: " + audit_record["status"],
                 *["- " + item for item in audit_record["checks"]],
                 *["- " + item for item in audit_record["warnings"]]]
    (output / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # Standalone report with explicit controls; all numbers come from saved CSV.
    payload = json.dumps({"rows": summaries, "models": {k: v[0] for k,v in MODEL_INFO.items()}}, allow_nan=False).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kalshi swing strategy lab</title><style>body{font:16px system-ui;background:#101b2b;color:#e0e9f4;padding:32px;max-width:1250px;margin:auto}
p{line-height:1.6;color:#b9c9df}select{padding:8px;background:#20334e;color:white;margin:8px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:right;border-bottom:1px solid #40516a}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}a{color:#90caff}</style>
<h1>Large-swing strategy lab</h1><p>Predict ≥5¢ quoted gross returns and choose YES, NO, or no trade. All spreads included.</p>
<p><strong>Offline development validation.</strong> Cost buffers are assumed round-trip scenarios, not actual fees. Quotes do not guarantee fills.
Unresolved exits are conservatively valued at zero. Missing-future-quote episodes remain censored. No live orders or holdout scoring.</p>
<p><strong>Small samples:</strong> a positive total from one or two trades is not evidence of a reliable edge.
SGD results are budget-limited if its iteration cap was reached; check the independent audit warnings.
Unresolved-exit mark-downs are conservative scenarios, not realized liquidation prices.</p>
<label>Horizon<select id="h"><option>30</option><option selected>60</option></select></label>
<label>Entry rule<select id="e"><option value="p50">Probability ≥50% (primary)</option><option value="calibration_top1pct">Above training-calibration 99th percentile</option></select></label>
<label>Exit<select id="p"><option value="fixed">Fixed horizon (primary)</option><option value="bracket">+10¢ / −5¢ bracket, then time exit</option></select></label>
<label>Additional round-trip cost<select id="c"><option>0</option><option>2</option><option selected>4</option><option>5</option></select>¢</label>
<label>Entry delay<select id="d"><option>0</option><option selected>1</option></select> observations</label>
<div class="scroll"><table><thead><tr><th>Model</th><th>Signals</th><th>Trades</th><th>Unresolved</th><th>Total scenario ¢</th><th>Mean scenario ¢</th><th>Win rate</th><th>Drawdown ¢</th></tr></thead><tbody id="r"></tbody></table></div>
<p>All alternatives were predeclared. Viewing the strongest validation variant does not make it an independently tested strategy.
The calibration threshold is chosen inside original training; the evaluation period is never used to set it. No-trade is a baseline.</p>
<p><a href="REPORT.md">Method and limitations</a> · <a href="strategy_results.csv">All strategy results</a> · <a href="classification_metrics.json">Prediction metrics</a> · <a href="protocol.json">Protocol</a> · <a href="audit.json">Independent audit</a></p>
<script>const DATA=__DATA__;const el=id=>document.getElementById(id);const fmt=x=>x===null?'—':Number(x).toFixed(3);
function render(){const rows=DATA.rows.filter(x=>x.horizon_seconds===Number(el('h').value)&&x.entry_rule===el('e').value&&x.exit_policy===el('p').value&&x.cost_buffer_cents===Number(el('c').value)&&x.entry_delay_observations===Number(el('d').value));el('r').replaceChildren();
for(const x of rows){const tr=document.createElement('tr');for(const value of [DATA.models[x.model]||x.model,x.signals,x.trades,x.unresolved_exits,fmt(x.total_net_scenario_cents),fmt(x.mean_net_scenario_cents),x.win_rate===null?'—':(100*x.win_rate).toFixed(1)+'%',fmt(x.maximum_realized_drawdown_cents)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('r').append(tr)}}
for(const id of ['h','e','p','c','d'])el(id).onchange=render;render();</script>'''
    (output / "index.html").write_text(page.replace("__DATA__", payload), encoding="utf-8")


def run(output, diagnostics, epochs):
    root = Path.cwd().resolve()
    output = output.resolve()
    if root not in output.parents or root / "data" in output.parents or output.exists():
        raise ValueError("Use a new artifact directory inside project, outside data/")
    source = Path(json.loads((root / "artifacts/model_lab_latest.json").read_text())["run_dir"]).resolve()
    output.mkdir(parents=True)
    (output / "models").mkdir()
    (output / "trades").mkdir()
    save_json(output / "process.json", {"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()})

    def progress(stage, **extra):
        state = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
        save_json(output / "status.json", state)
        print(json.dumps(state), flush=True)

    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "horizons": HORIZONS, "gross_label_threshold_cents": LABEL_GROSS_CENTS,
        "primary": "p>=0.50; one-observation delayed entry; fixed exit; 4c additional round-trip cost buffer",
        "alternative_entry": "strictly above calibration 99th-percentile score (calibration in original training); no outcome-based threshold search",
        "cost_scenarios_cents": [0, 2, 4, 5], "entry_delays_observations": [0, 1],
        "bracket": {"take_profit_gross_cents": 10, "stop_loss_gross_cents": 5, "time_exit": "forecast horizon"},
        "contract_count": 1, "capacity": "Current and entry quote depth checked; unresolved exit valued at zero in conservative scenario",
        "models": list(MODEL_INFO), "seed": 42, "gru_epochs": epochs,
        "validation_use": "Evaluation only; original training split further divided chronologically into fit/calibration with 70s purge",
        "spread_filter": "No width cutoff; all valid two-sided quotes retained",
        "holdout": "No reserved outcome access", "fees": "Cost buffers only; no historical market-specific fee mapping available",
        "fee_reference": "https://kalshi.com/fee-schedule", "diagnostics_sha256": digest(diagnostics / "development_diagnostics.parquet")}
    save_json(output / "protocol.json", protocol)  # Freeze choices before fitting.
    try:
        progress("preparing")
        meta, histories, future, future_times, tabular = prepare(source, diagnostics, output, progress)
        fit, calibration, validation = inner_split(meta)
        for name, rows in (("fit", fit), ("calibration", calibration), ("validation", validation)):
            meta.iloc[rows].to_parquet(output / f"{name}_indices.parquet", index=False)
        protocol["populations"] = {"fit": len(fit), "calibration": len(calibration), "validation": len(validation)}
        protocol["sample_metadata_sha256"] = digest(output / "sample_metadata.parquet")
        save_json(output / "protocol.json", protocol)
        combined = np.r_[calibration, validation]
        progress("monte_carlo", examples=len(combined))
        mc = simulate_monte_carlo(histories[combined], meta.iloc[combined].reset_index(drop=True))
        metrics, summaries = {}, []
        # CPU-only, bounded threading; one GRU at a time.
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        for horizon in HORIZONS:
            labels = fixed_labels(future, horizon)
            np.save(output / f"labels_{horizon}.npy", labels)
            save_json(output / f"label_counts_{horizon}.json", {
                name: {str(key): int((labels[index] == key).sum()) for key in (0, 1, 2)}
                for name, index in (("fit", fit), ("calibration", calibration), ("validation", validation))})
            for name in [*make_models(), "gru", "monte_carlo"]:
                progress("fitting", model=name, horizon=horizon)
                if name == "gru":
                    p_cal, p_val = neural_probabilities(histories, labels, fit, calibration, validation, output, horizon, progress, epochs)
                elif name == "monte_carlo":
                    p_cal, p_val = mc[horizon][:len(calibration)], mc[horizon][len(calibration):]
                else:
                    model = make_models()[name]
                    if name == "decision_tree":
                        model.set_params(min_samples_leaf=20)
                    elif name == "random_forest":
                        model.set_params(min_samples_leaf=10)
                    with threadpool_limits(limits=2):
                        model.fit(tabular[fit], labels[fit])
                        p_cal, p_val = np.zeros((len(calibration), 3)), np.zeros((len(validation), 3))
                        p_cal[:, model.classes_.astype(int)] = model.predict_proba(tabular[calibration])
                        p_val[:, model.classes_.astype(int)] = model.predict_proba(tabular[validation])
                    joblib.dump(model, output / "models" / f"{name}_{horizon}.joblib")
                    save_json(output / "models" / f"{name}_{horizon}_params.json", {k: str(v) for k,v in model.get_params().items()})
                if not np.isfinite(p_val).all() or not np.allclose(p_val.sum(axis=1), 1, atol=1e-5):
                    raise ValueError("Invalid predicted class probabilities")
                side_cal, score_cal = entry_choice(p_cal, histories[calibration])
                side, score = entry_choice(p_val, histories[validation])
                eligible_cal = score_cal[score_cal >= 0]
                threshold = float(np.quantile(eligible_cal, .99)) if len(eligible_cal) else 1.
                metrics[f"{name}_{horizon}"] = {"model": name, "horizon": horizon,
                    "calibration_threshold": threshold, "event_metrics": event_metrics(labels[validation], p_val)}
                np.savez_compressed(output / f"{name}_{horizon}_probabilities.npz", calibration=p_cal, validation=p_val)
                summaries.extend(strategy_rows(name, horizon, side, score, threshold, meta.iloc[validation].reset_index(drop=True),
                    histories[validation], future[validation], future_times[validation], output))
                write_report(output, metrics, summaries, protocol)
                progress("model_complete", model=name, horizon=horizon)
            # Zero-position baseline: no entry, no cost, no risk.
            summaries.extend(strategy_rows("no_trade", horizon, np.ones(len(validation), int),
                np.full(len(validation), -1.), 1., meta.iloc[validation].reset_index(drop=True),
                histories[validation], future[validation], future_times[validation], output))
            # Baseline threshold is fixed on past-only calibration volatility.
            volatility_col = 49
            cal_vol = tabular[calibration, volatility_col]
            threshold = float(np.quantile(cal_vol, .99))
            momentum = histories[validation, -1, 3] - histories[validation, -11, 3]
            side = np.where(momentum > 0, 1, 2)
            score = np.asarray(tabular[validation, volatility_col], dtype=float).copy()
            score[momentum == 0] = -1
            # Only the calibration entry rule is meaningful for raw volatility;
            # p50 is deliberately disabled, since volatility is not a probability.
            baseline_rows = strategy_rows("volatility_momentum", horizon, side, score, threshold,
                meta.iloc[validation].reset_index(drop=True), histories[validation], future[validation], future_times[validation], output)
            summaries.extend(row for row in baseline_rows if row["entry_rule"] == "calibration_top1pct")
        write_report(output, metrics, summaries, protocol)
        save_json(output / "verification.json", {"source_hashes_checked": True,
            "original_dev_boundaries_checked": True, "reserved_sessions_and_times_excluded": True,
            "fit_calibration_session_separation": True, "no_validation_threshold_selection": True,
            "no_peak_exit_or_future_direction_selection": True})
        progress("auditing")
        from src.audit_swing_strategy import audit
        audit(output)
        write_report(output, metrics, summaries, protocol)
        progress("complete", report=str(output / "index.html"))
    except BaseException:
        progress("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/swing_strategy_20261104"))
    parser.add_argument("--diagnostics", type=Path, default=Path("artifacts/swing_opportunities_20261104"))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.output, args.diagnostics, args.epochs)
