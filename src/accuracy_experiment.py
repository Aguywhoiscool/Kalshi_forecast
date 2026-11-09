"""Predeclared staged accuracy search with quarantined once-only final data."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from src.accuracy_data import prepare, walkforward
from src.accuracy_models import Data, fit_model, raw_predict, load_model
from src.accuracy_targets import AccuracySGD, correction, error_metrics, event_bootstrap
from src.causal_features import build_features, engineered_sequence
from src.model_benchmark import summarize
from src.swing_return_experiment import return_bounds, evaluate
from src.model_traces import simulate_mc
from src.v3_research import digest, save_json


MODELS = ("ridge", "sgd", "gradient_boosting", "gru")
CONTROLS = ("mean", "decision_tree", "random_forest", "monte_carlo_60", "monte_carlo_180")
FAMILIES = ("seven60", "engineered60", "engineered180")
DEFAULTS = {"ridge": {"alpha": 1.}, "sgd": {"learning_rate": .001, "alpha": .001},
    "gradient_boosting": {"learning_rate": .1, "max_leaf_nodes": 15, "l2_regularization": 1., "max_iter": 100},
    "gru": {"layers": [64, 32], "dropout": .2}, **{name: {} for name in CONTROLS}}
GRIDS = {"ridge": [{"alpha": a} for a in (.1, 1., 10., 100.)],
    "sgd": [{"learning_rate": lr, "alpha": alpha} for lr, alpha in itertools.product((.0001, .001), (.0001, .01))],
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2, "max_iter": 300}
        for lr, leaves, l2 in itertools.product((.03, .1), (7, 15), (1., 10.))],
    "gru": [{"layers": list(layers), "dropout": drop} for layers, drop in itertools.product(((32, 16), (64, 32), (96, 48)), (.1, .3))]}


def utc():
    return datetime.now(timezone.utc).isoformat()


def protocol(output):
    value = {"created_utc": utc(), "seed_search": 42, "refit_seeds": [7, 42, 101], "primary_seed": 42,
        "grids": GRIDS, "defaults": DEFAULTS, "families": list(FAMILIES), "interval_huber_delta_cents": 5.,
        "search": "Feature/objective ablation defaults first; lock family/objective then exact 4/4/8/6 config search by mean 3-fold observed-side MAE. First config wins exact ties.",
        "calibration": "Clip raw predictions [-104,96]; conservative mean(lower-raw); interval mean(clip(raw,lower,upper)-raw), once over all calibration sides; final clip[-104,96]. Target+calibration contrast, not pure loss ablation.",
        "observed_definition": "Side statuses0 closed quotedreturn and1 known rejectedentry zero; closed-only MAE also reported. Unknown2/3/4 retained in intervals/full-population diagnostics.",
        "gru": "Adam .001; max30epochs; patience5; checkpoint only within-training calibration objective; dollars-scaled output, Huber delta .05dollars=5c",
        "folds": "Chronological expanding session blocks40/60/80 percent train, following20percent validation; inner20percent calibration;70second embargo by input_start/target_end",
        "final_gate": "Metadata-only later than prior developmentmax+70s; >=20eventproxies. Choices/models frozen before once-only final target materialization.",
        "promotion": "No automatic live promotion; per horizon primary seed42>=5percentrelative observed-MAE improvement andpositivepaired95percent event-proxy bootstrapCI. Seeds7/101 robustness only.",
        "primary_baseline": "Strongest matched DEFAULT among mean/tree/forest/MC60/MC180/ridge/SGD/boost/GRU by mean developmentfold observedMAE. Primarywinner likewise selected from tuned4 by dev only. Same-family default contrasts secondary.",
        "bootstrap": {"repetitions": 1000, "seed": 42, "unit": "event_proxy", "minimum_observed_groups": 20,
                      "interpretation": "Unadjusted per-comparison exploratory CI and heuristic group-count screen; not power proof"},
        "execution": {"entry_delay_seconds": 1, "horizons": [30, 60], "margin_cents": 1, "cost_buffer_cents": 4}}
    root = Path(__file__).resolve().parents[1]
    old = root / "artifacts/model_lab_traced_20261109"
    value["frozen_old_secondary"] = {"run": str(old), "manifest": {str(p.relative_to(old)): digest(p)
        for p in sorted([*list((old / "models").glob("*.joblib")), old / "metrics.json",
                         *list(old.glob("gru_*/best.keras")), *list(old.glob("gru_*/scaler.joblib"))])}}
    value["implementation_sha256"] = {name: digest(root / "src" / name) for name in
        ("accuracy_data.py", "accuracy_targets.py", "accuracy_models.py", "accuracy_experiment.py", "causal_features.py", "model_traces.py")}
    value["development_artifacts_sha256"] = {p.name: digest(p) for p in sorted((output / "development").glob("*")) if p.is_file()}
    path = output / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        value["created_utc"] = previous["created_utc"]
        if previous != value:
            raise ValueError("Frozen protocol/implementation differs; use a new experiment")
    else:
        save_json(path, value)
    return value










def progress(output, stage, **extra):
    value = {"utc": utc(), "stage": stage, **extra}
    save_json(output / "status.json", value)
    line = json.dumps(value)
    print(line, flush=True)
    with (output.parent / "agent_status/research.log").open("a", encoding="utf-8") as stream:
        stream.write(line+"\n")


def crossvalidate(output, data, folds, model, family, objective, params, horizon, identifier):
    lower, upper, status = return_bounds(data.future, data.known, horizon)
    summary_path = output / "search" / identifier / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        expected = {"model": model, "family": family, "objective": objective, "params": params, "horizon": horizon,
                    "protocol_sha256": digest(output / "protocol.json")}
        if any(existing.get(k) != v for k, v in expected.items()):
            raise ValueError("Cached search protocol mismatch")
        for index, fold in enumerate(folds):
            with np.load(summary_path.parent / f"fold{index}/validation.npz") as saved:
                np.testing.assert_array_equal(saved["rows"], fold["validation"])
                score = error_metrics(saved["prediction"], lower[fold["validation"]], upper[fold["validation"]], status[fold["validation"]])["observed_mae"]
                np.testing.assert_allclose(score, existing["scores"][index], atol=1e-10)
        progress(output, "search_reused", candidate=identifier)
        return existing
    scores = []
    for index, fold in enumerate(folds):
        folder = output / "search" / identifier / f"fold{index}"
        progress(output, "training_search", candidate=identifier, fold=index+1, model=model, horizon=horizon)
        fitted = fit_model(model, family, objective, params, 42, data, lower, upper, fold["fit"], fold["calibration"], folder)
        fitted["horizon"] = horizon
        raw_cal = raw_predict(fitted, data, fold["calibration"])
        bias = correction(raw_cal, lower[fold["calibration"]], upper[fold["calibration"]], objective)
        raw_val = raw_predict(fitted, data, fold["validation"])
        prediction = np.clip(raw_val+bias, -104, 96)
        metrics = error_metrics(prediction, lower[fold["validation"]], upper[fold["validation"]], status[fold["validation"]])
        if metrics["observed_mae"] is None:
            raise ValueError("No observed sides in fold")
        scores.append(metrics["observed_mae"])
        np.savez_compressed(folder / "validation.npz", rows=fold["validation"], prediction=prediction,
            raw_calibration=raw_cal, raw_validation=raw_val, correction=bias)
        save_json(folder / "metrics.json", metrics)
    result = {"id": identifier, "model": model, "family": family, "objective": objective,
        "params": params, "horizon": horizon, "scores": scores, "mean_observed_mae": float(np.mean(scores)),
        "protocol_sha256": digest(output / "protocol.json")}
    save_json(output / "search" / identifier / "summary.json", result)
    return result


def run(output):
    output = output.resolve()
    source = Path(json.loads((output / "segments.json").read_text())["source"])
    if (output / "choices.json").exists() or (output / "final_started.json").exists():
        raise ValueError("Choices/final already frozen; use a fresh experiment, never adapt to final")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    if not (output / "development/metadata.parquet").exists():
        prepare(source, output, "development")
    protocol(output)
    data = Data(output / "development")
    folds = walkforward(data.meta)
    for index, fold in enumerate(folds):
        folder = output / "folds" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in fold.items():
            np.save(folder / f"{name}.npy", rows)
    selections, ablations, searches, primary_baselines, primary_winners = {}, [], [], {}, {}
    for horizon in (30, 60):
        defaults = []
        for control in CONTROLS:
            identifier = f"default_{control}_{horizon}"
            result = crossvalidate(output, data, folds, control, "seven60", "conservative", DEFAULTS[control], horizon, identifier)
            ablations.append(result)
            defaults.append(result)
        for model in MODELS:
            candidates = []
            for family in FAMILIES:
                for objective in (("conservative", "interval") if model in ("sgd", "gru") else ("conservative",)):
                    identifier = f"ablation_{model}_{horizon}_{family}_{objective}"
                    result = crossvalidate(output, data, folds, model, family, objective, DEFAULTS[model], horizon, identifier)
                    ablations.append(result)
                    candidates.append(result)
                    if family == "seven60" and objective == "conservative":
                        defaults.append(result)
                    save_json(output / "ablations.json", ablations)
            chosen = min(candidates, key=lambda row: row["mean_observed_mae"])
            results = []
            for index, params in enumerate(GRIDS[model]):
                identifier = f"search_{model}_{horizon}_{index}"
                result = crossvalidate(output, data, folds, model, chosen["family"], chosen["objective"], params, horizon, identifier)
                searches.append(result)
                results.append(result)
                save_json(output / "search_results.json", searches)
            selections[f"{model}_{horizon}"] = min(results, key=lambda row: row["mean_observed_mae"])
        primary_baselines[str(horizon)] = min(defaults, key=lambda row: row["mean_observed_mae"])
        primary_winners[str(horizon)] = min([selections[f"{model}_{horizon}"] for model in MODELS], key=lambda row: row["mean_observed_mae"])
    save_json(output / "choices.json", {"frozen_utc": utc(), "selections": selections,
        "primary_baselines": primary_baselines, "primary_winners": primary_winners,
        "protocol_sha256": digest(output / "protocol.json"), "segments_sha256": digest(output / "segments.json"),
        "development_metadata_sha256": digest(output / "development/metadata.parquet")})
    (output / "choices.sha256").write_text(digest(output / "choices.json"), encoding="utf-8")
    # Refit using identical within-development calibration boundaries for every
    # model/family/seed; no final values are accessed in this phase.
    sessions = data.meta.groupby("session_id").input_start.min().sort_values()
    fit_sessions = set(sessions.index[:int(len(sessions)*.8)])
    fit = data.meta.index[data.meta.session_id.isin(fit_sessions)].to_numpy()
    cal = data.meta.index[~data.meta.session_id.isin(fit_sessions) & (data.meta.input_start >= data.meta.iloc[fit].target_end.max()+70*10**9)].to_numpy()
    refits = []
    for horizon in (30, 60):
        lower, upper, status = return_bounds(data.future, data.known, horizon)
        for model in (*MODELS, *CONTROLS):
            specs = [("baseline", "seven60", "conservative", DEFAULTS[model])]
            if model in MODELS:
                chosen = selections[f"{model}_{horizon}"]
                specs.insert(0, ("selected", chosen["family"], chosen["objective"], chosen["params"]))
            for kind, family, objective, params in specs:
                for seed in (7, 42, 101):
                    name = f"{kind}_{model}_{horizon}_seed{seed}"
                    progress(output, "refitting", name=name)
                    folder = output / "refits" / name
                    fitted = fit_model(model, family, objective, params, seed, data, lower, upper, fit, cal, folder)
                    fitted["horizon"] = horizon
                    raw_cal = raw_predict(fitted, data, cal)
                    bias = correction(raw_cal, lower[cal], upper[cal], objective)
                    save_json(folder / "calibration.json", {"correction": bias.tolist(), "family": family,
                        "objective": objective, "fit_count": len(fit), "calibration_count": len(cal)})
                    refits.append({"name": name, "model": model, "horizon": horizon, "seed": seed, "kind": kind,
                                   "family": family, "objective": objective, "correction": bias.tolist()})
    save_json(output / "refits.json", refits)
    save_json(output / "refit_manifest.json", {str(path.relative_to(output)): digest(path)
        for path in sorted((output / "refits").rglob("*")) if path.is_file()})
    segments = json.loads((output / "segments.json").read_text())
    if segments["mode"] != "final_eligible":
        progress(output, "development_only", reason="Fewer than20 timestamp-only final eventproxies")
        return
    if digest(output / "choices.json") != (output / "choices.sha256").read_text():
        raise ValueError("Frozen choices changed")
    with (output / "final_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"utc": utc(), "choices_sha256": digest(output / "choices.json"),
                   "refit_manifest_sha256": digest(output / "refit_manifest.json")}, stream)
    if prepare(source, output, "final") is None:
        progress(output, "development_only", reason="Actual common180 final cohort has fewer than20eventproxies; no final outcomes opened")
        return
    final = Data(output / "final")
    final_rows = np.arange(len(final.meta))
    final_metrics, predictions = {}, {}
    diagnostics = []
    for entry in refits:
        progress(output, "final_once", name=entry["name"])
        fitted = load_model(output / "refits" / entry["name"], entry["model"], entry["family"])
        fitted["horizon"] = entry["horizon"]
        prediction = np.clip(raw_predict(fitted, final, final_rows)+entry["correction"], -104, 96)
        lower, upper, status = return_bounds(final.future, final.known, entry["horizon"])
        predictions[entry["name"]] = prediction
        metrics = error_metrics(prediction, lower, upper, status)
        final_metrics[entry["name"]] = metrics
        np.save(output / "final" / f"{entry['name']}_prediction.npy", prediction)
        trades, economics = evaluate(entry["model"], entry["horizon"], prediction, final.history[:, -60:], final.meta,
                                    final.future, final.known, 1., 4.)
        metrics["execution"] = economics
        trades.to_parquet(output / "final" / f"{entry['name']}_trades.parquet", index=False)
        if entry["seed"] == 42:
            seconds = final.meta.seconds_to_close.to_numpy()
            bands = np.select([seconds <= 600, seconds <= 1800, seconds <= 3600], ["under10m", "10to30m", "30to60m"], default="over60m")
            for grouping, labels in (("time_band", bands), ("series", final.meta.series.to_numpy())):
                for label in np.unique(labels):
                    keep = labels == label
                    diagnostics.append({"name": entry["name"], "grouping": grouping, "group": str(label),
                        "candidates": int(keep.sum()), **error_metrics(prediction[keep], lower[keep], upper[keep], status[keep])})
    # Frozen old artifacts are a secondary comparator only. Their development
    # predictions are not used for tuning because they saw earlier development.
    prior = json.loads((output / "protocol.json").read_text())["frozen_old_secondary"]
    old = Path(prior["run"])
    for name, expected in prior["manifest"].items():
        if digest(old / name) != expected:
            raise ValueError("Frozen old baseline bytes changed")
    old_metrics = json.loads((old / "metrics.json").read_text())
    for horizon in (30, 60):
        lower, upper, status = return_bounds(final.future, final.known, horizon)
        for model in ("mean", "ridge", "sgd", "decision_tree", "random_forest", "gradient_boosting", "gru", "monte_carlo"):
            name = f"frozen_old_{model}_{horizon}"
            if model == "gru":
                fitted = {"kind": "gru", "family": "seven60",
                    "model": tf.keras.models.load_model(old / f"gru_{horizon}/best.keras", compile=False),
                    "scaler": joblib.load(old / f"gru_{horizon}/scaler.joblib")}
            elif model == "monte_carlo":
                fitted = {"kind": "monte_carlo", "model_name": "monte_carlo_60", "family": "seven60", "horizon": horizon}
            else:
                fitted = {"kind": "tabular", "family": "seven60", "model": joblib.load(old / "models" / f"{model}_{horizon}.joblib")}
            bias = old_metrics[f"{model}_{horizon}"]["calibration_correction_cents"]
            prediction = np.clip(raw_predict(fitted, final, final_rows)+bias, -104, 96)
            final_metrics[name] = error_metrics(prediction, lower, upper, status)
            np.save(output / "final" / f"{name}_prediction.npy", prediction)
    gates, primary_gates = {}, {}
    for horizon in (30, 60):
        lower, upper, status = return_bounds(final.future, final.known, horizon)
        for model in MODELS:
            selected = predictions[f"selected_{model}_{horizon}_seed42"]
            baseline = predictions[f"baseline_{model}_{horizon}_seed42"]
            gates[f"{model}_{horizon}"] = event_bootstrap(selected, baseline, lower, status, final.meta.event_proxy)
        winner = primary_winners[str(horizon)]["model"]
        strongest = primary_baselines[str(horizon)]["model"]
        primary_gates[str(horizon)] = {"winner": winner, "strongest_default": strongest,
            **event_bootstrap(predictions[f"selected_{winner}_{horizon}_seed42"],
                predictions[f"baseline_{strongest}_{horizon}_seed42"], lower, status, final.meta.event_proxy),
            "interpretation": "Unadjusted per-horizon exploratory observed-side screen, not global significance or power proof"}
    save_json(output / "final_metrics.json", final_metrics)
    save_json(output / "gates.json", gates)
    save_json(output / "primary_gates.json", primary_gates)
    pd.DataFrame(diagnostics).to_csv(output / "subgroup_diagnostics.csv", index=False)
    seed_spread = {}
    for horizon in (30, 60):
        for model in (*MODELS, *CONTROLS):
            for kind in (("selected", "baseline") if model in MODELS else ("baseline",)):
                values = [final_metrics[f"{kind}_{model}_{horizon}_seed{seed}"]["observed_mae"] for seed in (7, 42, 101)]
                seed_spread[f"{kind}_{model}_{horizon}"] = {"seeds": [7,42,101], "observed_mae": values,
                    "primary_seed":42, "note":"Robustness observations, not independent event samples or a seed-selection rule"}
    save_json(output / "seed_spread.json", seed_spread)
    save_json(output / "final_completed.json", {"utc": utc(), "choices_sha256": digest(output / "choices.json"),
        "candidates": len(final.meta), "event_proxies": final.meta.event_proxy.nunique(), "promotion": "none; independent audit and root review required"})
    progress(output, "complete", final_rows=len(final.meta), final_events=int(final.meta.event_proxy.nunique()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
