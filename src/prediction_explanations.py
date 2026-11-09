"""Numerical explanations of saved predictors, not generated model narratives."""

import json
import numpy as np

from src.model_benchmark import FEATURE_NAMES, summarize
from src.model_traces import simulate_mc


def linear_details(model, x):
    transformed = model.named_steps["standardscaler"].transform(x)
    final = model.steps[-1][1]
    if hasattr(final, "estimators_"):
        coefficients = np.stack([est.coef_ for est in final.estimators_])
        intercept = np.array([est.intercept_[0] for est in final.estimators_])
    else:
        coefficients = np.atleast_2d(final.coef_)
        intercept = np.atleast_1d(final.intercept_)
    contributions = coefficients.astype(np.float64)*transformed[0].astype(np.float64)
    reconstructed = contributions.sum(axis=1)+intercept
    # Some saved pipelines predict in float32 BLAS; the displayed sum is float64.
    np.testing.assert_allclose(reconstructed, model.predict(x)[0], atol=1e-5,rtol=1e-6)
    return {"kind": "linear", "intercept": intercept.tolist(),
        "features": [{"name": name, "raw": float(x[0,i]), "standardized": float(transformed[0,i]),
            "coefficient": coefficients[:,i].tolist(), "contribution": contributions[:,i].tolist()}
            for i,name in enumerate(FEATURE_NAMES)],
        "reconstructed_raw": reconstructed.tolist()}


def tree_details(model, x):
    tree = model.tree_
    indices = model.decision_path(x).indices
    steps = []
    for node in indices:
        feature = int(tree.feature[node])
        steps.append({"node": int(node), "leaf": feature < 0,
            "feature": FEATURE_NAMES[feature] if feature >= 0 else None,
            "observed": float(x[0,feature]) if feature >= 0 else None,
            "threshold": float(tree.threshold[node]) if feature >= 0 else None,
            "direction": "left" if feature >= 0 and x[0,feature] <= tree.threshold[node] else "right" if feature >= 0 else None,
            "training_samples": int(tree.n_node_samples[node]),
            "value": tree.value[node].reshape(-1).tolist()})
    return {"kind": "tree", "steps": steps, "leaf": int(model.apply(x)[0]),
            "prediction": model.predict(x)[0].tolist()}


def explanation(predictor, item, model, horizon, run, include_all=False, tree_index=0):
    if model not in ("mean","ridge","sgd","decision_tree","random_forest","gradient_boosting","gru","monte_carlo") or horizon not in (30,60):
        raise ValueError("Unknown model or horizon")
    history = np.asarray(item["history"], float)[None]
    x = summarize(history)
    details = {}
    if model == "monte_carlo":
        identity = {k:item[k] for k in ("session_id","ticker","input_end")}
        trace = simulate_mc(history[0], identity, trace=True)
        raw = np.array(trace["predictions"][horizon])
        returns = np.asarray(trace["per_path_returns"][horizon])
        count = len(trace["paths"])
        selected = np.linspace(0,count-1,min(48,count),dtype=int)
        paths = np.asarray(trace["paths"])
        hist, edges = np.histogram(returns[:,0], bins=32)
        details = {"kind":"monte_carlo", "seed":trace["seed"], "path_count":count,
            "preview_indices":selected.tolist(), "preview_paths":paths[selected,:horizon+1].tolist(),
            "preview_draw_indices":np.asarray(trace["sampled_indices"])[selected,:horizon].tolist(),
            "preview_returns":returns[selected].tolist(),
            "historical_increments":trace["historical_increments"],
            "histogram":{"counts":hist.tolist(),"edges":edges.tolist()},
            "convergence_yes":(np.cumsum(returns[:,0])/np.arange(1,count+1)).tolist(),
            "scenario_percentiles":np.percentile(returns,[5,50,95],axis=0).tolist(),
            "limitations":"Bootstrap scenarios, not confidence intervals. Constant current spread; prices bounded to [spread/2, 1-spread/2]."}
        if include_all:
            details["full_trace"] = trace
    elif model == "gru":
        import tensorflow as tf
        scaled = predictor.scalers[horizon].transform(history.reshape(-1,7)).reshape(1,60,7).astype(np.float32)
        tensor = tf.convert_to_tensor(scaled)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            values = predictor.models[(model,horizon)](tensor,training=False)*100
            yes = values[:,0]
        derivative = tape.gradient(yes,tensor).numpy()[0]/predictor.scalers[horizon].scale_[None]
        raw = values.numpy()[0]
        details = {"kind":"gru", "sensitivity":np.abs(derivative).mean(axis=0).tolist(),
            "time_sensitivity":np.abs(derivative).sum(axis=1).tolist(),
            "feature_names":["bid","ask","spread","midpoint","bid_depth","ask_depth","imbalance"],
            "limitations":"Local derivative of YES return with respect to input features; sensitivity is not causal importance or an additive explanation."}
    else:
        fitted = predictor.models[(model,horizon)]
        raw = fitted.predict(x)[0]
        if model in ("ridge","sgd"):
            details = linear_details(fitted,x)
        elif model == "decision_tree":
            details = tree_details(fitted,x)
        elif model == "random_forest":
            if not 0 <= tree_index < len(fitted.estimators_):
                raise ValueError("Tree index out of range")
            outputs = np.stack([tree.predict(x)[0] for tree in fitted.estimators_])
            np.testing.assert_allclose(outputs.mean(axis=0),raw,atol=1e-7)
            details = {"kind":"forest","tree_outputs":outputs.tolist(),"selected_tree":tree_index,
                       "path":tree_details(fitted.estimators_[tree_index],x)}
        elif model == "gradient_boosting":
            stages = np.array([list(est.staged_predict(x)) for est in fitted.estimators_])[:,:,0].T
            np.testing.assert_allclose(stages[-1],raw,atol=1e-7)
            details = {"kind":"boosting","stages":stages.tolist(),
                       "limitations":"Actual cumulative predictions after each boosting stage. This is not an SGD coefficient trajectory."}
        else:
            details = {"kind":"mean","training_mean":np.asarray(fitted.constant_).reshape(-1).tolist()}
    correction = predictor.corrections[(model,horizon)]
    clipped = np.clip(raw,-104,96)
    final = np.clip(clipped+correction,-104,96)
    saved = next(p for p in item["predictions"] if p["model"]==model and p["horizon"]==horizon)
    np.testing.assert_allclose(final,[saved["yes"],saved["no"]],atol=1e-4,rtol=1e-5)
    trace_path = run/"traces"/f"{model}_{horizon}.json"
    training = json.loads(trace_path.read_text()) if trace_path.exists() else {"available":False,
        "message":"No training trace was recorded for this model/run. No synthetic replay substituted."}
    return {"model":model,"horizon":horizon,"details":details,"training":training,
        "calculation":{"raw":np.asarray(raw).tolist(),"clipped_raw":clipped.tolist(),
            "calibration_correction":correction.tolist(),"final":final.tolist(),"signal":saved["signal"],
            "margin_cents":1,"cost_buffer_cents":4},
        "input":{"features":history[0].tolist(),"summary_names":FEATURE_NAMES,"summary_values":x[0].tolist()},
        "disclaimer":"Actual saved model calculation. Returns are conservative cost-scenario estimates, not guaranteed fills or profits."}
