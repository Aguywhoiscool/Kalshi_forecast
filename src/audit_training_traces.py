"""Replay saved training observations against source fit rows and fitted models."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.v3_research import save_json


def audit(run):
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    lookup = pd.MultiIndex.from_frame(meta[["session_id", "ticker", "input_end"]])
    x = np.load(run / "summary_features.npy", mmap_mode="r")
    history = np.load(run / "history.npy", mmap_mode="r")
    counts = {}
    for horizon in (30, 60):
        fit = pd.read_parquet(run / f"fit_{horizon}.parquet")
        ids = lookup.get_indexer(pd.MultiIndex.from_frame(fit[["session_id", "ticker", "input_end"]]))
        with np.load(run / f"targets_{horizon}.npz") as saved:
            target = saved["lower"]
        model = joblib.load(run / "models" / f"sgd_{horizon}.joblib")
        trace = json.loads((run / "traces" / f"sgd_{horizon}.json").read_text())
        for step in trace["steps"]:
            np.testing.assert_allclose(step["after"], np.asarray(step["before"])-step["learning_rate"]*np.asarray(step["gradient"]), atol=1e-12)
        sgd = model.steps[-1][1]
        np.testing.assert_array_equal(trace["final_coefficients"], sgd.coef_)
        np.testing.assert_array_equal(trace["final_intercepts"], sgd.intercept_)
        batch = np.random.default_rng(42).permutation(len(ids))[:sgd.batch_size]
        scaled = model.named_steps["standardscaler"].transform(x[ids[batch]])
        residual = -target[ids[batch]]
        gradient = residual.T @ scaled/len(batch)
        np.testing.assert_allclose(trace["steps"][0]["gradient"], gradient[:, :2], atol=1e-10)
        np.testing.assert_allclose(trace["steps"][0]["gradient_norm"], np.linalg.norm(gradient), atol=1e-10)
        np.testing.assert_allclose(trace["steps"][0]["loss"], np.sum(residual**2)/(2*len(batch)), atol=1e-10)
        counts[f"sgd_{horizon}"] = len(trace["steps"])
        path = run / "traces" / f"gru_{horizon}.json"
        gru_trace = json.loads(path.read_text())
        epochs = pd.read_csv(run / f"gru_{horizon}/epochs.csv")
        np.testing.assert_allclose([s["loss"] for s in gru_trace["steps"]], epochs.loss)
        np.testing.assert_allclose([s["calibration_loss"] for s in gru_trace["steps"]], epochs.val_loss)
        best = int(np.argmin(epochs.val_loss))
        import tensorflow as tf
        model = tf.keras.models.load_model(run / f"gru_{horizon}/best.keras", compile=False)
        np.testing.assert_allclose(gru_trace["steps"][best]["selected_weights"], model.trainable_weights[0].numpy().reshape(-1)[:2], atol=1e-7)
        scaler = joblib.load(run / f"gru_{horizon}/scaler.joblib")
        probe_ids = ids[:min(64, len(ids))]
        probe = scaler.transform(history[probe_ids].reshape(-1, 7)).reshape(len(probe_ids), 60, 7).astype(np.float32)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(probe, training=False)-(target[probe_ids]/100).astype(np.float32)))
        grads = tape.gradient(loss, model.trainable_weights)
        # Training probe target was float32; allow small CPU reduction roundoff.
        np.testing.assert_allclose(gru_trace["steps"][best]["probe_gradient_norm"], float(tf.linalg.global_norm(grads)), rtol=1e-4, atol=1e-6)
        gru_trace["selected_epoch"] = best+1
        save_json(path, gru_trace)
        counts[f"gru_{horizon}"] = len(gru_trace["steps"])
    protocol = json.loads((run / "protocol.json").read_text())
    protocol.update(sgd_algorithm="mini_batch_sgd_v1", sgd_epochs=20, mc_kernel="src.model_traces.simulate_mc")
    save_json(run / "protocol.json", protocol)
    result = {"status": "passed", "observations_verified": counts,
        "checks": ["All recorded SGD projection updates", "SGD final coefficients and actual first-batch gradient/loss",
            "GRU epoch CSV losses", "Best-checkpoint selected weights and probe gradient"]}
    save_json(run / "trace_audit.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
