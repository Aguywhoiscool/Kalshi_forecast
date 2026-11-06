"""Verify derived research arrays against immutable source recordings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_benchmark import summarize
from src.v3_research import FEATURES, digest, valid_rows


def equal(actual, expected, label, float32_summary=False):
    matches = (np.allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True)
               if float32_summary else np.array_equal(actual, expected, equal_nan=True))
    if not matches:
        raise ValueError("Source reconstruction mismatch: " + label)


def verify_derived_inputs(run):
    protocol = json.loads((run / "protocol.json").read_text())
    source = Path(protocol["source"])
    meta = pd.read_parquet(run / "sample_metadata.parquet")
    if digest(run / "sample_metadata.parquet") != protocol["sample_metadata_sha256"]:
        raise ValueError("Sample metadata hash changed")
    manifest = json.loads((source / "input_manifest.json").read_text())
    source_hashes = {item["name"]: item["sha256"] for item in manifest["files"]}
    arrays = {name: np.load(run / filename, mmap_mode="r") for name, filename in {
        "history": "history.npy", "future_quotes": "future_quotes.npy",
        "future_times": "future_times.npy", "summary_features": "summary_features.npy"}.items()}
    shapes = {"history": (len(meta), 60, 7), "future_quotes": (len(meta), 61, 4),
              "future_times": (len(meta), 61), "summary_features": (len(meta), 50)}
    for name, array in arrays.items():
        if array.shape != shapes[name]:
            raise ValueError("Invalid derived array shape: " + name)
    causal = protocol.get("dataset_policy") == "past_only_v1"
    if causal:
        availability = np.load(run / "future_valid.npy", mmap_mode="r")
        if availability.shape != (len(meta), 61):
            raise ValueError("Invalid future availability shape")
    verified = 0
    for session, selected in meta.groupby("session_id", sort=False):
        name = f"features_{session}.csv"
        path = source / "snapshot" / name
        if name not in source_hashes or digest(path) != source_hashes[name]:
            raise ValueError("Frozen source hash mismatch: " + name)
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not causal:
            frame = frame.loc[valid_rows(frame)]
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
        for ticker, rows in selected.groupby("ticker", sort=False):
            group = frame.loc[(frame.session_id.astype(str) == str(session)) & (frame.ticker == ticker)].sort_values("timestamp")
            times = group.timestamp.array.as_unit("ns").asi8
            ends = np.searchsorted(times, rows.input_end.to_numpy())
            if (ends < 59).any() or (ends >= len(group)).any():
                raise ValueError("Source history bounds invalid")
            equal(times[ends], rows.input_end.to_numpy(), "input_end")
            equal(times[ends - 59], rows.input_start.to_numpy(), "input_start")
            history = group[FEATURES].to_numpy(dtype=float)[ends[:, None] + np.arange(-59, 1)]
            equal(arrays["history"][rows.index], history, "history.npy")
            equal(arrays["summary_features"][rows.index], summarize(history), "summary_features.npy", float32_summary=True)
            raw_quotes = group[["bid", "ask", "bid_depth", "ask_depth"]].to_numpy(dtype=float)
            if not causal:
                if (ends + 60 >= len(group)).any():
                    raise ValueError("Legacy complete-horizon bounds invalid")
                indices = ends[:, None] + np.arange(61)
                expected_times = times[indices]
                expected_quotes = raw_quotes[indices]
                equal(expected_times[:, 60], rows.target_end.to_numpy(), "target_end")
            else:
                # Independent nearest-time reconstruction: no interpolation or
                # carrying quotes across recording/session boundaries.
                desired = rows.input_end.to_numpy()[:, None] + np.arange(61) * 10**9
                right = np.searchsorted(times, desired)
                left = np.maximum(0, right - 1)
                right = np.minimum(len(times) - 1, right)
                choose_left = np.abs(times[left] - desired) <= np.abs(times[right] - desired)
                indices = np.where(choose_left, left, right)
                allowed = rows.target_end.to_numpy()[:, None] >= desired
                close = (np.abs(times[indices] - desired) <= 500_000_000) & allowed
                valid_quote = valid_rows(group)[indices] & close
                expected_times = np.where(close, times[indices], -1)
                expected_quotes = np.where(valid_quote[:, :, None], raw_quotes[indices], np.nan)
                equal(availability[rows.index], valid_quote, "future_valid.npy")
            equal(arrays["future_times"][rows.index], expected_times, "future_times.npy")
            equal(arrays["future_quotes"][rows.index], expected_quotes, "future_quotes.npy")
            verified += len(rows)
    return {"source_reconstructed_examples": verified,
        "derived_arrays_verified": [*arrays, *( ["future_valid"] if causal else [])]}


def verify_exit_times(trades, future_times, rows):
    indices = trades.exit_index.to_numpy(dtype=int)
    if ((indices < 0) | (indices >= future_times.shape[1])).any():
        raise ValueError("Invalid exit index")
    equal(trades.exit_time_ns.to_numpy(), future_times[rows, indices], "trade exit timestamps")
