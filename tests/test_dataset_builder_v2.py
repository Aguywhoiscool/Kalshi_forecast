
"""Integration tests for the actual Version 2 dataset builder."""

from datetime import datetime, timedelta, timezone
from unittest import result

import numpy as np
import pandas as pd
import pytest

from src import dataset_builder_v2 as builder


def make_feature_rows():
    """Create 70 valid one-second market observations."""
    start = datetime(2026, 10, 28, tzinfo=timezone.utc)

    rows = []

    for i in range(70):
        bid = 0.40 + (i % 5) * 0.01
        ask = bid + 0.02

        rows.append({
            "timestamp": (
                start + timedelta(seconds=i)
            ).isoformat(),
            "session_id": "TEST_SESSION",
            "ticker": "TEST_MARKET",
            "bid": bid,
            "ask": ask,
            "spread": ask - bid,
            "midpoint": (bid + ask) / 2,
            "bid_depth": 100,
            "ask_depth": 120,
            "imbalance": -20 / 220,
        })

    return rows


def run_builder(tmp_path, monkeypatch, rows):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"

    source_dir.mkdir()

    filepath = source_dir / "features_test.csv"
    pd.DataFrame(rows).to_csv(filepath, index=False)

    monkeypatch.setattr(builder, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(builder, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        builder,
        "OUTPUT_FILE",
        output_dir / "gru_dataset_combined.npz",
    )

    # The builder excludes files modified within the
    # last 10 minutes. Give this synthetic file an older
    # modification time.
    import os
    import time

    old_time = time.time() - 3600
    os.utime(filepath, (old_time, old_time))

    builder.build_combined_dataset()

    output = builder.OUTPUT_FILE

    if not output.exists():
        return None

    with np.load(output) as data:
       return {
            "X": data["X"].copy(),
            "y": data["y"].copy(),
            "y_regression": data["y_regression"].copy(),
        }


def test_complete_70_second_sequence(tmp_path, monkeypatch):
    rows = make_feature_rows()

    result = run_builder(tmp_path, monkeypatch, rows)

    assert result is not None
    assert result["X"].shape == (1, 60, 7)
    assert result["y"].shape == (1, 1)
    assert result["y_regression"].shape == (1,)
    assert np.isfinite(result["y_regression"]).all()


def test_wide_spread_breaks_sequence(tmp_path, monkeypatch):
    rows = make_feature_rows()

    # Insert an unacceptable quote within the history.
    rows[25]["ask"] = rows[25]["bid"] + 0.20
    rows[25]["spread"] = 0.20
    rows[25]["midpoint"] = (
        rows[25]["bid"] + rows[25]["ask"]
    ) / 2

    result = run_builder(tmp_path, monkeypatch, rows)

    assert result is None


def test_missing_second_breaks_sequence(tmp_path, monkeypatch):
    rows = make_feature_rows()

    # Remove one actual observation.
    del rows[30]

    result = run_builder(tmp_path, monkeypatch, rows)

    assert result is None


def test_incorrect_target_time_rejected(tmp_path, monkeypatch):
    rows = make_feature_rows()

    # Shift the forecast portion by one second.
    for row in rows[60:]:
        timestamp = datetime.fromisoformat(
            row["timestamp"]
        )
        row["timestamp"] = (
            timestamp + timedelta(seconds=1)
        ).isoformat()

    result = run_builder(tmp_path, monkeypatch, rows)

    assert result is None


def test_one_sided_quote_breaks_sequence(tmp_path, monkeypatch):
    rows = make_feature_rows()

    rows[20]["bid"] = None

    result = run_builder(tmp_path, monkeypatch, rows)

    assert result is None

def test_regression_label_is_in_cents(tmp_path, monkeypatch):
    rows = make_feature_rows()

    # Set the final input midpoint to 45 cents.
    rows[59]["bid"] = 0.44
    rows[59]["ask"] = 0.46
    rows[59]["spread"] = 0.02
    rows[59]["midpoint"] = 0.45

    # Set the 10-second future midpoint to 47 cents.
    rows[69]["bid"] = 0.46
    rows[69]["ask"] = 0.48
    rows[69]["spread"] = 0.02
    rows[69]["midpoint"] = 0.47

    result = run_builder(tmp_path, monkeypatch, rows)

    assert result is not None
    assert np.isclose(result["y_regression"][0], 2.0)
    assert result["y"][0, 0] == 2  # UP
