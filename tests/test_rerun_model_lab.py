import json

import numpy as np
import pandas as pd
import pytest

from src.rerun_model_lab import clone_dataset
from src.v3_research import digest


def test_clone_preserves_exact_splits_and_copies_no_trained_state(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    (source / "snapshot").mkdir()
    (source / "snapshot/input.csv").write_text("timestamp,bid\n0,0.4\n")
    (source / "shards").mkdir()
    np.save(source / "shards/00000.npy", np.ones((70, 7)))
    meta = pd.DataFrame({"split": ["train", "val", "holdout"],
        "session_id": ["a", "b", "c"], "input_start": np.array([0, 200, 400]) * 10**9,
        "target_end": np.array([69, 269, 469]) * 10**9})
    meta.to_parquet(source / "windows.parquet", index=False)
    protected = {"start": 1000 * 10**9, "end": 1100 * 10**9, "sessions": []}
    (source / "protected_v2_period.json").write_text(json.dumps(protected))
    (source / "dataset_complete.json").write_text('{}')
    (source / "input_manifest.json").write_text(json.dumps({"files": [
        {"name": "input.csv", "sha256": digest(source / "snapshot/input.csv")}]}))
    (source / "feature_scaler.joblib").write_text("never copy")
    (source / "direction").mkdir()
    (source / "direction/best.keras").write_text("never copy")
    clone_dataset(source, destination)
    assert digest(source / "windows.parquet") == digest(destination / "windows.parquet")
    assert not (destination / "feature_scaler.joblib").exists()
    assert not (destination / "direction").exists()
    np.save(destination / "shards/00000.npy", np.zeros((70, 7)))
    assert np.load(source / "shards/00000.npy").sum() == 490
    with pytest.raises(FileExistsError):
        clone_dataset(source, destination)
