import json
import numpy as np
import pandas as pd
import pytest
from src.accuracy_experiment import protocol, crossvalidate, GRIDS


def test_predeclared_protocol_is_immutable_on_restart(tmp_path, monkeypatch):
    first = protocol(tmp_path)
    assert protocol(tmp_path) == first
    monkeypatch.setitem(GRIDS, "ridge", [{"alpha": 999.}])
    with pytest.raises(ValueError, match="Frozen protocol"):
        protocol(tmp_path)


def test_real_fold_fit_calibration_prediction_and_cache_replay(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    (tmp_path / "agent_status").mkdir()
    protocol(output)
    class Data:
        def __init__(self):
            self.meta = pd.DataFrame(index=range(24))
            self.future = np.tile([.4, .42, 2., 2.], (24, 61, 1))
            self.future[:, 30, :2] += np.arange(24)[:, None]*.001
            self.known = np.ones((24, 61), bool)
        def table(self, family):
            assert family == "seven60"
            return np.column_stack([np.arange(24), np.sin(np.arange(24))])
    data = Data()
    folds = [{"fit": np.arange(stop), "calibration": np.arange(stop, stop+2),
              "validation": np.arange(stop+2, stop+4)} for stop in (8, 12, 16)]
    result = crossvalidate(output, data, folds, "ridge", "seven60", "conservative", {"alpha":1.}, 30, "real_ridge")
    assert len(result["scores"]) == 3 and np.isfinite(result["mean_observed_mae"])
    assert all((output / "search/real_ridge" / f"fold{i}/model.joblib").exists() for i in range(3))
    repeated = crossvalidate(output, data, folds, "ridge", "seven60", "conservative", {"alpha":1.}, 30, "real_ridge")
    assert repeated == result
    with pytest.raises(ValueError, match="Cached search"):
        crossvalidate(output, data, folds, "ridge", "seven60", "conservative", {"alpha":100.}, 30, "real_ridge")
