import numpy as np
import pandas as pd
import pytest

from src.causal_swing_data import NS, validate_development_splits
from src.expanded_swing_source import split_sessions
from src.v3_research import overlaps


def test_additional_holdout_period_is_excluded():
    protected = {"start": 10, "end": 20, "additional_periods": [{"start": 40, "end": 50}]}
    assert overlaps(np.array([0, 15, 39, 51]), np.array([5, 25, 41, 60]), protected).tolist() == [False, True, True, False]


def test_long_session_cannot_cross_new_validation_boundary():
    meta = pd.DataFrame({"session_id": list("abcdefghij"),
        "input_start": np.arange(10)*1000*NS, "target_end": (np.arange(10)*1000+300)*NS})
    meta.loc[0, "target_end"] = 9000*NS
    result = split_sessions(meta)
    assert "a" not in result.session_id.tolist()
    protected = {"start": -20*NS, "end": -10*NS, "sessions": []}
    validate_development_splits(result, protected)
    protected["additional_periods"] = [{"start": 8100*NS, "end": 8200*NS}]
    with pytest.raises(ValueError, match="Reserved"):
        validate_development_splits(result, protected)
