import pandas as pd
import pytest

from src.audit_swing_returns import verify_trade_keys


def test_empty_trade_table_accepts_parquet_dtype_difference():
    actual = pd.DataFrame(columns=["session_id", "ticker", "input_end"])
    expected = actual.astype({"session_id": "string", "ticker": "string", "input_end": "int64"})
    verify_trade_keys(actual, expected)


def test_trade_identity_mismatch_still_fails():
    actual = pd.DataFrame({"session_id": ["s"], "ticker": ["T"], "input_end": [123]})
    expected = actual.assign(input_end=124)
    with pytest.raises(AssertionError):
        verify_trade_keys(actual, expected)
