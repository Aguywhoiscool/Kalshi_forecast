import csv
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backfill_close_metadata import annotate, fetch_schedule
from src.feature_sampler import FeatureSampler
from src.orderbook import OrderBookManager


def test_retrospective_schedule_is_not_claimed_as_point_in_time():
    frame = pd.DataFrame({"ticker": ["T", "UNKNOWN"], "timestamp": ["2026-11-06T12:00:00Z"]*2})
    result = annotate(frame, {"T": {"close_time": "2026-11-06T12:30:00Z"}})
    assert result.retrospective_seconds_to_close.iloc[0] == 1800
    assert pd.isna(result.retrospective_seconds_to_close.iloc[1])
    assert not result.asof_verified.any()


def test_archive_fallback_discards_settlement_and_price_fields():
    calls = []
    class Response:
        def __init__(self, status):
            self.status_code = status
        def raise_for_status(self):
            pass
        def json(self):
            return {"market": {"ticker": "T", "close_time": "2026-11-06T12:00:00Z",
                "result": "yes", "last_price_dollars": "1", "settlement_value_dollars": "1"}}
    class Session:
        def get(self, endpoint, timeout):
            calls.append(endpoint)
            return Response(404 if len(calls) == 1 else 200)
    record = fetch_schedule(Session(), "T", "https://example.test")
    assert calls[-1].endswith("/historical/markets/T")
    assert record["asof_verified"] is False
    assert not {"result", "last_price_dollars", "settlement_value_dollars"} & record.keys()


def test_collector_records_only_observed_schedules_without_rewriting_old_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("src.feature_sampler.PROCESSED_DIR", tmp_path)
    sampler = FeatureSampler()
    manager = OrderBookManager()
    manager.process({"type": "orderbook_snapshot", "sid": 1, "seq": 1, "msg": {
        "market_ticker": "T", "yes_dollars_fp": [[".4", "2"]], "no_dollars_fp": [[".58", "3"]]}})
    sampler.record(manager, ["T"])
    close = datetime.now(timezone.utc)+timedelta(minutes=30)
    sampler.update_market_metadata("T", {"close_time": close.isoformat()})
    sampler.record(manager, ["T"])
    with sampler.filepath.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["scheduled_close_time"] == ""
    assert rows[1]["scheduled_close_time"] == close.isoformat()
    assert 1790 < float(rows[1]["seconds_to_scheduled_close"]) <= 1800
    assert rows[1]["close_time_observed_at"] <= rows[1]["timestamp"]
