
import asyncio

import pytest

import src.observation_coordinator as coordinator_module
from src.observation_coordinator import ObservationCoordinator


@pytest.mark.asyncio
async def test_coordinator_refresh(monkeypatch, tmp_path):
    candidate = {
        "ticker": "TEST-EVENT-YES",
        "event_ticker": "TEST-EVENT",
    }

    def fake_rank_active_markets(
        max_markets,
        excluded_tickers=None,
    ):
        assert excluded_tickers == {"ALREADY-MONITORED"}
        return [candidate]

    async def fake_observe_markets(tickers, duration_seconds):
        return {
            tickers[0]: {
                "ticker": tickers[0],
                "duration_seconds": duration_seconds,
                "delta_count": 10,
                "midpoint_changes": 3,
                "meaningful_changes": 3,
                "spread_cents": 1.0,
                "valid_book": True,
                "valid_coverage": 1.0,
            }
        }

    monkeypatch.setattr(
        coordinator_module,
        "rank_active_markets",
        fake_rank_active_markets,
    )

    monkeypatch.setattr(
        coordinator_module,
        "observe_markets",
        fake_observe_markets,
    )

    manager = ObservationCoordinator(
        candidate_count=1,
        batch_size=1,
        observation_seconds=10,
    )

    from src.market_observation_cache import MarketObservationCache
    from src.market_probe_batch import MarketProbeBatch

    manager.cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )
    manager.probe = MarketProbeBatch(
        path=tmp_path / "cursor.json"
    )

    assert manager.start(
        excluded_tickers={"ALREADY-MONITORED"}
    ) is True
    assert manager.start() is False

    await manager.task
    manager.check()

    assert manager.last_error is None
    assert manager.get_candidates() == [candidate]
    assert manager.cache.get(candidate["ticker"]) is not None


@pytest.mark.asyncio
async def test_coordinator_failure_does_not_advance(
    monkeypatch, tmp_path
):
    candidate = {
        "ticker": "TEST-EVENT-YES",
        "event_ticker": "TEST-EVENT",
    }

    def fake_rank_active_markets(
        max_markets,
        excluded_tickers=None,
    ):
        return [candidate]

    async def failing_observe(tickers, duration_seconds):
        raise RuntimeError("Simulated connection failure")

    monkeypatch.setattr(
        coordinator_module,
        "rank_active_markets",
        fake_rank_active_markets,
    )

    monkeypatch.setattr(
        coordinator_module,
        "observe_markets",
        failing_observe,
    )

    manager = ObservationCoordinator(
        candidate_count=1,
        batch_size=1,
        observation_seconds=10,
    )

    from src.market_observation_cache import MarketObservationCache
    from src.market_probe_batch import MarketProbeBatch

    manager.cache = MarketObservationCache(
        path=tmp_path / "cache.json"
    )
    manager.probe = MarketProbeBatch(
        path=tmp_path / "cursor.json"
    )

    assert manager.start() is True

    with pytest.raises(RuntimeError):
        await manager.task

    manager.check()

    assert manager.last_error == "Simulated connection failure"
    assert manager.probe.cursor == 0
    assert manager.cache.get(candidate["ticker"]) is None
