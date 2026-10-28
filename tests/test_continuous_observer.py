
import asyncio

import pytest

from src.observation_coordinator import ObservationCoordinator


@pytest.mark.asyncio
async def test_continuous_observer_starts_once_and_stops(monkeypatch):
    coordinator = ObservationCoordinator()

    called = asyncio.Event()
    seen_exclusions = []

    async def fake_refresh(excluded_tickers=None):
        seen_exclusions.append(set(excluded_tickers or []))
        called.set()

    monkeypatch.setattr(
        coordinator,
        "refresh",
        fake_refresh,
    )

    assert coordinator.start_continuous(
        lambda: {"CURRENT-MARKET"}
    ) is True

    # A second call must not launch another observer.
    assert coordinator.start_continuous(
        lambda: {"OTHER-MARKET"}
    ) is False

    try:
        await asyncio.wait_for(called.wait(), timeout=2)

        assert seen_exclusions == [{"CURRENT-MARKET"}]
    finally:
        await coordinator.stop()

    assert coordinator.task is None
