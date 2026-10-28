
import math

from src.activity_market_ranker import (
    calculate_activity_score,
    select_active_markets,
)


def observation(
    ticker,
    meaningful=0,
    changes=0,
    deltas=0,
    spread=1.0,
    valid=True,
):
    return {
        "ticker": ticker,
        "duration_seconds": 120,
        "delta_count": deltas,
        "midpoint_changes": changes,
        "meaningful_changes": meaningful,
        "spread_cents": spread,
        "valid_book": valid,
        "valid_coverage": 1.0 if valid else 0.0,
    }


def market(ticker, event):
    return {
        "ticker": ticker,
        "event_ticker": event,
    }


def test_meaningful_movement_improves_score():
    quiet = observation(
        "QUIET", deltas=10, spread=2
    )

    moving = observation(
        "MOVING",
        meaningful=10,
        changes=10,
        deltas=10,
        spread=2,
    )

    assert (
        calculate_activity_score(moving)
        > calculate_activity_score(quiet)
    )


def test_wider_spread_reduces_score():
    narrow = observation(
        "NARROW",
        meaningful=5,
        changes=5,
        spread=1,
    )

    wide = observation(
        "WIDE",
        meaningful=5,
        changes=5,
        spread=10,
    )

    assert (
        calculate_activity_score(narrow)
        > calculate_activity_score(wide)
    )


def test_invalid_book_is_rejected():
    invalid = observation("BAD", valid=False)

    assert calculate_activity_score(invalid) == -math.inf


def test_activity_beats_deltas_alone():
    busy_but_static = observation(
        "STATIC",
        deltas=500,
        changes=0,
        meaningful=0,
        spread=2,
    )

    actually_moving = observation(
        "MOVING",
        deltas=50,
        changes=20,
        meaningful=10,
        spread=2,
    )

    assert (
        calculate_activity_score(actually_moving)
        > calculate_activity_score(busy_but_static)
    )


def test_selects_most_active_markets():
    candidates = [
        market("SERIESA-EVENT1-A", "SERIESA-EVENT1"),
        market("SERIESB-EVENT2-A", "SERIESB-EVENT2"),
        market("SERIESC-EVENT3-A", "SERIESC-EVENT3"),
    ]

    observations = {
        candidates[0]["ticker"]: observation(
            candidates[0]["ticker"],
            meaningful=10,
            changes=10,
        ),
        candidates[1]["ticker"]: observation(
            candidates[1]["ticker"],
            meaningful=5,
            changes=5,
        ),
        candidates[2]["ticker"]: observation(
            candidates[2]["ticker"],
            meaningful=0,
            changes=0,
        ),
    }

    selected = select_active_markets(
        candidates,
        observations,
        target_markets=2,
    )

    assert [m["ticker"] for m in selected] == [
        "SERIESA-EVENT1-A",
        "SERIESB-EVENT2-A",
    ]


def test_preserves_event_diversity():
    candidates = [
        market("SERIESA-EVENT1-A", "SERIESA-EVENT1"),
        market("SERIESA-EVENT1-B", "SERIESA-EVENT1"),
        market("SERIESB-EVENT2-A", "SERIESB-EVENT2"),
    ]

    observations = {
        m["ticker"]: observation(
            m["ticker"],
            meaningful=10,
            changes=10,
        )
        for m in candidates
    }

    selected = select_active_markets(
        candidates,
        observations,
        target_markets=3,
    )

    assert len(selected) == 2

    events = [
        m["event_ticker"] for m in selected
    ]

    assert len(events) == len(set(events))

def test_rejects_markets_without_midpoint_movement():
    candidates = [
        market("SERIESA-EVENT1-A", "SERIESA-EVENT1"),
        market("SERIESB-EVENT2-A", "SERIESB-EVENT2"),
    ]

    observations = {
        "SERIESA-EVENT1-A": observation(
            "SERIESA-EVENT1-A",
            deltas=100,
            changes=0,
            meaningful=0,
        ),
        "SERIESB-EVENT2-A": observation(
            "SERIESB-EVENT2-A",
            deltas=20,
            changes=3,
            meaningful=2,
        ),
    }

    selected = select_active_markets(
        candidates,
        observations,
        target_markets=2,
    )

    assert [m["ticker"] for m in selected] == [
        "SERIESB-EVENT2-A"
    ]
def test_rejects_low_book_coverage():
    low_coverage = observation(
        "LOW",
        meaningful=10,
        changes=10,
        deltas=100,
    )
    low_coverage["valid_coverage"] = 0.50

    high_coverage = observation(
        "HIGH",
        meaningful=5,
        changes=5,
        deltas=50,
    )
    high_coverage["valid_coverage"] = 0.99

    assert calculate_activity_score(low_coverage) == -math.inf
    assert calculate_activity_score(high_coverage) > -math.inf