import numpy as np
import pytest

from src.analyze_spread_transitions import transition_labels


def test_exact_ten_cents_is_narrow_and_widening_uses_future_only():
    spreads = np.full(22, .10)
    spreads[0] = .30  # Past wide spread must not become a future event.
    labels = transition_labels(spreads, [1])
    assert not labels["currently_wide"][0]
    assert not labels["widens_within_10s"][0]
    spreads[11] = .11  # Last observation inside the ten-second horizon.
    assert transition_labels(spreads, [1])["widens_within_10s"][0]
    spreads[11], spreads[12] = .10, .11  # Just outside the horizon.
    assert not transition_labels(spreads, [1])["widens_within_10s"][0]


def test_narrowing_requires_current_wide_and_future_narrow():
    spreads = np.full(12, .2)
    assert not transition_labels(spreads, [0])["narrows_within_10s"][0]
    spreads[10] = .1
    labels = transition_labels(spreads, [0])
    assert labels["currently_wide"][0]
    assert labels["narrows_within_10s"][0]
    assert not labels["widens_within_10s"][0]
    with pytest.raises(ValueError):
        transition_labels(spreads, [2])
