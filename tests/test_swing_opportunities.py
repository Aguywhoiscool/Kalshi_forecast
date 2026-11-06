import numpy as np

from src.analyze_swing_opportunities import common_ends, swing_outcomes


def test_common_horizon_rejects_gaps_and_short_histories():
    assert not len(common_ends(np.arange(119) * 10**9))
    times = np.arange(160, dtype=np.int64) * 10**9
    assert np.array_equal(common_ends(times), np.arange(59, 100))
    times[80:] += 2 * 10**9
    assert not len(common_ends(times))


def test_wide_spread_big_midpoint_swing_may_still_lose_money():
    quotes = np.tile([.3, .5, .4], (12, 1))
    quotes[10] = [.4, .6, .5]  # Midpoint +10c; executable YES loses 10c.
    outcome = swing_outcomes(quotes, [0], 10)
    assert np.isclose(outcome["midpoint_change_cents"][0], 10)
    assert np.isclose(outcome["yes_fixed_gross"][0], -10)
    assert np.isclose(outcome["no_fixed_gross"][0], -30)


def test_peak_exit_is_separate_from_fixed_exit_and_no_complements_are_correct():
    quotes = np.tile([.4, .42, .41], (12, 1))
    quotes[5] = [.55, .57, .56]
    outcome = swing_outcomes(quotes, [0], 10)
    assert np.isclose(outcome["hindsight_direction_and_exit_gross"][0], 13)
    assert np.isclose(outcome["yes_fixed_gross"][0], -2)
    quotes[10] = [.30, .32, .31]
    assert np.isclose(swing_outcomes(quotes, [0], 10)["no_fixed_gross"][0], 8)
