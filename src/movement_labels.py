"""Numerically stable movement labels, shared by all current research methods."""

import numpy as np


LABEL_POLICY = "absolute_0.5_cent_with_1e-9_cent_roundoff_tolerance_v1"
MOVEMENT_THRESHOLD_CENTS = .5
ROUNDOFF_TOLERANCE_CENTS = 1e-9


def meaningful_movement(change_cents):
    """Include exact threshold moves despite floating-point subtraction error.

    Tolerance is in cents (1e-11 dollars), not a change in the economic target.
    Labels must be computed before downcasting changes to float32.
    """
    values = np.asarray(change_cents, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite movement target")
    return np.abs(values) >= MOVEMENT_THRESHOLD_CENTS - ROUNDOFF_TOLERANCE_CENTS
