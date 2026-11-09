"""Versioned, forecast-time-only microstructure features. No fitting or imputation."""

import numpy as np

from src.v3_research import FEATURES


SCHEMA_VERSION = "causal_microstructure_v1"
CHANGE_COLUMNS = (0, 1, 2, 4, 5)
SEQUENCE_NAMES = [*FEATURES, "seconds_to_close", "log1p_seconds_to_close",
    "delta_bid", "delta_ask", "delta_spread", "delta_bid_depth", "delta_ask_depth",
    "log1p_bid_depth", "log1p_ask_depth",
    *[f"quote_change_frequency_{width}" for width in (10, 30, 60)],
    *[f"depth_change_frequency_{width}" for width in (10, 30, 60)]]
STATISTICS = ("last", "mean", "std", "min", "max", "change")


def _history(values, length=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 7 or values.shape[1] not in (60, 180):
        raise ValueError("Expected N x 60/180 x 7 observed book history")
    if length is not None and values.shape[1] != length:
        raise ValueError(f"Expected exactly {length} observations")
    if not np.isfinite(values).all():
        raise ValueError("History cannot contain missing or nonfinite values")
    bid, ask, spread, mid, bd, ad, imbalance = np.moveaxis(values, -1, 0)
    depth = bd+ad
    expected = np.divide(bd-ad, depth, out=np.zeros_like(depth), where=depth > 0)
    valid = ((0 <= bid) & (bid < ask) & (ask <= 1) & (spread > 0) & (bd >= 0) & (ad >= 0)
        & (np.abs(imbalance) <= 1+1e-6)
        & (np.abs(spread-(ask-bid)) <= 1e-6) & (np.abs(mid-(ask+bid)/2) <= 1e-6)
        & (np.abs(imbalance-expected) <= 1e-5))
    if not valid.all():
        raise ValueError("History contains an invalid quote/depth/imbalance")
    return values


def verified_timing(timestamps_ns, scheduled_close_ns, observed_ns, stored_seconds):
    """Validate each original timing observation; never backfill an earlier row."""
    stamps, close, known = [np.asarray(value) for value in (timestamps_ns, scheduled_close_ns, observed_ns)]
    if any(value.dtype.kind not in "iu" for value in (stamps, close, known)):
        raise ValueError("Use integer UTC nanoseconds to preserve exact knowledge ordering")
    if any((value > np.iinfo(np.int64).max).any() for value in (stamps,close,known)):
        raise ValueError("Timing timestamps exceed signed nanosecond range")
    stamps, close, known = [value.astype(np.int64,copy=False) for value in (stamps,close,known)]
    seconds = np.asarray(stored_seconds, dtype=np.float64)
    if not (stamps.shape == close.shape == known.shape == seconds.shape):
        raise ValueError("Timing arrays must have identical shapes")
    if not all(np.isfinite(value).all() for value in (stamps, close, known, seconds)):
        raise ValueError("Timing metadata must be observed and finite on every row")
    # NaT encoded as int64.min must not become a plausible finite timestamp.
    if any((value <= 0).any() for value in (stamps, close, known)):
        raise ValueError("Timing timestamps must be valid positive UTC nanoseconds")
    if (known > stamps).any():
        raise ValueError("Schedule was not known at observation time")
    calculated = (close-stamps)/1e9
    if (seconds < 0).any() or (np.abs(calculated-seconds) > .001).any():
        raise ValueError("Invalid countdown or recorded schedule mismatch")
    return seconds.copy()


def context_indices(timestamps_ns, session_ids, ends, length=180, valid=None):
    """Past indices and eligibility only; no future quote/label inspection."""
    if length not in (60, 180):
        raise ValueError("Context length must be 60 or 180")
    times = np.asarray(timestamps_ns, dtype=np.int64)
    sessions = np.asarray(session_ids)
    ends = np.asarray(ends, dtype=np.int64)
    validity = np.ones(len(times), bool) if valid is None else np.asarray(valid, bool)
    if times.ndim != 1 or sessions.shape != times.shape or validity.shape != times.shape or ends.ndim != 1:
        raise ValueError("Context input arrays have incompatible shapes")
    indices = ends[:, None]+np.arange(-length+1, 1)
    good = (ends >= length-1) & (ends < len(times))
    if not len(times):
        return np.full(indices.shape, -1, dtype=int), np.zeros(len(ends), bool)
    safe = np.clip(indices, 0, len(times)-1)
    clocks = times[safe]
    intervals = np.diff(clocks, axis=1)/1e9
    good &= validity[safe].all(axis=1)
    valid_sessions=np.array([isinstance(s,(str,np.str_)) and bool(s.strip()) for s in sessions])
    good &= valid_sessions[safe].all(axis=1)
    good &= (sessions[safe] == sessions[safe[:, -1]][:, None]).all(axis=1)
    good &= ((intervals >= .5) & (intervals <= 1.5)).all(axis=1)
    good &= np.abs((clocks[:, -1]-clocks[:, 0])/1e9-(length-1)) <= 1.5
    return np.where(good[:, None], indices, -1), good


def _frequency(changes, width):
    cumulative = np.c_[np.zeros(len(changes)), np.cumsum(changes, axis=1)]
    # A width-observation lookback has at most width-1 observable intervals.
    end = np.arange(changes.shape[1])+1
    start = np.maximum(1, end-width+1)
    denominator = np.maximum(1, end-start)
    return (cumulative[:, end]-cumulative[:, start])/denominator


def engineered_sequence(history, seconds_to_close):
    values = _history(history)
    timing = np.asarray(seconds_to_close, dtype=np.float64)
    if timing.shape != values.shape[:2] or not np.isfinite(timing).all() or (timing < 0).any():
        raise ValueError("Supply verified nonnegative timing for every history row; no imputation")
    delta = np.concatenate([np.zeros_like(values[:, :1]), np.diff(values, axis=1)], axis=1)
    quote_change = np.any(np.abs(delta[:, :, :2]) > 1e-9, axis=2)
    depth_change = np.any(np.abs(delta[:, :, 4:6]) > 1e-9, axis=2)
    result = np.concatenate([values, timing[:, :, None], np.log1p(timing)[:, :, None],
        delta[:, :, CHANGE_COLUMNS], np.log1p(values[:, :, 4:6]),
        *[_frequency(quote_change, width)[:, :, None] for width in (10,30,60)],
        *[_frequency(depth_change, width)[:, :, None] for width in (10,30,60)]], axis=2)
    if result.shape[2] != len(SEQUENCE_NAMES) or not np.isfinite(result).all():
        raise ValueError("Engineered feature schema/finite-value mismatch")
    return result, SEQUENCE_NAMES.copy()


def feature_schema(context=False):
    names = [f"{name}_{stat}" for stat in STATISTICS for name in SEQUENCE_NAMES]
    names += ["midpoint_momentum_60obs", "midpoint_increment_std_60obs"]
    if context:
        names += ["midpoint_momentum_180obs", "midpoint_increment_std_180obs"]
    return {"version": SCHEMA_VERSION+("_context180" if context else ""),
        "dtype": "float64", "sequence_names": SEQUENCE_NAMES.copy(), "tabular_names": names,
        "sequence_feature_count": len(SEQUENCE_NAMES), "tabular_feature_count": len(names),
        "rules": ["Timing requires prior per-row knowledge verification; no retrospective schedules or imputation",
            "Differences at the first observed history row initialize to zero; no preceding observation is invented",
            "Trailing change frequencies use only available within-window intervals; tolerance 1e-9 dollars",
            "Depth transforms use log1p, allowing real zero depth",
            "60/180-observation momentum spans 59/179 observed intervals",
            "180 context must be same-session/continuous and end in the exact supplied 60-row history"]}


def build_features(history, seconds_to_close, context180=None):
    values = _history(history, 60)
    sequence, _ = engineered_sequence(values, seconds_to_close)
    blocks = [sequence[:, -1], sequence.mean(axis=1), sequence.std(axis=1),
              sequence.min(axis=1), sequence.max(axis=1), sequence[:, -1]-sequence[:, 0]]
    midpoint = values[:, :, 3]
    blocks += [(midpoint[:, -1]-midpoint[:, 0])[:, None], np.diff(midpoint,axis=1).std(axis=1)[:, None]]
    if context180 is not None:
        context = _history(context180, 180)
        if len(context) != len(values) or not np.array_equal(context[:, -60:], values):
            raise ValueError("Long context must end in the exact matching baseline history")
        mid = context[:, :, 3]
        blocks += [(mid[:, -1]-mid[:, 0])[:, None], np.diff(mid,axis=1).std(axis=1)[:, None]]
    result = np.concatenate(blocks,axis=1)
    schema = feature_schema(context180 is not None)
    if result.shape[1] != schema["tabular_feature_count"] or not np.isfinite(result).all():
        raise ValueError("Tabular schema/finite-value mismatch")
    return result, schema["tabular_names"]
