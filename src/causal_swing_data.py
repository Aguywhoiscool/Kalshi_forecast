"""Past-only entry cohorts with explicitly missing future observations."""

from collections import Counter
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from src.model_benchmark import summarize, validate_splits
from src.v3_research import FEATURES, digest, overlaps, save_json, valid_rows


NS = 10**9
HORIZONS = (30, 60)


def history_ends(times, valid):
    """Appending, removing or changing future quotes cannot alter old entries."""
    times, valid = np.asarray(times, dtype=np.int64), np.asarray(valid, dtype=bool)
    if len(times) != len(valid):
        raise ValueError("Timestamp/validity lengths differ")
    if len(times) < 60:
        return np.empty(0, dtype=int)
    ends = np.arange(59, len(times))
    starts = ends - 59
    bad_rows = np.r_[0, np.cumsum(~valid)]
    gaps = np.diff(times) / NS
    bad_gaps = np.r_[0, np.cumsum((gaps < .5) | (gaps > 1.5))]
    keep = (bad_rows[ends + 1] == bad_rows[starts]) & (bad_gaps[ends] == bad_gaps[starts])
    keep &= np.abs((times[ends] - times[starts]) / NS - 59) <= 1.5
    return ends[keep]


def align_future(times, quotes, valid, signals, max_horizons):
    """Nearest observed sample within 0.5s; no interpolation or carrying books."""
    times, signals = np.asarray(times, dtype=np.int64), np.asarray(signals, dtype=np.int64)
    desired = signals[:, None] + np.arange(61) * NS
    right = np.searchsorted(times, desired)
    previous = np.clip(right - 1, 0, len(times) - 1)
    following = np.clip(right, 0, len(times) - 1)
    left_distance = np.abs(desired - times[previous])
    right_distance = np.abs(desired - times[following])
    indices = np.where(left_distance <= right_distance, previous, following)
    close = (np.minimum(left_distance, right_distance) <= NS // 2)
    close &= np.arange(61)[None] <= np.asarray(max_horizons)[:, None]
    observed = np.where(close, times[indices], -1)
    available = close & np.asarray(valid, dtype=bool)[indices]
    values = np.where(available[:, :, None], np.asarray(quotes)[indices], np.nan)
    return values, observed, available


def source_groups(source):
    """Yield frozen development groups only, without testing future coverage."""
    original = pd.read_parquet(source / "windows.parquet")
    protected = json.loads((source / "protected_v2_period.json").read_text())
    if (source / "development_source.json").exists():
        validate_development_splits(original, protected)
    else:
        validate_splits(original, protected)
    assignment = original.groupby("session_id").split.first().to_dict()
    bounds = {name: (int(group.input_start.min()), int(group.target_end.max()))
              for name, group in original.groupby("split") if name in ("train", "val")}
    manifest = json.loads((source / "input_manifest.json").read_text())
    reserved = set(protected["sessions"])
    for item in manifest["files"]:
        session = item["name"].removeprefix("features_").removesuffix(".csv")
        if session in reserved or assignment.get(session) == "holdout":
            continue
        split = assignment.get(session)
        if split is None:
            if (source / "development_source.json").exists():
                continue  # Explicitly purged sessions stay excluded.
            started = pd.to_datetime(session, format="%Y-%m-%d_%H-%M-%S_%f", utc=True).value
            if started <= bounds["train"][1]:
                split = "train"
            elif bounds["val"][0] <= started <= bounds["val"][1]:
                split = "val"
            else:
                continue
        path = source / "snapshot" / item["name"]
        if digest(path) != item["sha256"]:
            raise ValueError("Frozen CSV hash mismatch: " + item["name"])
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        for name in FEATURES:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
        for (actual_session, ticker), group in frame.groupby(["session_id", "ticker"], sort=False):
            if str(actual_session) != session:
                raise ValueError("Source filename/session mismatch")
            group = group.sort_values("timestamp").reset_index(drop=True)
            yield item, split, ticker, group, bounds[split], protected


def validate_development_splits(meta, protected):
    if set(meta.split.unique()) != {"train", "val"}:
        raise ValueError("Development source must contain train and val only")
    if meta.groupby("session_id").split.nunique().max() > 1:
        raise ValueError("Session crosses development split")
    if meta.session_id.isin(protected["sessions"]).any() or overlaps(
            meta.input_start.to_numpy(), meta.target_end.to_numpy(), protected).any():
        raise ValueError("Reserved development source overlap")
    if meta.loc[meta.split == "val", "input_start"].min() < (
            meta.loc[meta.split == "train", "target_end"].max() + 70 * NS):
        raise ValueError("Development embargo violated")


def entry_metadata(session, ticker, split, times, validity, bounds, protected):
    ends = history_ends(times, validity)
    first, last = bounds
    keep = times[ends - 59] >= first
    eligibility = {}
    for horizon in HORIZONS:
        # A fixed administrative boundary is known before the experiment. It
        # protects reserved periods without consulting observed future prices.
        finish = times[ends] + horizon * NS + NS // 2
        eligibility[horizon] = (finish <= last) & ~overlaps(times[ends - 59], finish, protected)
    keep &= eligibility[30] | eligibility[60]
    selected, next_signal = [], -1
    for position in np.flatnonzero(keep):
        if times[ends[position]] >= next_signal:
            selected.append(position)
            next_signal = times[ends[position]] + 120 * NS
    positions = np.asarray(selected, dtype=int)
    chosen = ends[positions]
    maximum = np.where(eligibility[60][positions], 60, 30)
    meta = pd.DataFrame({"session_id": session, "ticker": ticker, "split": split,
        "input_start": times[chosen - 59], "input_end": times[chosen],
        "eligible_30": eligibility[30][positions], "eligible_60": eligibility[60][positions],
        "forecast_max_horizon": maximum,
        "target_end": times[chosen] + maximum * NS + NS // 2})
    return chosen, meta


def prepare_causal(source, output, progress):
    pieces, history_parts, future_parts, clock_parts, valid_parts = [], [], [], [], []
    provenance, coverage = {}, Counter()
    for number, (item, split, ticker, group, bounds, protected) in enumerate(source_groups(source), 1):
        times = group.timestamp.array.as_unit("ns").asi8
        validity = valid_rows(group)
        session = item["name"].removeprefix("features_").removesuffix(".csv")
        ends, meta = entry_metadata(session, ticker, split, times, validity, bounds, protected)
        if len(meta):
            provenance[item["name"]] = item["sha256"]
            history = group[FEATURES].to_numpy(dtype=float)[ends[:, None] + np.arange(-59, 1)]
            future, clock, known = align_future(times,
                group[["bid", "ask", "bid_depth", "ask_depth"]].to_numpy(dtype=float),
                validity, times[ends], meta.forecast_max_horizon.to_numpy())
            pieces.append(meta)
            history_parts.append(history)
            future_parts.append(future)
            clock_parts.append(clock)
            valid_parts.append(known)
            coverage["past_only_entries"] += len(meta)
            coverage["missing_entry_quotes"] += int((~known[:, 1]).sum())
            for horizon in HORIZONS:
                eligible = meta[f"eligible_{horizon}"].to_numpy()
                coverage[f"eligible_{horizon}"] += int(eligible.sum())
                coverage[f"missing_exit_quotes_{horizon}"] += int((eligible & ~known[:, horizon]).sum())
        if number % 500 == 0:
            progress("causal_dataset", groups=number, entries=coverage["past_only_entries"])
    if not pieces:
        raise ValueError("No past-only development entries")
    meta = pd.concat(pieces, ignore_index=True)
    order = meta.sort_values(["input_end", "ticker"]).index.to_numpy()
    meta = meta.iloc[order].reset_index(drop=True)
    histories = np.concatenate(history_parts)[order]
    future = np.concatenate(future_parts)[order]
    clocks = np.concatenate(clock_parts)[order]
    known = np.concatenate(valid_parts)[order]
    del history_parts, future_parts, clock_parts, valid_parts
    np.save(output / "history.npy", histories)
    np.save(output / "future_quotes.npy", future)
    np.save(output / "future_times.npy", clocks)
    np.save(output / "future_valid.npy", known)
    features = summarize(histories)
    np.save(output / "summary_features.npy", features)
    meta.to_parquet(output / "sample_metadata.parquet", index=False)
    save_json(output / "source_files.json", [{"file": name, "sha256": value} for name,value in provenance.items()])
    save_json(output / "coverage.json", {"utc": datetime.now(timezone.utc).isoformat(), **coverage})
    return meta, histories, future, clocks, known, features
