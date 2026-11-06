"""Feasibility counts for spread forecasting on frozen development data only."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v3_research import FEATURES, candidate_ends, overlaps, save_json, valid_rows


def transition_labels(spreads, ends):
    """Use only the next ten observations as outcomes, never as inputs."""
    wide = np.asarray(spreads, dtype=float) * 100 > 10 + 1e-9
    ends = np.asarray(ends, dtype=int)
    if len(ends) and (ends.min() < 0 or ends.max() + 10 >= len(wide)):
        raise ValueError("Incomplete forecast horizon")
    total = np.r_[0, np.cumsum(wide)]
    future_wide = total[ends + 11] - total[ends + 1]
    return {"currently_wide": wide[ends],
            "widens_within_10s": ~wide[ends] & (future_wide > 0),
            "narrows_within_10s": wide[ends] & (future_wide < 10),
            "wide_at_10s": wide[ends + 10]}


def count_labels(labels):
    current = labels["currently_wide"]
    return {"windows": len(current), "currently_narrow": int((~current).sum()),
            **{name: int(values.sum()) for name, values in labels.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/spread_forecast_feasibility_20261104"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    source = args.source_run or Path(json.loads(Path("artifacts/model_lab_latest.json").read_text())["run_dir"])
    meta = pd.read_parquet(source / "windows.parquet")
    protection = json.loads((source / "protected_v2_period.json").read_text())
    reserved = set(protection["sessions"])
    session_splits = meta.groupby("session_id").split.first().to_dict()
    bounds = {name: (int(group.input_start.min()), int(group.target_end.max()))
              for name, group in meta.loc[meta.split.isin(["train", "val"])].groupby("split")}
    manifest = json.loads((source / "input_manifest.json").read_text())
    counts = {name: {kind: Counter() for kind in ("overlapping", "spaced")} for name in bounds}
    ignored = Counter()
    for number, item in enumerate(manifest["files"], 1):
        file_session = item["name"].removeprefix("features_").removesuffix(".csv")
        # Known holdout/reserved session outcomes are never constructed.
        if session_splits.get(file_session) == "holdout" or file_session in reserved:
            ignored["reserved_or_holdout_files"] += 1
            continue
        frame = pd.read_csv(source / "snapshot" / item["name"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        for name in FEATURES:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.loc[valid_rows(frame)].drop_duplicates(["session_id", "ticker", "timestamp"])
        for (session, ticker), group in frame.groupby(["session_id", "ticker"], sort=False):
            if str(session) in reserved or session_splits.get(str(session)) == "holdout":
                continue
            group = group.sort_values("timestamp")
            times = group.timestamp.array.as_unit("ns").asi8
            ends = candidate_ends(times)
            ends = ends[~overlaps(times[ends - 59], times[ends + 10], protection)]
            spreads = group.spread.to_numpy(dtype=float)
            for split, (first, last) in bounds.items():
                assigned = session_splits.get(str(session))
                if assigned is not None and assigned != split:
                    continue
                # New wide-only sessions are included only when their entire
                # observed range lies inside an established development period.
                if assigned is None and (times[0] < first or times[-1] > last):
                    continue
                selected = ends[(times[ends - 59] >= first) & (times[ends + 10] <= last)]
                counts[split]["overlapping"].update(count_labels(transition_labels(spreads, selected)))
                spaced, next_time = [], -1
                for end in selected:
                    if times[end - 59] >= next_time:
                        spaced.append(end)
                        next_time = times[end + 10] + 1
                counts[split]["spaced"].update(count_labels(transition_labels(spreads, spaced)))
        if number % 200 == 0:
            print(f"Processed {number}/{len(manifest['files'])} frozen files", flush=True)
    result = {"source": str(source), "horizon_seconds": 10, "wide_threshold_cents": 10,
        "counts": counts, "ignored": dict(ignored),
        "policy": "All structurally valid spreads; unchanged development boundaries; known holdout/reserved sessions excluded",
        "targets": {"widens_within_10s": "Currently <=10c, then >10c at any next-ten-second observation",
                    "narrows_within_10s": "Currently >10c, then <=10c at any next-ten-second observation"},
        "limits": "Feasibility counts only. Overlapping windows repeat events; spaced examples remain cross-market correlated. No predictive model trained."}
    args.output.mkdir(parents=True)
    save_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
