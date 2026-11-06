"""All-spread, development-only 10/30/60-second swing opportunity diagnostic.

Hindsight direction/exit maxima are explicitly upper bounds, never backtests.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v3_research import FEATURES, overlaps, save_json, valid_rows


HORIZONS = (10, 30, 60)


def common_ends(times):
    times = np.asarray(times, dtype=np.int64)
    if len(times) < 120:
        return np.empty(0, dtype=int)
    ends = np.arange(59, len(times) - 60)
    gaps = np.diff(times) / 1e9
    bad = np.r_[0, np.cumsum((gaps < .5) | (gaps > 1.5))]
    valid = bad[ends + 60] == bad[ends - 59]
    valid &= np.abs((times[ends] - times[ends - 59]) / 1e9 - 59) <= 1.5
    for horizon in HORIZONS:
        valid &= np.abs((times[ends + horizon] - times[ends]) / 1e9 - horizon) <= .5
    return ends[valid]


def swing_outcomes(quotes, ends, horizon):
    """Executable-side gross returns, prior to fees/slippage, for one contract."""
    quotes, ends = np.asarray(quotes, dtype=float), np.asarray(ends, dtype=int)
    bid, ask, mid = quotes.T
    yes = (bid[ends + horizon] - ask[ends]) * 100
    no = (bid[ends] - ask[ends + horizon]) * 100
    future = ends[:, None] + np.arange(1, horizon + 1)
    best_yes = (bid[future].max(axis=1) - ask[ends]) * 100
    best_no = (bid[ends] - ask[future].min(axis=1)) * 100
    return {"midpoint_change_cents": (mid[ends + horizon] - mid[ends]) * 100,
            "yes_fixed_gross": yes, "no_fixed_gross": no,
            "hindsight_direction_fixed_exit_gross": np.maximum(yes, no),
            "hindsight_direction_and_exit_gross": np.maximum(best_yes, best_no)}


def summarize(frame):
    rows = []
    for (split, horizon, bucket), group in frame.groupby(["split", "horizon_seconds", "entry_spread_bucket"], observed=True):
        fixed = group.hindsight_direction_fixed_exit_gross.to_numpy()
        best = group.hindsight_direction_and_exit_gross.to_numpy()
        absolute = group.midpoint_change_cents.abs().to_numpy()
        row = {"split": split, "horizon_seconds": int(horizon), "entry_spread_bucket": bucket,
            "examples": len(group), "markets": group.ticker.nunique(),
            "midpoint_move_ge_5c": int((absolute >= 5 - 1e-9).sum()),
            "midpoint_move_ge_10c": int((absolute >= 10 - 1e-9).sum()),
            "fixed_exit_hindsight_positive_gross": int((fixed > 1e-9).sum()),
            "fixed_exit_hindsight_mean_gross_cents": float(fixed.mean()),
            "best_exit_hindsight_positive_gross": int((best > 1e-9).sum()),
            "best_exit_hindsight_gross_ge_5c": int((best >= 5 - 1e-9).sum())}
        for cost in (2, 4):
            row[f"fixed_exit_hindsight_profit_above_{cost}c_cost_buffer"] = int((fixed > cost + 1e-9).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/swing_opportunities_20261104"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    pointer = json.loads(Path("artifacts/model_lab_latest.json").read_text())
    source = Path(pointer["run_dir"])
    metadata = pd.read_parquet(source / "windows.parquet")
    assignments = metadata.groupby("session_id").split.first().to_dict()
    bounds = {name: (int(group.input_start.min()), int(group.target_end.max()))
              for name, group in metadata.loc[metadata.split.isin(["train", "val"])].groupby("split")}
    protected = json.loads((source / "protected_v2_period.json").read_text())
    reserved = set(protected["sessions"])
    manifest = json.loads((source / "input_manifest.json").read_text())
    pieces = []
    for number, item in enumerate(manifest["files"], 1):
        file_session = item["name"].removeprefix("features_").removesuffix(".csv")
        if file_session in reserved or assignments.get(file_session) == "holdout":
            continue
        frame = pd.read_csv(source / "snapshot" / item["name"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[valid_rows(frame)].drop_duplicates(["session_id", "ticker", "timestamp"])
        for (session, ticker), group in frame.groupby(["session_id", "ticker"], sort=False):
            if str(session) in reserved or assignments.get(str(session)) == "holdout":
                continue
            group = group.sort_values("timestamp")
            times = group.timestamp.array.as_unit("ns").asi8
            ends = common_ends(times)
            ends = ends[~overlaps(times[ends - 59], times[ends + 60], protected)]
            quotes = group[["bid", "ask", "midpoint"]].to_numpy(dtype=float)
            for split, (first, last) in bounds.items():
                assigned = assignments.get(str(session))
                if assigned is not None and assigned != split:
                    continue
                if assigned is None and (times[0] < first or times[-1] > last):
                    continue
                eligible = ends[(times[ends - 59] >= first) & (times[ends + 60] <= last)]
                selected, next_time = [], -1
                for end in eligible:
                    if times[end - 59] >= next_time:
                        selected.append(end)
                        next_time = times[end + 60] + 1
                selected = np.asarray(selected, dtype=int)
                if not len(selected):
                    continue
                spread_cents = (quotes[selected, 1] - quotes[selected, 0]) * 100
                bucket = np.select([spread_cents <= 2 + 1e-9, spread_cents <= 5 + 1e-9,
                    spread_cents <= 10 + 1e-9], ["<=2c", "2-5c", "5-10c"], default=">10c")
                for horizon in HORIZONS:
                    part = pd.DataFrame(swing_outcomes(quotes, selected, horizon))
                    part = part.assign(split=split, session_id=str(session), ticker=str(ticker),
                        input_start=times[selected - 59], input_end=times[selected],
                        horizon_seconds=horizon, entry_spread_cents=spread_cents,
                        entry_spread_bucket=bucket)
                    pieces.append(part)
        if number % 200 == 0:
            print(f"Processed {number}/{len(manifest['files'])} frozen files", flush=True)
    if not pieces:
        raise ValueError("No eligible development examples")
    rows = pd.concat(pieces, ignore_index=True)
    aggregate = summarize(rows)
    overall = summarize(rows.assign(entry_spread_bucket="all"))
    args.output.mkdir(parents=True)
    rows.to_parquet(args.output / "development_diagnostics.parquet", index=False)
    pd.concat([aggregate, overall], ignore_index=True).to_csv(args.output / "summary.csv", index=False)
    save_json(args.output / "protocol.json", {"source": str(source), "horizons_seconds": HORIZONS,
        "history_observations": 60, "future_spread_filter": "none",
        "entry_spread_filter": "none; entry-only spread buckets reported separately",
        "cohort": "Common complete 60-second future cohort; spacing covers history plus longest forecast",
        "protected": "Original reserved v2 sessions/time range and known v3 holdout sessions excluded",
        "costs": "Gross after bid/ask crossing; 2c/4c round-trip cost buffers are sensitivity scenarios, not actual fees",
        "fee_reference": "https://kalshi.com/fee-schedule",
        "limits": "Hindsight direction/exit maxima are opportunity upper bounds, not strategies or achievable performance; no model fitted or orders submitted"})
    print(overall.to_string(index=False), flush=True)
    report = ["# Large-swing opportunity diagnostic", "",
        "All valid spreads retained. Development data only; holdout outcomes untouched.",
        "Entry/exit use recorded bid/ask quotes. Future spread width never filters examples.", "",
        "Hindsight direction and peak-exit returns are upper-bound diagnostics, not achievable strategy returns.",
        "A model must choose direction/entry using past information; exit rules must be fixed before evaluation.",
        "Fees, latency, slippage, queueing and capacity are not modeled. Cost buffers are sensitivity scenarios.",
        "Complete future quote availability is required, so missing/one-sided periods remain censored, not no-opportunity labels.",
        "All horizons use the same 60-second-future cohort; short sessions cannot contribute.", "",
        "```", overall.to_string(index=False), "```", "",
        "See summary.csv for entry-spread buckets and development_diagnostics.parquet for row identities.",
        "Next: define a large executable-return target, fit past-only entry models, and evaluate predeclared",
        "fixed-horizon or take-profit/stop-loss exits. Do not select peaks or tune exits using validation outcomes."]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
