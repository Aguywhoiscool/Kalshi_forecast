"""Freeze final metadata first; materialize outcome intervals by explicit scope."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.causal_features import context_indices, verified_timing
from src.causal_swing_data import align_future
from src.v3_research import FEATURES, digest, overlaps, save_json, valid_rows


def declare_segments(source, output, previous):
    sessions = pd.read_parquet(source / "windows.parquet").sort_values("input_start")
    old = pd.read_parquet(previous / "windows.parquet")
    prior_end = int(old.target_end.max())
    prior_final_registry = []
    for marker in sorted(output.parent.glob("accuracy*/final_labels_opened.json")):
        segment_path = marker.parent / "segments.json"
        prior_segments = json.loads(segment_path.read_text())
        prior_end = max(prior_end, int(prior_segments["final_end_ns"]))
        prior_final_registry.append({"run": str(marker.parent.resolve()), "final_end_ns": int(prior_segments["final_end_ns"]),
            "opened_marker_sha256": digest(marker), "segments_sha256": digest(segment_path)})
    cutoff = max(int(sessions.iloc[int(len(sessions)*.8)].input_start), prior_end+70*10**9)
    final = sessions.loc[sessions.input_start >= cutoff].copy()
    development = sessions.loc[sessions.target_end <= cutoff-70*10**9].copy()
    # Read only ticker metadata, never prices or future return labels.
    proxies = set()
    for session in final.session_id:
        tickers = pd.read_csv(source / "snapshot" / f"features_{session}.csv", usecols=["ticker"]).ticker.dropna().unique()
        proxies.update(str(t).rsplit("-", 1)[0] for t in tickers)
    development.to_parquet(output / "development_sessions.parquet", index=False)
    final.to_parquet(output / "final_sessions.parquet", index=False)
    result = {"source": str(source.resolve()), "source_manifest_sha256": digest(source / "input_manifest.json"),
        "previous_development_max_ns": int(old.target_end.max()), "final_cutoff_ns": cutoff,
        "previously_opened_final_registry": prior_final_registry, "freshness_lower_bound_ns": prior_end,
        "development_end_ns": cutoff-70*10**9, "final_end_ns": int(final.target_end.max()) if len(final) else cutoff,
        "development_sessions": len(development), "final_sessions": len(final), "final_event_proxies": len(proxies),
        "mode": "final_eligible" if len(proxies) >= 20 else "development_only",
        "final_read_policy": "Metadata-only until choices.json and choices.sha256 are frozen; one final_started marker",
        "development_metadata_sha256": digest(output / "development_sessions.parquet"),
        "final_metadata_sha256": digest(output / "final_sessions.parquet")}
    save_json(output / "segments.json", result)
    return result


def prepare(source, output, scope):
    segments = json.loads((output / "segments.json").read_text())
    if str(source.resolve()) != segments["source"] or digest(source / "input_manifest.json") != segments["source_manifest_sha256"]:
        raise ValueError("Frozen source/manifest mismatch")
    if scope not in ("development", "final"):
        raise ValueError("Unknown scope")
    if scope == "final":
        if (output / "final").exists():
            raise ValueError("Final artifacts already exist; refusing re-entry")
        if not (output / "final_started.json").exists() or not (output / "choices.sha256").exists():
            raise ValueError("Final outcomes are quarantined")
        frozen = (output / "choices.sha256").read_text()
        choices = json.loads((output / "choices.json").read_text())
        started = json.loads((output / "final_started.json").read_text())
        if digest(output / "choices.json") != frozen or started["choices_sha256"] != frozen:
            raise ValueError("Frozen final choices mismatch")
        if choices["segments_sha256"] != digest(output / "segments.json") or choices["protocol_sha256"] != digest(output / "protocol.json"):
            raise ValueError("Frozen final protocol/segment mismatch")
        frozen_protocol = json.loads((output / "protocol.json").read_text())
        for name, expected in frozen_protocol.get("development_artifacts_sha256", {}).items():
            if digest(output / "development" / name) != expected:
                raise ValueError("Frozen development artifacts changed")
        if started["refit_manifest_sha256"] != digest(output / "refit_manifest.json"):
            raise ValueError("Frozen refit manifest changed")
        for name, expected in json.loads((output / "refit_manifest.json").read_text()).items():
            path = (output / name).resolve()
            if not path.is_relative_to((output / "refits").resolve()) or digest(path) != expected:
                raise ValueError("Frozen refit bytes changed")
    session_path = output / f"{scope}_sessions.parquet"
    if digest(session_path) != segments[f"{scope}_metadata_sha256"]:
        raise ValueError("Segment identities changed")
    sessions = pd.read_parquet(session_path)
    protected = json.loads((source / "protected_v2_period.json").read_text())
    manifest = {i["name"]: i["sha256"] for i in json.loads((source / "input_manifest.json").read_text())["files"]}
    pieces, histories, timing_parts, future_parts, known_parts, pending = [], [], [], [], [], []
    for number, session in enumerate(sessions.session_id):
        if session in protected["sessions"]:
            raise ValueError("Reserved session")
        name = f"features_{session}.csv"
        path = source / "snapshot" / name
        if digest(path) != manifest[name]:
            raise ValueError("Source hash changed")
        frame = pd.read_csv(path)
        if not frame.session_id.eq(session).all():
            raise ValueError("Filename/session mismatch")
        required = {"scheduled_close_time", "close_time_observed_at", "seconds_to_scheduled_close"}
        if not required <= set(frame.columns):
            continue
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for ticker, group in frame.groupby("ticker"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            times = group.timestamp.array.as_unit("ns").asi8
            closes = pd.to_datetime(group.scheduled_close_time, utc=True, errors="coerce", format="ISO8601").array.as_unit("ns").asi8
            observed = pd.to_datetime(group.close_time_observed_at, utc=True, errors="coerce", format="ISO8601").array.as_unit("ns").asi8
            remaining = pd.to_numeric(group.seconds_to_scheduled_close, errors="coerce").to_numpy()
            timing_ok = ((times > 0) & (closes > 0) & (observed > 0) & (observed <= times)
                & np.isfinite(remaining) & (remaining >= 0) & (np.abs((closes-times)/1e9-remaining) <= .001))
            if timing_ok.any():
                verified_timing(times[timing_ok], closes[timing_ok], observed[timing_ok], remaining[timing_ok])
            valid = valid_rows(group) & timing_ok
            ends = np.arange(179, len(group))
            indices, okay = context_indices(times, group.session_id.to_numpy(), ends, length=180, valid=valid)
            ends, indices = ends[okay], indices[okay]
            bound = segments["development_end_ns"] if scope == "development" else segments["final_end_ns"]
            selected, next_signal = [], -1
            for local, end in enumerate(ends):
                finish = times[end]+60*10**9+500_000_000
                if scope == "final" and times[end-179] < segments["final_cutoff_ns"]:
                    continue
                if finish > bound or overlaps(times[end-179], finish, protected):
                    continue
                if times[end] >= next_signal:
                    selected.append(local)
                    next_signal = times[end]+120*10**9
            if not selected:
                continue
            indices, ends = indices[selected], ends[selected]
            history = group[FEATURES].to_numpy(float)[indices]
            if scope == "development":
                future, _, known = align_future(times, group[["bid", "ask", "bid_depth", "ask_depth"]].to_numpy(float),
                    valid_rows(group), times[ends], np.full(len(ends), 60))
                future_parts.append(future)
                known_parts.append(known)
            else:
                pending.append((name, str(ticker), ends.copy()))
            pieces.append(pd.DataFrame({"session_id": session, "ticker": str(ticker),
                "event_proxy": str(ticker).rsplit("-", 1)[0], "series": str(ticker).split("-")[0],
                "input_start": times[ends-179], "input_end": times[ends], "target_end": times[ends]+60*10**9+500_000_000,
                "seconds_to_close": remaining[ends]}))
            histories.append(history)
            timing_parts.append(remaining[indices])
        if number % 100 == 0:
            print(json.dumps({"stage": "materialize", "scope": scope, "sessions": number+1}), flush=True)
    if not pieces:
        raise ValueError("No common verified 180-second histories in "+scope)
    meta = pd.concat(pieces, ignore_index=True)
    if scope == "final":
        proxies = int(meta.event_proxy.nunique())
        if proxies < 20:
            save_json(output / "final_gate_unavailable.json", {"candidate_event_proxies": proxies,
                "mode": "development_only", "future_quotes_materialized": False})
            return None
        # Final labels are first opened only AFTER the actual feature-eligible
        # cohort passes the event-proxy gate. First pass used input histories.
        save_json(output / "final_labels_opened.json", {"candidate_event_proxies": proxies,
            "choices_sha256": digest(output / "choices.json"), "utc": pd.Timestamp.now(tz="UTC").isoformat()})
        for name, ticker, ends in pending:
            path = source / "snapshot" / name
            if digest(path) != manifest[name]:
                raise ValueError("Source changed before final outcome opening")
            frame = pd.read_csv(path)
            frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
            frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
            group = frame.loc[frame.ticker == ticker].sort_values("timestamp").reset_index(drop=True)
            for column in FEATURES:
                group[column] = pd.to_numeric(group[column], errors="coerce")
            times = group.timestamp.array.as_unit("ns").asi8
            future, _, known = align_future(times, group[["bid", "ask", "bid_depth", "ask_depth"]].to_numpy(float),
                valid_rows(group), times[ends], np.full(len(ends), 60))
            future_parts.append(future)
            known_parts.append(known)
    order = meta.sort_values(["input_end", "ticker"]).index.to_numpy()
    meta = meta.iloc[order].reset_index(drop=True)
    folder = output / scope
    folder.mkdir()
    meta.to_parquet(folder / "metadata.parquet", index=False)
    for name, parts in (("history180", histories), ("timing180", timing_parts), ("future_quotes", future_parts), ("future_valid", known_parts)):
        np.save(folder / f"{name}.npy", np.concatenate(parts)[order])
    return folder


def walkforward(meta):
    sessions = meta.groupby("session_id").input_start.min().sort_values()
    folds = []
    for fraction in (.4, .6, .8):
        a, b = int(len(sessions)*fraction), min(len(sessions), int(len(sessions)*(fraction+.2)+1e-8))
        train_sessions, val_sessions = set(sessions.index[:a]), set(sessions.index[a:b])
        train = meta.index[meta.session_id.isin(train_sessions)].to_numpy()
        validation = meta.index[meta.session_id.isin(val_sessions) & (meta.input_start >= meta.iloc[train].target_end.max()+70*10**9)].to_numpy()
        inner_sessions = meta.iloc[train].groupby("session_id").input_start.min().sort_values()
        fit_sessions = set(inner_sessions.index[:int(len(inner_sessions)*.8)])
        fit = meta.index[meta.session_id.isin(fit_sessions)].to_numpy()
        calibration = meta.index[meta.session_id.isin(train_sessions-fit_sessions) & (meta.input_start >= meta.iloc[fit].target_end.max()+70*10**9)].to_numpy()
        if min(len(fit), len(calibration), len(validation)) == 0:
            raise ValueError("Insufficient chronology for three folds")
        folds.append({"fit": fit, "calibration": calibration, "validation": validation})
    return folds
