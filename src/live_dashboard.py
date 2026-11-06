"""Local read-only inference over Kalshi WebSocket or synced feature files."""

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import copy
from io import StringIO
import json
from pathlib import Path
import threading
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from threadpoolctl import threadpool_limits

from src.causal_swing_data import history_ends
from src.model_benchmark import summarize
from src.swing_return_experiment import MODEL_NAMES, mc_expected
from src.v3_research import FEATURES, overlaps, valid_rows


ROOT = Path(__file__).resolve().parents[1]


def read_tail(path, limit=1024*1024):
    """Read complete CSV rows only, with bounded IO on active files."""
    with path.open("rb") as stream:
        header = stream.readline()
        stream.seek(0, 2)
        size = stream.tell()
        start = max(len(header), size-limit)
        stream.seek(start)
        if start > len(header):
            stream.readline()  # Discard an incomplete leading row.
        body = stream.read(limit)
    if not body.endswith(b"\n"):
        body = body[:body.rfind(b"\n")+1]
    return pd.read_csv(StringIO((header+body).decode("utf-8")))


def latest_histories(directory, protected, max_markets=8):
    found = {}
    errors = []
    for path in sorted(directory.glob("features_*.csv"), reverse=True)[:6]:
        session = path.stem.removeprefix("features_")
        if "sync-conflict" in path.name or session in protected["sessions"]:
            continue
        try:
            frame = read_tail(path)
            frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce", format="ISO8601")
            for name in FEATURES:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).drop_duplicates(["session_id", "ticker", "timestamp"])
            if not frame.session_id.eq(session).all():
                raise ValueError("Filename/session mismatch")
            for ticker, group in frame.groupby("ticker"):
                if ticker in found:
                    continue
                group = group.sort_values("timestamp").tail(60)
                clocks = group.timestamp.array.as_unit("ns").asi8
                if len(group) != 60 or overlaps(clocks[0], clocks[-1], protected):
                    continue
                if history_ends(clocks, valid_rows(group)).tolist() != [59]:
                    continue
                found[str(ticker)] = {"ticker": str(ticker), "session_id": session,
                    "input_end": int(clocks[-1]), "clocks": clocks, "history": group[FEATURES].to_numpy(float)}
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            errors.append({"file": path.name, "message": str(error)[:180]})
    values = sorted(found.values(), key=lambda item: (-item["input_end"], item["ticker"]))[:max_markets]
    return values, errors


def corrected(raw, correction):
    result = np.clip(np.asarray(raw, dtype=float), -104, 96)
    result = np.clip(result + correction, -104, 96)
    if result.shape[-1] != 2 or not np.isfinite(result).all():
        raise ValueError("Invalid YES/NO return forecast")
    return result


class Predictor:
    def __init__(self, run):
        self.models, self.scalers, self.corrections = {}, {}, {}
        metrics = json.loads((run / "metrics.json").read_text())
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        for horizon in (30, 60):
            for name in MODEL_NAMES:
                if name == "no_trade":
                    continue
                key = (name, horizon)
                self.corrections[key] = np.array(metrics[f"{name}_{horizon}"]["calibration_correction_cents"])
                if name == "gru":
                    self.models[key] = tf.keras.models.load_model(run / f"gru_{horizon}/best.keras", compile=False)
                    self.scalers[horizon] = joblib.load(run / f"gru_{horizon}/scaler.joblib")
                elif name != "monte_carlo":
                    self.models[key] = joblib.load(run / "models" / f"{name}_{horizon}.joblib")

    def predict(self, item):
        history = item["history"][None]
        summary = summarize(history)
        mc = mc_expected(history, pd.DataFrame([{key: item[key] for key in ("session_id", "ticker", "input_end")}]))
        rows = []
        with threadpool_limits(2):
            for horizon in (30, 60):
                for name, label in MODEL_NAMES.items():
                    if name == "no_trade":
                        continue
                    if name == "monte_carlo":
                        raw = mc[horizon]
                    elif name == "gru":
                        scaled = self.scalers[horizon].transform(history.reshape(-1, 7)).reshape(1, 60, 7).astype(np.float32)
                        raw = self.models[(name, horizon)](scaled, training=False).numpy()*100
                    else:
                        raw = self.models[(name, horizon)].predict(summary)
                    yes, no = corrected(raw, self.corrections[(name, horizon)])[0]
                    yes_depth, no_depth = history[0, -1, 5], history[0, -1, 4]
                    scores = np.array([yes if yes_depth >= 1 else -np.inf, no if no_depth >= 1 else -np.inf])
                    side = int(scores.argmax())
                    signal = ("YES" if side == 0 else "NO") if scores[side] >= 1 and scores[0] != scores[1] else "WAIT"
                    rows.append({"model": name, "label": label, "horizon": horizon,
                                 "yes": float(yes), "no": float(no), "signal": signal})
        return rows


class Dashboard:
    def __init__(self, run, directory, predictor_factory=Predictor, feed=None):
        self.run, self.directory, self.factory = run, directory, predictor_factory
        self.feed = feed
        protocol = json.loads((run / "protocol.json").read_text())
        self.protected = json.loads((Path(protocol["source"]) / "protected_v2_period.json").read_text())
        self.lock, self.cache, self.timeline = threading.Lock(), {}, deque(maxlen=240)
        self.state = {"status": "loading_models", "markets": [], "timeline": [], "errors": [],
            "model_run": run.name, "source": "Kalshi direct WebSocket" if feed else "Syncthing feature files", "poll_seconds": 5,
            "stale_after_seconds": 30, "cost_buffer_cents": 4, "entry_margin_cents": 1,
            "disclaimer": "Research forecasts, not executable orders. Conservative return targets; no demonstrated trading edge."}
        self.predictor = None

    def update(self):
        if self.predictor is None:
            self.predictor = self.factory(self.run)
        if self.feed:
            items, errors, health = self.feed.latest(self.protected)
        else:
            items, errors = latest_histories(self.directory, self.protected)
            health = None
        markets = []
        now = time.time()
        keep = set()
        for item in items:
            key = (item["session_id"], item["ticker"], item["input_end"])
            keep.add(key)
            age = now-item["input_end"]/1e9
            # Never manufacture live forecasts on old or future-dated clocks.
            fresh = -2 <= age <= 30
            if fresh and key not in self.cache:
                predictions = self.predictor.predict(item)
                self.cache[key] = predictions
                self.timeline.append({"ticker": item["ticker"], "time": item["input_end"]//10**6,
                                      "predictions": predictions})
            h = item["history"]
            markets.append({"ticker": item["ticker"], "session_id": item["session_id"],
                "time": item["input_end"]//10**6, "age_seconds": max(0, age), "fresh": fresh,
                "bid": float(h[-1, 0]*100), "ask": float(h[-1, 1]*100),
                "spread": float(h[-1, 2]*100), "midpoint": float(h[-1, 3]*100),
                "imbalance": float(h[-1, 6]), "bid_depth": float(h[-1, 4]), "ask_depth": float(h[-1, 5]),
                "history": [{"time": int(clock//10**6), "bid": float(row[0]*100),
                    "ask": float(row[1]*100), "mid": float(row[3]*100)} for clock, row in zip(item["clocks"], h)],
                "predictions": self.cache.get(key, [])})
        self.cache = {key: value for key, value in self.cache.items() if key in keep}
        with self.lock:
            self.state.update(status="live" if any(m["fresh"] for m in markets) else "stale" if markets else "waiting_for_history",
                              updated_at=int(time.time()*1000), markets=markets, errors=errors,
                              timeline=list(self.timeline), feed=health)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)


def create_app(run=None, directory=None, dashboard=None, mode="kalshi", tickers=None,
               rescan_seconds=600, min_close_minutes=10):
    if run is None:
        run = Path(json.loads((ROOT / "artifacts/swing_return_latest.json").read_text())["run"])
    feed = None
    if mode == "kalshi" and dashboard is None:
        from src.dashboard_feed import KalshiFeed
        feed = KalshiFeed(tickers, rescan_seconds, min_close_minutes)
    engine = dashboard or Dashboard(Path(run), Path(directory or ROOT / "data/processed"), feed=feed)

    @asynccontextmanager
    async def lifespan(app):
        async def refresh():
            while True:
                try:
                    await asyncio.to_thread(engine.update)
                except Exception as error:
                    with engine.lock:
                        engine.state.update(status="error", errors=[{"message": str(error)[:250]}])
                await asyncio.sleep(5)
        tasks = [asyncio.create_task(refresh())]
        if feed:
            tasks.append(asyncio.create_task(feed.run()))
        yield
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="Kalshi Model Lab", lifespan=lifespan)

    @app.get("/api/state")
    def state():
        return engine.snapshot()

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--feed", choices=("kalshi", "files"), default="kalshi")
    parser.add_argument("--tickers", nargs="+", help="Optional market tickers; otherwise discover eight open markets")
    parser.add_argument("--rescan-seconds", type=int, default=600)
    parser.add_argument("--min-close-minutes", type=float, default=10)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.run, args.data_dir, mode=args.feed, tickers=args.tickers,
        rescan_seconds=args.rescan_seconds, min_close_minutes=args.min_close_minutes), host="127.0.0.1", port=args.port)
