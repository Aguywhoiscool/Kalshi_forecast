"""Local read-only inference over Kalshi WebSocket or synced feature files."""

import argparse
import asyncio
from collections import deque, OrderedDict
from contextlib import asynccontextmanager
import copy
import hashlib
from io import StringIO
import json
from pathlib import Path
import threading
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from threadpoolctl import threadpool_limits

from src.causal_swing_data import history_ends
from src.model_benchmark import FEATURE_NAMES, summarize
from src.model_scoreboard import SCOREBOARD_MODELS, build_scoreboard
from src.swing_return_experiment import MODEL_NAMES, mc_expected
from src.v3_research import FEATURES, overlaps, valid_rows


ROOT = Path(__file__).resolve().parents[1]


def run_fingerprint(run):
    """Content identity of weights, scalers, calibration and recorded traces."""
    paths = [run/"protocol.json",run/"metrics.json",*sorted((run/"models").glob("*.joblib")),
        *sorted(run.glob("gru_*/best.keras")),*sorted(run.glob("gru_*/scaler.joblib")),
        *sorted((run/"traces").glob("*.json"))]
    combined, manifest = hashlib.sha256(), {}
    for path in paths:
        if not path.is_file(): continue
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                digest.update(block)
        name=path.relative_to(run).as_posix()
        manifest[name]=digest.hexdigest()
        combined.update((name+":"+manifest[name]+"\n").encode())
    return combined.hexdigest()[:24],manifest


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


def evenly(count, limit):
    """At most `limit` evenly spaced indices over range(count), keeping both ends."""
    return np.unique(np.linspace(0, count-1, min(limit, count)).round().astype(int)) if count else np.array([], int)


def compact_details(model, details):
    """Bound window payload size; drops only bulky raw draws, never recomputes values."""
    if model != "monte_carlo":
        return details
    keep = evenly(len(details["preview_paths"]), 32)
    convergence = np.asarray(details["convergence_yes"], float)
    points = evenly(len(convergence), 128)
    compact = {key: details[key] for key in ("kind", "seed", "path_count", "scenario_percentiles", "histogram", "limitations")
               if key in details}
    compact.update(preview_indices=[details["preview_indices"][i] for i in keep],
        preview_paths=[details["preview_paths"][i] for i in keep],
        preview_returns=[details["preview_returns"][i] for i in keep],
        convergence_yes=convergence[points].tolist(),
        convergence_paths=(points+1).tolist())  # Paths averaged at each plotted point.
    return compact


def corrected(raw, correction):
    result = np.clip(np.asarray(raw, dtype=float), -104, 96)
    result = np.clip(result + correction, -104, 96)
    if result.shape[-1] != 2 or not np.isfinite(result).all():
        raise ValueError("Invalid YES/NO return forecast")
    return result


class Predictor:
    def __init__(self, run):
        self.models, self.scalers, self.corrections = {}, {}, {}
        protocol = json.loads((run/"protocol.json").read_text())
        if protocol.get("feature_count",7) != 7:
            raise ValueError("Live dashboard requires the seven-feature artifact; timing models are separate comparisons")
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
        self.run_id,self.artifact_manifest = run_fingerprint(run)
        self.forecasts = OrderedDict()
        self.inference_lock = threading.RLock()
        self.feed = feed
        protocol = json.loads((run / "protocol.json").read_text())
        self.protected = json.loads((Path(protocol["source"]) / "protected_v2_period.json").read_text())
        self.lock, self.cache, self.timeline = threading.Lock(), {}, deque(maxlen=240)
        self.state = {"status": "loading_models", "markets": [], "timeline": [], "errors": [],
            "model_run": run.name, "source": "Kalshi direct WebSocket" if feed else "Syncthing feature files", "poll_seconds": 5,
            "stale_after_seconds": 30, "cost_buffer_cents": 4, "entry_margin_cents": 1,
            "disclaimer": "Research forecasts, not executable orders. Conservative return targets; no demonstrated trading edge."}
        self.predictor = None
        self.windows_cache, self.windows_lock = OrderedDict(), threading.Lock()
        self.scoreboard_value, self.scoreboard_lock = None, threading.Lock()

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
            # C order like training history.npy: summarize() reductions are layout-sensitive at ~1e-16,
            # enough to flip tree thresholds between the prediction and the stored copy explained later.
            item = {**item, "history": np.ascontiguousarray(item["history"], dtype=float)}
            key = (item["session_id"], item["ticker"], item["input_end"])
            keep.add(key)
            age = now-item["input_end"]/1e9
            # Never manufacture live forecasts on old or future-dated clocks.
            fresh = -2 <= age <= 30
            if fresh and key not in self.cache:
                with self.inference_lock:
                    predictions = self.predictor.predict(item)
                forecast_id = hashlib.sha256(f"{self.run_id}|{key}".encode()).hexdigest()[:24]
                predictions = [{**p,"forecast_id":forecast_id} for p in predictions]
                stored = {**item,"history":item["history"].copy(),"predictions":predictions}
                with self.lock:
                    self.forecasts[forecast_id] = stored
                    while len(self.forecasts) > 240:
                        self.forecasts.popitem(last=False)
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

    def explain(self, forecast_id, model, horizon, include_all=False, tree_index=0):
        from src.prediction_explanations import explanation
        with self.lock:
            item = self.forecasts.get(forecast_id)
            if item is None:
                raise KeyError("Forecast expired or unknown; select a current forecast")
            item = copy.deepcopy(item)
        with self.inference_lock:
            if run_fingerprint(self.run)[0] != self.run_id:
                raise ValueError("Model artifacts changed. Restart the dashboard and select a new forecast.")
            result = explanation(self.predictor,item,model,horizon,self.run,include_all,tree_index)
        result.update(forecast_id=forecast_id,run_id=self.run_id,model_run=self.run.name,
            ticker=item["ticker"],session_id=item["session_id"],input_end=item["input_end"],
            forecast_age_seconds=max(0,time.time()-item["input_end"]/1e9),historical=True)
        result["artifact_hashes"] = self.artifact_manifest
        return result

    def windows(self, forecast_id, horizon):
        """Every model's actual explanation for one forecast; one model's failure is isolated."""
        from src.prediction_explanations import explanation
        if horizon not in (30, 60):
            raise ValueError("Horizon must be 30 or 60")
        key = (forecast_id, horizon)
        # Expiry and artifact checks run before the cache so cached payloads never outlive either.
        with self.lock:
            item = self.forecasts.get(forecast_id)
            if item is None:
                raise KeyError("Forecast expired or unknown; select a current forecast")
            item = copy.deepcopy(item)
        if run_fingerprint(self.run)[0] != self.run_id:
            raise ValueError("Model artifacts changed. Restart the dashboard and select a new forecast.")
        with self.windows_lock:
            if key in self.windows_cache:
                self.windows_cache.move_to_end(key)
                return copy.deepcopy(self.windows_cache[key])
        history = np.asarray(item["history"], float)
        models, errors = {}, {}
        with self.inference_lock:
            for model in SCOREBOARD_MODELS:
                try:
                    result = explanation(self.predictor, item, model, horizon, self.run)
                    models[model] = {"calculation": result["calculation"],
                                     "details": compact_details(model, result["details"])}
                except Exception as error:  # Includes failed prediction/explanation parity assertions.
                    errors[model] = f"{type(error).__name__}: {str(error)[:300]}"
        payload = {"forecast_id": forecast_id, "ticker": item["ticker"], "session_id": item["session_id"],
            "horizon": horizon, "input_end": item["input_end"], "model_run": self.run.name, "run_id": self.run_id,
            "input": {"features": history.tolist(), "summary_names": list(FEATURE_NAMES),
                      "summary_values": summarize(history[None])[0].tolist()},
            "models": models, "errors": errors,
            "disclaimer": "Actual saved model calculations for one historical forecast. Not orders; no demonstrated trading edge."}
        if errors:  # Failures may be transient (e.g. a file mid-sync); retry on the next request.
            return copy.deepcopy(payload)
        with self.windows_lock:
            self.windows_cache[key] = payload
            self.windows_cache.move_to_end(key)
            while len(self.windows_cache) > 32:
                self.windows_cache.popitem(last=False)
        return copy.deepcopy(payload)

    def scoreboard(self):
        """Lazily computed relative leaderboard; failures are not cached.

        Refuses (like explanations) once the run's artifacts differ from the ones loaded at startup."""
        with self.scoreboard_lock:
            if run_fingerprint(self.run)[0] != self.run_id:
                self.scoreboard_value = None
                raise ValueError("Model artifacts changed. Restart the dashboard.")
            if self.scoreboard_value is None or self.scoreboard_value[0] != self.run_id:
                self.scoreboard_value = (self.run_id, {**build_scoreboard(self.run), "run_id": self.run_id})
            return copy.deepcopy(self.scoreboard_value[1])


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
    app.add_middleware(GZipMiddleware,minimum_size=1024)

    @app.get("/api/state")
    def state():
        return engine.snapshot()

    @app.get("/api/explanations/{forecast_id}/{model}")
    def explain(forecast_id: str, model: str, horizon: int=30, download: bool=False, tree: int=0):
        try:
            result = engine.explain(forecast_id,model,horizon,download,tree)
        except KeyError as error:
            raise HTTPException(status_code=410,detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400,detail=str(error)) from error
        headers = {"Cache-Control":"no-store"}
        if download:
            headers["Content-Disposition"] = 'attachment; filename="model-explanation.json"'
        return JSONResponse(result,headers=headers)

    @app.get("/api/windows/{forecast_id}")
    def windows(forecast_id: str, horizon: str="30"):
        if horizon not in ("30","60"):
            raise HTTPException(status_code=400,detail="Horizon must be 30 or 60")
        try:
            result = engine.windows(forecast_id,int(horizon))
        except KeyError as error:
            raise HTTPException(status_code=410,detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400,detail=str(error)) from error
        return JSONResponse(result,headers={"Cache-Control":"no-store"})

    @app.get("/api/scoreboard")
    def scoreboard():
        try:
            result = engine.scoreboard()
        except Exception as error:
            raise HTTPException(status_code=503,detail=f"Scoreboard unavailable: {str(error)[:250]}") from error
        return JSONResponse(result,headers={"Cache-Control":"no-store"})

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
