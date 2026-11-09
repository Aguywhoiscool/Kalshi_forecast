"""Fixed, reloadable accuracy-study predictors. No split/final-selection policy."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from src.accuracy_targets import AccuracySGD
from src.causal_features import build_features, engineered_sequence
from src.model_benchmark import summarize
from src.model_traces import simulate_mc


FAMILIES = ("seven60", "engineered60", "engineered180")
MC_MODELS = ("monte_carlo_60", "monte_carlo_180")


class Data:
    """Loads only the explicit partition chosen by the research orchestrator."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = pd.read_parquet(self.folder / "metadata.parquet")
        self.history = np.load(self.folder / "history180.npy", mmap_mode="r")
        self.timing = np.load(self.folder / "timing180.npy", mmap_mode="r")
        self.future = np.load(self.folder / "future_quotes.npy", mmap_mode="r")
        self.known = np.load(self.folder / "future_valid.npy", mmap_mode="r")
        if self.history.shape != (len(self.meta),180,7) or self.timing.shape != (len(self.meta),180):
            raise ValueError("Partition history/timing/metadata shapes disagree")
        self.tables, self.sequences, self.mc_cache = {}, {}, {}

    def table(self, family):
        if family not in FAMILIES:
            raise ValueError("Unknown feature family")
        if family not in self.tables:
            h = self.history[:, -60:]
            value = summarize(h) if family == "seven60" else build_features(h,self.timing[:,-60:],
                context180=self.history if family == "engineered180" else None)[0]
            self.tables[family] = value.astype(np.float32)
        return self.tables[family]

    def sequence(self, family):
        if family not in FAMILIES:
            raise ValueError("Unknown feature family")
        if family not in self.sequences:
            if family == "seven60":
                value = self.history[:, -60:]
            elif family == "engineered60":
                value = engineered_sequence(self.history[:,-60:],self.timing[:,-60:])[0]
            else:
                value = engineered_sequence(self.history,self.timing)[0]
            self.sequences[family] = value.astype(np.float32)
        return self.sequences[family]

    def mc_predictions(self, mode, horizon):
        if mode not in MC_MODELS or horizon not in (30,60):
            raise ValueError("Unknown Monte Carlo control/horizon")
        if mode not in self.mc_cache:
            predictions = {h:np.empty((len(self.meta),2)) for h in (30,60)}
            for i,row in enumerate(self.meta.itertuples()):
                identity = {"session_id":row.session_id,"ticker":row.ticker,"input_end":int(row.input_end)}
                result = simulate_mc(self.history[i,-60:],identity,
                    context=self.history[i] if mode == "monte_carlo_180" else None)
                for h in predictions:
                    predictions[h][i] = result["predictions"][h]
            self.mc_cache[mode] = predictions
        return self.mc_cache[mode][horizon]


def fit_model(model_name, family, objective, params, seed, data, lower, upper, fit, cal, folder):
    if family not in FAMILIES or objective not in ("conservative","interval"):
        raise ValueError("Unknown feature family or objective")
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if model_name in MC_MODELS:
        result = {"kind":"monte_carlo","model_name":model_name,"family":family}
        (folder/"descriptor.json").write_text(json.dumps(result),encoding="utf-8")
        return result
    fit,cal = np.asarray(fit,dtype=int),np.asarray(cal,dtype=int)
    if not len(fit) or (model_name=="gru" and not len(cal)):
        raise ValueError("Empty fit/calibration population")
    if np.intersect1d(fit,cal).size:
        raise ValueError("Fit and calibration observations overlap")
    for indices in (fit,cal):
        if (indices<0).any() or (indices>=len(data.meta)).any() or len(np.unique(indices))!=len(indices):
            raise ValueError("Invalid or duplicated fit/calibration indices")
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if lower.shape!=upper.shape or lower.shape!=(len(data.meta),2) or not np.isfinite([lower,upper]).all() or (lower>upper).any():
        raise ValueError("Invalid two-side interval bounds")
    if objective=="interval" and model_name not in ("sgd","gru"):
        raise ValueError("Only SGD/GRU implement the interval objective")
    bounds=np.column_stack([lower,upper])
    if model_name!="gru":
        x=data.table(family)
        if model_name=="ridge": model,y=make_pipeline(StandardScaler(),Ridge(**params)),lower
        elif model_name=="sgd": model,y=make_pipeline(StandardScaler(),AccuracySGD(**params,objective=objective,seed=seed)),bounds
        elif model_name=="gradient_boosting": model,y=MultiOutputRegressor(HistGradientBoostingRegressor(**params,early_stopping=False,random_state=seed)),lower
        elif model_name=="mean": model,y=DummyRegressor(),lower
        elif model_name=="decision_tree": model,y=DecisionTreeRegressor(max_depth=8,min_samples_leaf=30,random_state=seed),lower
        elif model_name=="random_forest": model,y=RandomForestRegressor(n_estimators=64,max_depth=12,min_samples_leaf=20,n_jobs=2,random_state=seed),lower
        else: raise ValueError("Unknown model")
        with threadpool_limits(2): model.fit(x[fit],y[fit])
        joblib.dump(model,folder/"model.joblib")
        return {"kind":"tabular","model":model,"family":family}
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sequence=data.sequence(family)
    scaler=StandardScaler().fit(sequence[fit].reshape(-1,sequence.shape[2]))
    def scaled(indices):
        raw=sequence[indices]
        return scaler.transform(raw.reshape(-1,raw.shape[2])).reshape(raw.shape).astype(np.float32)
    class Objective(tf.keras.losses.Loss):
        def call(self,y,prediction):
            lo,hi=y[:,:2],y[:,2:]
            if objective=="conservative": return tf.reduce_mean(tf.square(prediction-lo),axis=-1)
            residual=prediction-tf.clip_by_value(prediction,lo,hi)
            absolute=tf.abs(residual)
            return tf.reduce_mean(tf.where(absolute<=.05,.5*residual**2,.05*(absolute-.025)),axis=-1)
    first,second=params["layers"]
    model=tf.keras.Sequential([tf.keras.layers.Input(sequence.shape[1:]),tf.keras.layers.GRU(first,return_sequences=True),
        tf.keras.layers.Dropout(params["dropout"]),tf.keras.layers.GRU(second),tf.keras.layers.Dropout(params["dropout"]),
        tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dense(2)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss=Objective())
    model.fit(scaled(fit),(bounds[fit]/100).astype(np.float32),validation_data=(scaled(cal),(bounds[cal]/100).astype(np.float32)),
        batch_size=256,epochs=30,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5),
        tf.keras.callbacks.ModelCheckpoint(folder/"best.keras",monitor="val_loss",save_best_only=True),
        tf.keras.callbacks.CSVLogger(folder/"epochs.csv")])
    model=tf.keras.models.load_model(folder/"best.keras",compile=False)
    joblib.dump(scaler,folder/"scaler.joblib")
    return {"kind":"gru","model":model,"scaler":scaler,"family":family}


def raw_predict(fitted,data,rows):
    rows=np.asarray(rows,dtype=int)
    if not len(rows): return np.empty((0,2),float)
    if rows.ndim!=1 or (rows<0).any() or (rows>=len(data.meta)).any():
        raise ValueError("Invalid prediction row indices")
    if fitted["kind"]=="monte_carlo":
        value=data.mc_predictions(fitted["model_name"],fitted["horizon"])[rows]
    elif fitted["kind"]=="tabular":
        with threadpool_limits(2): value=fitted["model"].predict(data.table(fitted["family"])[rows])
    elif fitted["kind"]=="gru":
        raw=data.sequence(fitted["family"])[rows]
        scaled=fitted["scaler"].transform(raw.reshape(-1,raw.shape[2])).reshape(raw.shape).astype(np.float32)
        value=np.concatenate([fitted["model"](scaled[start:start+256],training=False).numpy()*100 for start in range(0,len(rows),256)])
    else: raise ValueError("Unknown fitted model kind")
    value=np.asarray(value)
    if value.shape!=(len(rows),2) or not np.isfinite(value).all():
        raise ValueError("Nonfinite or incorrectly shaped model output")
    return np.clip(value,-104,96)


def load_model(folder,model_name,family):
    folder=Path(folder)
    if family not in FAMILIES or model_name not in (*MC_MODELS,"gru","mean","ridge","sgd","decision_tree","random_forest","gradient_boosting"):
        raise ValueError("Unknown model or feature family")
    if model_name in MC_MODELS:
        return {"kind":"monte_carlo","model_name":model_name,"family":family}
    if model_name!="gru":
        return {"kind":"tabular","model":joblib.load(folder/"model.joblib"),"family":family}
    import tensorflow as tf
    return {"kind":"gru","model":tf.keras.models.load_model(folder/"best.keras",compile=False),
            "scaler":joblib.load(folder/"scaler.joblib"),"family":family}
