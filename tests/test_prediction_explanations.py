import hashlib
import asyncio
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.live_dashboard import Dashboard, corrected, create_app
from src.model_benchmark import summarize
from src.prediction_explanations import explanation, linear_details, tree_details


def history():
    mid = .5+np.sin(np.arange(60)*.7)*.08
    return np.column_stack([mid-.01,mid+.01,np.full(60,.02),mid,np.full(60,4),np.full(60,5),np.full(60,-1/9)])


def data():
    x=np.random.default_rng(2).normal(size=(80,50))
    y=np.column_stack([x[:,0]*3-x[:,1]+7,x[:,2]*2-3])
    return x,y


def item_and_predictor(model,name):
    item={"session_id":"s","ticker":"T","input_end":123,"history":history()}
    correction=np.array([1.,-2.])
    values=corrected(model.predict(summarize(item["history"][None])),correction)[0]
    item["predictions"]=[{"model":name,"horizon":30,"yes":values[0],"no":values[1],"signal":"WAIT"}]
    return item,SimpleNamespace(models={(name,30):model},corrections={(name,30):correction})


def test_linear_contributions_sum_to_both_actual_model_outputs():
    x,y=data()
    model=make_pipeline(StandardScaler(),Ridge()).fit(x,y)
    detail=linear_details(model,x[:1])
    total=np.sum([f["contribution"] for f in detail["features"]],axis=0)+detail["intercept"]
    np.testing.assert_allclose(total,model.predict(x[:1])[0],atol=1e-10)


def test_real_saved_float32_ridge_contributions_reconcile():
    import joblib
    run=Path(__file__).resolve().parents[1]/"artifacts/swing_returns_all_20261106"
    if not run.exists():
        pytest.skip("Local saved Ridge artifact unavailable")
    x=summarize(np.load(run/"history.npy",mmap_mode="r")[:3])
    for horizon in (30,60):
        model=joblib.load(run/"models"/f"ridge_{horizon}.joblib")
        for row in x:
            detail=linear_details(model,row[None])
            np.testing.assert_allclose(detail["reconstructed_raw"],model.predict(row[None])[0],atol=1e-5,rtol=1e-6)


@pytest.mark.parametrize("name",["decision_tree","random_forest","gradient_boosting","ridge"])
def test_tree_ensemble_and_calibration_reconcile_with_saved_forecast(tmp_path,name):
    x,y=data()
    models={"decision_tree":DecisionTreeRegressor(max_depth=3,random_state=1),
        "random_forest":RandomForestRegressor(n_estimators=4,max_depth=3,random_state=1),
        "gradient_boosting":MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=4,min_samples_leaf=2,early_stopping=False)),
        "ridge":make_pipeline(StandardScaler(),Ridge())}
    model=models[name].fit(x,y)
    item,predictor=item_and_predictor(model,name)
    result=explanation(predictor,item,name,30,tmp_path)
    np.testing.assert_allclose(result["calculation"]["final"],[item["predictions"][0]["yes"],item["predictions"][0]["no"]])
    if name=="decision_tree":
        assert result["details"]["leaf"]==model.apply(summarize(item["history"][None]))[0]
    assert result["training"]["available"] is False


def test_mc_random_draws_rebuild_paths_and_both_return_sides(tmp_path):
    item={"session_id":"s","ticker":"T","input_end":123,"history":history()}
    seed=int.from_bytes(hashlib.sha256(b"return42|s|T|123").digest()[:8],"little")
    rng=np.random.default_rng(seed)
    increments=np.diff(item["history"][:,3]); paths=np.full((2048,61),item["history"][-1,3])
    for step in range(1,61):
        draws=rng.choice(len(increments),size=2048)
        paths[:,step]=np.minimum(.99,np.maximum(.01,paths[:,step-1]+increments[draws]))
    yes=(paths[:,30]-.01-(paths[:,1]+.01))*100-4
    no=((1-paths[:,30]-.01)-(1-paths[:,1]+.01))*100-4
    item["predictions"]=[{"model":"monte_carlo","horizon":30,"yes":yes.mean(),"no":no.mean(),"signal":"WAIT"}]
    p=SimpleNamespace(corrections={("monte_carlo",30):np.zeros(2)})
    result=explanation(p,item,"monte_carlo",30,tmp_path,True)
    trace=result["details"]["full_trace"]
    np.testing.assert_allclose(trace["paths"],paths,atol=1e-12)
    np.testing.assert_allclose(trace["per_path_returns"][30],np.column_stack([yes,no]),atol=1e-10)
    for index,preview in zip(result["details"]["preview_indices"],result["details"]["preview_paths"]):
        np.testing.assert_array_equal(preview,np.asarray(trace["paths"])[index,:31])
    assert np.std(paths[:,30])>0


def test_explanation_rejects_mismatched_saved_forecast(tmp_path):
    x,y=data(); model=make_pipeline(StandardScaler(),Ridge()).fit(x,y)
    item,predictor=item_and_predictor(model,"ridge")
    item["predictions"][0]["yes"]+=1
    with pytest.raises(AssertionError):
        explanation(predictor,item,"ridge",30,tmp_path)


def test_forecast_identity_cache_is_bounded_and_expired_ids_fail(tmp_path):
    source=tmp_path/"source"; source.mkdir()
    (source/"protected_v2_period.json").write_text(json.dumps({"start":0,"end":1,"sessions":[]}))
    run=tmp_path/"run"; run.mkdir(); (run/"protocol.json").write_text(json.dumps({"source":str(source)}))
    class Predictor:
        def __init__(self,run): pass
        def predict(self,item): return [{"model":"mean","horizon":30,"yes":0.,"no":0.,"signal":"WAIT"}]
    class Feed:
        counter=0
        def latest(self,protected):
            self.counter+=1
            return [{"session_id":"s","ticker":"T","input_end":stamp+self.counter,
                     "history":history(),"clocks":np.arange(60)*10**9}],[],{}
    stamp=time.time_ns()
    dashboard=Dashboard(run,tmp_path,Predictor,Feed())
    dashboard.update(); first=next(iter(dashboard.forecasts))
    for _ in range(241): dashboard.update()
    assert len(dashboard.forecasts)==240 and first not in dashboard.forecasts
    with pytest.raises(KeyError,match="expired"):
        dashboard.explain(first,"mean",30)
    current=next(iter(dashboard.forecasts))
    (run/"metrics.json").write_text('{"changed":true}')
    with pytest.raises(ValueError,match="artifacts changed"):
        dashboard.explain(current,"mean",30)


def test_asgi_explanation_routes_download_and_expired_identity(tmp_path):
    calls=[]
    class Engine:
        def explain(self,id,model,horizon,download,tree):
            calls.append((id,model,horizon,download,tree))
            if id=="expired": raise KeyError("Forecast expired")
            return {"forecast_id":id,"payload":"x"*5000}
        def snapshot(self): return {}
    app=create_app(run=tmp_path,dashboard=Engine(),mode="files")
    async def request(id,query):
        messages=[]
        async def receive(): return {"type":"http.request","body":b"","more_body":False}
        async def send(message): messages.append(message)
        await app({"type":"http","method":"GET","path":f"/api/explanations/{id}/ridge",
            "query_string":query,"headers":[(b"accept-encoding",b"gzip")],"scheme":"http",
            "http_version":"1.1","root_path":"","server":("test",80),"client":("test",1)},receive,send)
        return messages
    messages=asyncio.run(request("known",b"horizon=60&download=true&tree=2"))
    assert messages[0]["status"]==200
    headers=dict(messages[0]["headers"])
    assert headers[b"content-disposition"].startswith(b"attachment;")
    body=b"".join(m.get("body",b"") for m in messages[1:])
    assert json.loads(gzip.decompress(body))["forecast_id"]=="known"
    assert calls[0]==("known","ridge",60,True,2)
    assert asyncio.run(request("expired",b"horizon=30"))[0]["status"]==410
    assert asyncio.run(request("known",b"horizon=not-a-number"))[0]["status"]==422
