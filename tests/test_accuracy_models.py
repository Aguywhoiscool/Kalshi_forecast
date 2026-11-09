import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.accuracy_models import Data, fit_model, load_model, raw_predict


def make_partition(folder,n=36):
    folder.mkdir()
    t=np.arange(180)
    histories=[]
    for i in range(n):
        mid=.4+.01*np.sin(t*.2+i*.3)
        histories.append(np.column_stack([mid-.01,mid+.01,np.full(180,.02),mid,
            np.full(180,2.),np.full(180,3.),np.full(180,-.2)]))
    np.save(folder/'history180.npy',np.array(histories))
    np.save(folder/'timing180.npy',np.tile(1800-t,(n,1)))
    np.save(folder/'future_quotes.npy',np.zeros((n,61,4)))
    np.save(folder/'future_valid.npy',np.ones((n,61),bool))
    pd.DataFrame({'session_id':['s']*n,'ticker':['T']*n,'input_end':np.arange(n)*120*10**9}).to_parquet(folder/'metadata.parquet',index=False)
    return Data(folder)


@pytest.mark.parametrize('model,params',[
    ('mean',{}),('ridge',{'alpha':1.}),('sgd',{'learning_rate':.001,'alpha':.001}),
    ('decision_tree',{}),('random_forest',{}),
    ('gradient_boosting',{'learning_rate':.1,'max_leaf_nodes':7,'l2_regularization':1.,'max_iter':3})])
def test_real_tabular_fit_reload_preserves_predictions_and_fit_only_scaling(tmp_path,model,params):
    data=make_partition(tmp_path/'data')
    lower=np.column_stack([np.arange(36)*.1-4,np.arange(36)*-.1-5]);upper=lower.copy()
    folder=tmp_path/model
    fitted=fit_model(model,'engineered60','conservative',params,42,data,lower,upper,np.arange(24),np.arange(24,30),folder)
    before=raw_predict(fitted,data,np.arange(30,36))
    after=raw_predict(load_model(folder,model,'engineered60'),data,np.arange(30,36))
    np.testing.assert_allclose(before,after,atol=1e-7)
    if model in ('ridge','sgd'):
        scaler=fitted['model'].named_steps['standardscaler']
        assert scaler.n_samples_seen_==24
        np.testing.assert_allclose(scaler.mean_,data.table('engineered60')[:24].mean(axis=0,dtype=np.float64))


def test_actual_interval_gru_tiny_fit_checkpoint_reload(tmp_path,monkeypatch):
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    losses=[]
    original_compile=tf.keras.Sequential.compile
    def capture_loss(self,*args,**kwargs):
        if 'loss' in kwargs: losses.append(kwargs['loss'])
        return original_compile(self,*args,**kwargs)
    monkeypatch.setattr(tf.keras.Sequential,'compile',capture_loss)
    data=make_partition(tmp_path/'data',n=12)
    lower=np.tile([-5.,-7.],(12,1));upper=lower+2
    folder=tmp_path/'gru'
    fitted=fit_model('gru','seven60','interval',{'layers':[2,2],'dropout':0.},42,data,lower,upper,np.arange(8),np.arange(8,10),folder)
    predicted=raw_predict(fitted,data,[10,11])
    reloaded=raw_predict(load_model(folder,'gru','seven60'),data,[10,11])
    np.testing.assert_allclose(predicted,reloaded,atol=1e-6)
    epochs=pd.read_csv(folder/'epochs.csv')
    assert 1<=len(epochs)<=30 and np.isfinite(epochs.val_loss).all()
    scaler=joblib.load(folder/'scaler.joblib')
    assert scaler.n_samples_seen_==8*60
    assert np.isfinite(predicted).all()
    assert len(losses)==1
    y=tf.constant([[-.05,-.07,-.03,-.05]],dtype=tf.float32)
    prediction=tf.Variable([[.05,-.06]],dtype=tf.float32)
    with tf.GradientTape() as tape:
        objective=losses[0](y,prediction)
    # Residual8c on YES and0 on NO: mean Huber5c =13.75 cents²=.001375 dollars².
    np.testing.assert_allclose(objective.numpy(),.001375,atol=1e-8)
    np.testing.assert_allclose(tape.gradient(objective,prediction).numpy(),[[.025,0]],atol=1e-7)


def test_mc_controls_never_access_fit_labels_and_cache_real_simulations(tmp_path,monkeypatch):
    data=make_partition(tmp_path/'data',n=2)
    class Forbidden:
        def __array__(self,*args,**kwargs): raise AssertionError('MC must not fit future labels')
    for model in ('monte_carlo_60','monte_carlo_180'):
        fitted=fit_model(model,'seven60','conservative',{},42,data,Forbidden(),Forbidden(),[0],[1],tmp_path/model)
        fitted['horizon']=30
        predicted=raw_predict(fitted,data,[0,1])
        loaded=load_model(tmp_path/model,model,'seven60');loaded['horizon']=30
        np.testing.assert_array_equal(predicted,raw_predict(loaded,data,[0,1]))
        assert data.mc_predictions(model,60).shape==(2,2)
        assert raw_predict(fitted,data,[]).shape==(0,2)
    assert set(data.mc_cache)=={'monte_carlo_60','monte_carlo_180'}


def test_unknown_feature_family_and_overlap_fail_before_training(tmp_path):
    data=make_partition(tmp_path/'data',n=4)
    with pytest.raises(ValueError,match='feature family'): data.sequence('unknown')
    bounds=np.zeros((4,2))
    with pytest.raises(ValueError,match='overlap'):
        fit_model('ridge','seven60','conservative',{},42,data,bounds,bounds,[0,1],[1,2],tmp_path/'bad')
    with pytest.raises(ValueError,match='indices'):
        fit_model('ridge','seven60','conservative',{},42,data,bounds,bounds,[-1],[1,2],tmp_path/'bad')
    with pytest.raises(ValueError,match='indices'):
        fit_model('ridge','seven60','conservative',{},42,data,bounds,bounds,[0,0],[1,2],tmp_path/'bad')


def test_family_shapes_expose_actual_engineered_context(tmp_path):
    data=make_partition(tmp_path/'data',n=2)
    assert data.table('seven60').shape==(2,50)
    assert data.table('engineered60').shape==(2,134)
    assert data.table('engineered180').shape==(2,136)
    assert data.sequence('seven60').shape==(2,60,7)
    assert data.sequence('engineered60').shape==(2,60,22)
    assert data.sequence('engineered180').shape==(2,180,22)
