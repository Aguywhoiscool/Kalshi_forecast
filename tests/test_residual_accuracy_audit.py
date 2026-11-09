import numpy as np
import pytest
from src.residual_accuracy_audit import audit,replay,check_final_registry,extended_metrics,check_execution_registry
from src.residual_accuracy_models import fit,predict


def test_final_audit_refuses_before_any_final_or_protocol_reads(tmp_path):
    with pytest.raises(ValueError,match='Completed final'):
        audit(tmp_path,allow_final=True)


def test_replay_actual_tiny_saved_model_and_tampered_predictions(tmp_path):
    h=np.zeros((12,60,7))
    h[:,:,0]=.4;h[:,:,1]=.42;h[:,:,2]=.02;h[:,:,3]=.41;h[:,:,4:6]=100
    lower=np.full((12,2),-6.)
    upper=lower.copy();upper[2]=10
    spec=dict(architecture='mlp',target='residual',loss='huber5',layers=[128,64],dropout=.1)
    fit_rows,cal_rows,rows=np.arange(8),np.arange(8,10),np.arange(10,12)
    fitted=fit(h,lower,upper,fit_rows,cal_rows,spec,42,tmp_path,epochs=2)
    raw=np.clip(predict(fitted,h[cal_rows]),-104,96)
    bias=np.mean(np.clip(raw,lower[cal_rows],upper[cal_rows])-raw,axis=0)
    saved=dict(rows=rows,raw_calibration=raw,correction=bias,
        prediction=np.clip(np.clip(predict(fitted,h[rows]),-104,96)+bias,-104,96))
    replay(tmp_path,h,fit_rows,cal_rows,lower,upper,saved)
    saved['prediction'][0,0]+=1
    with pytest.raises(AssertionError):
        replay(tmp_path,h,fit_rows,cal_rows,lower,upper,saved)


def test_final_registry_rejects_duplicate_identity_even_with_right_count():
    entries=[dict(name=f'{a}_{h}_seed{s}',architecture=a,horizon=h,seed=s)
        for a in ('gru','tcn','mlp') for h in (30,60) for s in (7,42,101)]
    keys={e['name'] for e in entries}|{f'ensemble_{h}_seed{s}' for h in (30,60) for s in (7,42,101)}|{f'{c}_{h}' for c in ('nochange','current_gru','strongest_default') for h in (30,60)}
    check_final_registry(entries,keys)
    entries[-1]=entries[0]
    with pytest.raises(ValueError,match='unique'):
        check_final_registry(entries,keys)


def test_independent_extended_metric_uses_observed_sides_only():
    lower=np.array([[0.,0.],[10.,10.]])
    prediction=np.array([[1.,-3.],[90.,90.]])
    status=np.array([[0,1],[2,4]])
    metric=extended_metrics(prediction,lower,np.full((2,2),100.),status)
    assert metric['observed_mae']==2
    assert metric['signed_bias_cents']==-1
    assert metric['median_absolute_error_cents']==2
    assert metric['p90_absolute_error_cents']==pytest.approx(2.8)


def test_execution_registry_rejects_missing_zero_trade_records():
    complete={f'{a}_{h}_seed42':{} for a in ('gru','tcn','mlp','ensemble') for h in (30,60)}
    complete.update({f'{c}_{h}':{} for c in ('nochange','current_gru','strongest_default') for h in (30,60)})
    check_execution_registry(complete)
    del complete['nochange_30']
    with pytest.raises(ValueError,match='14 execution'):
        check_execution_registry(complete)
