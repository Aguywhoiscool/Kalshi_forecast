import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import src.prospective_hybrid as hybrid
from src.v3_research import digest,save_json


def history(spreads):
    value=np.zeros((len(spreads),180,7))
    value[:,:,0]=np.linspace(.1,.7,len(spreads))[:,None]
    value[:,:,1]=value[:,:,0]+np.asarray(spreads)[:,None]/100
    value[:,:,2]=np.asarray(spreads)[:,None]/100
    value[:,:,3]=(value[:,:,0]+value[:,:,1])/2
    value[:,:,4:6]=100
    return value


def frozen(tmp_path):
    run=tmp_path/'hybrid';run.mkdir()
    model=run/'model';model.mkdir()
    for name in ('best.keras','scaler.joblib','spec.json','calibration.json'):(model/name).write_text('{}')
    (run/'protected_periods.json').write_text(json.dumps(dict(start=1,end=2,sessions=[])))
    protocol=dict(created_ns=1_000_000,prospective_input_start_ns=1_000_000,consumed_final_registry=[],
        threshold_cents=4.,correction=[0.,0.],code_sha256={},
        model_sha256={p.name:digest(p) for p in model.iterdir()},protected_sha256=digest(run/'protected_periods.json'))
    save_json(run/'protocol.json',protocol);(run/'protocol.sha256').write_text(digest(run/'protocol.json'))
    return run,protocol


def source_fixture(run,protocol,events=20,cutoff=None):
    source=run/'sources'/'fixture';(source/'snapshot').mkdir(parents=True)
    base=pd.Timestamp('2026-11-01',tz='UTC')
    times=pd.date_range(base,periods=300,freq='s')
    pieces=[]
    for event in range(events):
        pieces.append(pd.DataFrame(dict(session_id='fixture',ticker=f'EVENT{event}-A',timestamp=times,
            bid=.4,ask=.42,spread=.02,midpoint=.41,bid_depth=100.,ask_depth=100.,imbalance=0.,
            scheduled_close_time=(base+pd.Timedelta(days=1)).isoformat(),close_time_observed_at=base.isoformat(),
            seconds_to_scheduled_close=86400-np.arange(300))))
    path=source/'snapshot/features_fixture.csv';pd.concat(pieces).to_csv(path,index=False)
    manifest=dict(files=[dict(name=path.name,sha256=digest(path))],consumed_final_registry=[],
        prospective_cutoff_ns=int(base.value) if cutoff is None else cutoff)
    save_json(source/'manifest.json',manifest)
    rows=[dict(session_id='fixture',input_start=int(base.value),target_end=int(times[-1].value))]
    meta,h,timing,bound=hybrid.input_cohort(source,rows,protocol,dict(start=1,end=2,sessions=[]))
    attempt=run/'readiness/fixture';attempt.mkdir(parents=True)
    meta.to_parquet(attempt/'metadata.parquet',index=False)
    np.save(attempt/'history180.npy',h);np.save(attempt/'timing180.npy',timing)
    record=dict(ready=events>=20,candidates=len(meta),event_proxies=int(meta.event_proxy.nunique()),attempt=str(attempt),
        source=str(source),protocol_sha256=digest(run/'protocol.json'),source_manifest_sha256=digest(source/'manifest.json'),
        final_cutoff_ns=manifest['prospective_cutoff_ns'],final_end_ns=bound,
        input_artifacts_sha256={p.name:digest(p) for p in attempt.iterdir()})
    save_json(run/'latest_readiness.json',record)
    return record,meta


def test_routing_threshold_equality_and_identical_price_ticks():
    h=history([3.999999999999999,4.,4.00000000000001,3.9])[:,-60:]
    predicted=np.full((4,2),7.)
    actual,mask=hybrid.route(h,predicted,4.)
    np.testing.assert_array_equal(mask,[True,True,True,False])
    np.testing.assert_allclose(actual[:3],7.)
    np.testing.assert_allclose(actual[3],-7.9)


def test_threshold_only_uses_frozen_fit_rows():
    h=history([1,1,2,2,3,3,4,4,10,10])
    expected=hybrid.training_threshold(h,np.arange(8))
    assert expected==3.25
    h[8:]=history([20,30])
    assert hybrid.training_threshold(h,np.arange(8))==expected


def test_boundary_and_registry_include_failed_and_non_accuracy_runs(tmp_path):
    for name,end in [('accuracy_old',100),('residual_old',200),('hybrid_failed',300)]:
        run=tmp_path/name;run.mkdir();save_json(run/'segments.json',dict(final_end_ns=end))
        save_json(run/'final_labels_opened.json',dict(incomplete=True))
    registry=hybrid.consumed_registry(tmp_path)
    assert len(registry)==3
    assert hybrid.prospective_boundary(1000,registry)==300+70_000_000_000
    assert hybrid.prospective_boundary(900_000_000_000,registry)==900_000_000_000


def test_zero_future_data_waits_without_outcome_alignment(tmp_path,monkeypatch):
    run,_=frozen(tmp_path);empty=tmp_path/'data';empty.mkdir()
    monkeypatch.setattr(hybrid,'align_future',lambda *a,**k:pytest.fail('outcomes opened'))
    record=hybrid.readiness(run,empty)
    assert not record['ready'] and record['candidates']==0
    assert hybrid.evaluate(run) is None
    assert not (run/'final_started.json').exists()
    assert json.loads((run/'status.json').read_text())['stage']=='waiting_for_fresh_data'


def test_actual_input_cohort_applies_prospective_start_before_spacing(tmp_path):
    run,protocol=frozen(tmp_path)
    base=pd.Timestamp('2026-11-01',tz='UTC').value
    record,meta=source_fixture(run,protocol,events=1,cutoff=base+1_000_000_000)
    assert len(meta)==1
    assert meta.input_start.iloc[0]==base+1_000_000_000
    assert meta.input_end.iloc[0]==base+180_000_000_000


def test_actual_input_cohort_future_prices_do_not_filter_current_input(tmp_path):
    run,protocol=frozen(tmp_path)
    record,meta=source_fixture(run,protocol,events=1)
    source=record['source'];path=next((Path(source)/'snapshot').glob('*.csv'))
    frame=pd.read_csv(path)
    frame.loc[180:,'bid']=.7;frame.loc[180:,'ask']=.72;frame.loc[180:,'midpoint']=.71
    frame.to_csv(path,index=False)
    manifest=json.loads((Path(source)/'manifest.json').read_text());manifest['files'][0]['sha256']=digest(path)
    save_json(Path(source)/'manifest.json',manifest)
    copied,_,_,_=hybrid.input_cohort(Path(source),[dict(target_end=record['final_end_ns'])],protocol,dict(start=1,end=2,sessions=[]))
    pd.testing.assert_frame_equal(copied,meta,check_exact=True)


@pytest.mark.parametrize('name',['protocol.json','model/best.keras'])
def test_frozen_hash_tampering_rejected(tmp_path,name):
    run,_=frozen(tmp_path)
    (run/name).write_text('changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):
        hybrid.verify(run)


def test_subthreshold_event_count_never_opens_outcomes(tmp_path,monkeypatch):
    run,protocol=frozen(tmp_path)
    record,_=source_fixture(run,protocol,events=19)
    monkeypatch.setattr(hybrid,'align_future',lambda *a,**k:pytest.fail('outcomes opened'))
    assert hybrid.materialize_outcomes(run,record) is None
    assert not (run/'final_labels_opened.json').exists()


def test_once_only_outcomes_actual_quotes_and_stale_readiness(tmp_path):
    run,protocol=frozen(tmp_path)
    record,meta=source_fixture(run,protocol)
    final=hybrid.materialize_outcomes(run,record)
    assert np.load(final/'future_valid.npy')[:,60].all()
    np.testing.assert_allclose(np.load(final/'future_quotes.npy')[:,60,:2],np.tile([.4,.42],(20,1)))
    with pytest.raises(ValueError,match='Once-only'):
        hybrid.materialize_outcomes(run,record)


def test_newly_consumed_period_invalidates_pending_inputs(tmp_path):
    run,protocol=frozen(tmp_path)
    record,_=source_fixture(run,protocol)
    other=tmp_path/'hybrid_other';other.mkdir()
    save_json(other/'final_labels_opened.json',{})
    save_json(other/'segments.json',dict(final_end_ns=record['final_end_ns']))
    with pytest.raises(ValueError,match='stale'):
        hybrid.once_gate(run,record)


def test_end_to_end_synthetic_evaluation_freezes_cohort_then_refuses_second_look(tmp_path,monkeypatch):
    run,protocol=frozen(tmp_path)
    source_fixture(run,protocol)
    monkeypatch.setattr(hybrid,'load',lambda folder:None)
    monkeypatch.setattr(hybrid,'predict',lambda model,h:np.full((len(h),2),-5.))
    metrics=hybrid.evaluate(run)
    assert metrics['hybrid']==metrics['nochange']
    assert metrics['hybrid']['observed_sides']==40
    completed=json.loads((run/'final_completed.json').read_text())
    assert completed['gru_routed']==0 and not completed['primary_gate_passes']
    assert not np.load(run/'final/gru_routing_mask.npy').any()
    with pytest.raises(ValueError,match='Once-only'):
        hybrid.evaluate(run)


@pytest.mark.parametrize('tamper',['protocol','refit_manifest','protection'])
def test_prior_control_or_protection_tamper_rejected_before_protocol_freeze(tmp_path,tamper):
    prior=tmp_path/'prior';prior.mkdir()
    source=tmp_path/'source';source.mkdir()
    protection=source/'protected_v2_period.json';protection.write_text('{}')
    save_json(prior/'segments.json',dict(source=str(source),final_end_ns=1))
    save_json(prior/'protocol.json',dict(segments_sha256=digest(prior/'segments.json'),
        source_inputs_sha256={'protected_v2_period.json':digest(protection)}))
    save_json(prior/'choices.json',dict(protocol_sha256=digest(prior/'protocol.json')))
    (prior/'choices.sha256').write_text(digest(prior/'choices.json'))
    save_json(prior/'refit_manifest.json',{})
    save_json(prior/'final_started.json',dict(choices_sha256=digest(prior/'choices.json'),refit_manifest_sha256=digest(prior/'refit_manifest.json')))
    for name in ('final_labels_opened.json','final_completed.json'):
        save_json(prior/name,dict(choices_sha256=digest(prior/'choices.json')))
    hybrid.prior_anchors(prior)
    changed={'protocol':prior/'protocol.json','refit_manifest':prior/'refit_manifest.json','protection':protection}[tamper]
    changed.write_text('{"changed":true}')
    output=tmp_path/'new'
    with pytest.raises(ValueError,match='Prior'):
        hybrid.freeze(output,prior)
    assert not (output/'protocol.json').exists()


def test_windows_refit_manifest_paths_are_portable_and_duplicates_rejected():
    expected='fixed-sha'
    assert hybrid.manifest_digest({'refits\\gru_60_seed42\\best.keras':expected},'refits/gru_60_seed42/best.keras')==expected
    with pytest.raises(ValueError,match='Duplicate normalized'):
        hybrid.manifest_digest({'refits\\x':'a','refits/x':'b'},'refits/x')
