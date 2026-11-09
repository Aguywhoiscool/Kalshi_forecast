import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.accuracy_audit import (audit,candidate_specs,check_hashes,check_partition,
    check_values,metrics,prescribed_correction,targets,check_execution,check_choices,check_identity_metadata)
from src.v3_research import digest


def test_hash_manifest_rejects_tampered_weights_and_path_escape(tmp_path):
    path=tmp_path/'weights.bin';path.write_bytes(b'frozen')
    manifest={'weights.bin':digest(path)}
    check_hashes(tmp_path,manifest)
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'): check_hashes(tmp_path,manifest)
    with pytest.raises(ValueError,match='hash mismatch'): check_hashes(tmp_path,{'../outside':'0'})


def test_fold_guard_rejects_temporal_or_session_leakage():
    meta=pd.DataFrame({'session_id':['a','b','c'],'input_start':[0,200,400], 'target_end':[100,300,500]})*1
    meta[['input_start','target_end']]*=10**9
    fold={'fit':np.array([0]),'calibration':np.array([1]),'validation':np.array([2])}
    check_partition(meta,fold)
    invalid=meta.copy();invalid.loc[2,'input_start']=350*10**9
    with pytest.raises(ValueError,match='embargo'): check_partition(invalid,fold)
    invalid=meta.copy();invalid.loc[2,'session_id']='a'
    with pytest.raises(ValueError,match='sessions overlap'): check_partition(invalid,fold)


def test_metrics_retain_unknown_intervals_and_rejected_zero_as_observed():
    prediction=np.array([[1.,2.],[10.,20.]])
    lower=np.array([[0.,0.],[-104.,-46.]])
    upper=np.array([[0.,0.],[96.,54.]])
    status=np.array([[0,1],[2,3]])
    calculated=metrics(prediction,lower,upper,status)
    assert calculated['observed_mae']==1.5 and calculated['closed_mae']==1.
    assert calculated['full_lower_bound_mae']==45.75
    assert calculated['interval_distance_mae']==.75
    assert calculated['observed_sides']==2 and calculated['unknown_sides']==2
    tampered=dict(calculated,observed_mae=0.)
    with pytest.raises(ValueError,match='Mismatch'): check_values(tampered,calculated,'saved')


def test_calibration_objectives_recomputed_from_known_math():
    raw=np.array([[10.,-10.],[2.,4.]])
    lower=np.array([[0.,-2.],[0.,0.]])
    upper=np.array([[5.,2.],[4.,8.]])
    np.testing.assert_array_equal(prescribed_correction(raw,lower,upper,'interval'),[-2.5,4.])
    np.testing.assert_array_equal(prescribed_correction(raw,lower,upper,'conservative'),[-6.,2.])


def test_independent_book_targets_preserve_unknown_and_rejected_execution():
    quotes=np.tile([.4,.42,2.,3.],(2,61,1));known=np.ones((2,61),bool)
    known[0,30]=False;quotes[0,30]=np.nan;quotes[1,1,3]=.5
    lo,hi,status=targets(SimpleNamespace(future=quotes,known=known),30)
    np.testing.assert_allclose(lo[0],[-46,-64]);np.testing.assert_allclose(hi[0],[54,36])
    assert status[0].tolist()==[3,3] and status[1,0]==1
    assert lo[1,0]==hi[1,0]==0


def test_final_quarantine_blocks_before_any_partition_read(tmp_path):
    # No protocol or data exists: must fail on the final gate, not try opening it.
    with pytest.raises(ValueError,match='Final completion required'):
        audit(tmp_path,allow_final=True)


def test_exact_ninety_candidate_registry_with_versioned_families():
    protocol={'families':['seven60','engineered60','engineered180'],
        'defaults':{m:{} for m in ['ridge','sgd','gradient_boosting','gru','mean','decision_tree','random_forest','monte_carlo_60','monte_carlo_180']},
        'grids':{m:[{'index':i} for i in range(n)] for m,n in [('ridge',4),('sgd',4),('gradient_boosting',8),('gru',6)]}}
    specs=candidate_specs(protocol)
    assert len(specs)==90
    assert specs['default_monte_carlo_180_60']['model']=='monte_carlo_180'
    assert specs['ablation_gru_30_engineered180_interval']['family']=='engineered180'


def test_execution_replay_keeps_unknown_position_blocked_and_rejects_fabricated_close():
    clocks=np.array([0,120*10**9])
    meta=pd.DataFrame({'session_id':['s','s'],'ticker':['T','T'],'input_end':clocks})
    history=np.tile([.4,.42,.02,.41,2.,3.,-.2],(2,180,1))
    prediction=np.array([[2.,-1.],[2.,-1.]])
    lower=np.array([[-46.,-64.],[-46.,-64.]]);upper=np.array([[54.,36.],[54.,36.]])
    status=np.full((2,2),3)
    trades=pd.DataFrame({'session_id':['s','s'],'ticker':['T','T'],'input_end':clocks,
        'side':[1,1],'status':['missing_exit_quote','blocked_uncertain_or_pending'],
        'predicted_net_cents':[2.,2.],'net_lower_cents':[-46.,0.],'net_upper_cents':[54.,0.],
        'planned_exit_time_ns':clocks+30*10**9})
    result=check_execution(trades,prediction,meta,history,lower,upper,status,'ridge',30)
    assert result['attempts']==1 and result['blocked']==1 and result['unresolved_exits']==1
    assert result['net_lower_total_cents']==-46 and result['net_upper_total_cents']==54
    trades.loc[1,'status']='closed'
    with pytest.raises(ValueError,match='state'): check_execution(trades,prediction,meta,history,lower,upper,status,'ridge',30)


def test_frozen_choices_reject_self_consistent_hash_but_wrong_winner(tmp_path):
    models=['ridge','sgd','gradient_boosting','gru']
    protocol={'families':['seven60','engineered60','engineered180'],
        'defaults':{m:{} for m in [*models,'mean','decision_tree','random_forest','monte_carlo_60','monte_carlo_180']},
        'grids':{m:[{'index':i} for i in range(n)] for m,n in [('ridge',4),('sgd',4),('gradient_boosting',8),('gru',6)]}}
    summaries={}
    for name,spec in candidate_specs(protocol).items():
        row={**spec,'id':name,'mean_observed_mae':10.}
        if name.startswith('ablation_'):
            row['mean_observed_mae']={'seven60':10.,'engineered60':8.,'engineered180':9.}[row['family']]-(1 if row['objective']=='interval' else 0)
        if name.startswith('search_'):
            row.update(family='engineered60',objective='interval' if row['model'] in ('sgd','gru') else 'conservative',
                mean_observed_mae=6.-row['params']['index']*.1)
        summaries[name]=row
    (tmp_path/'development').mkdir();(tmp_path/'development/metadata.parquet').write_bytes(b'fixed identity bytes')
    (tmp_path/'protocol.json').write_text(json.dumps(protocol));(tmp_path/'segments.json').write_text('{}')
    selections={f'{m}_{h}':summaries[f'search_{m}_{h}_{len(protocol["grids"][m])-1}'] for m in models for h in (30,60)}
    choices={'selections':selections,'primary_baselines':{str(h):summaries[f'default_mean_{h}'] for h in (30,60)},
        'primary_winners':{str(h):selections[f'gradient_boosting_{h}'] for h in (30,60)},
        'protocol_sha256':digest(tmp_path/'protocol.json'),'segments_sha256':digest(tmp_path/'segments.json'),
        'development_metadata_sha256':digest(tmp_path/'development/metadata.parquet')}
    def save_choices():
        (tmp_path/'choices.json').write_text(json.dumps(choices))
        (tmp_path/'choices.sha256').write_text(digest(tmp_path/'choices.json'))
    save_choices();assert check_choices(tmp_path,protocol,summaries)
    choices['primary_winners']['30']=selections['ridge_30'];save_choices()
    with pytest.raises(ValueError,match='minima'): check_choices(tmp_path,protocol,summaries)


def test_event_group_counts_cannot_be_inflated_by_forged_proxy_labels():
    meta=pd.DataFrame({'session_id':['s','s'],'ticker':['SERIES-EVENT-YES','SERIES-EVENT-NO'],
        'event_proxy':['SERIES-EVENT','SERIES-EVENT'],'series':['SERIES','SERIES']})
    check_identity_metadata(meta)
    meta.loc[1,'event_proxy']='invented-separate-event'
    with pytest.raises(ValueError,match='Event proxies'): check_identity_metadata(meta)
