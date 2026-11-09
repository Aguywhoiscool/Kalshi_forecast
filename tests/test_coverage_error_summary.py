import numpy as np
import pandas as pd
import pytest
from src.coverage_error_summary import movement_groups,pool_side_groups,summarize,spread_bins


def test_movement_thresholds_and_unknown_are_only_diagnostic():
    future=np.zeros((4,61,4))
    future[:,:,0]=.4;future[:,:,1]=.42
    future[1,30,:2]=[.405,.425]
    future[2,30,:2]=[.41,.43]
    known=np.ones((4,61),bool);known[3,30]=False
    np.testing.assert_array_equal(movement_groups(future,known,30),
        ['unchanged','positive_under1c','at_least1c','unknown'])


def test_pooled_pairs_count_candidate_once_and_observed_events_not_sides():
    frame=pd.DataFrame(dict(horizon=[30]*4,model=['winner']*4,original_row=[8,8,9,9],side=[0,1,0,1],
        event_proxy=['event']*4,prediction_cents=[1,-3,90,90],baseline_cents=[0]*4,
        lower_cents=[0,0,10,10],upper_cents=[100]*4,status=[0,1,2,4],spread_group=['0']*4))
    result=pool_side_groups(frame).iloc[0]
    assert result.candidates==2 and result.observed_sides==2
    assert result.observed_event_proxies==1 and result.low_observed_event_support
    assert result.observed_mae==2 and result.signed_bias==-1
    assert result.signed_paired_mae_difference==2
    frame.loc[3,'side']=0
    with pytest.raises(ValueError,match='unpaired'):
        pool_side_groups(frame)


def test_summary_refuses_before_completed_audit_without_data_read(tmp_path):
    (tmp_path/'audit.json').write_text('{"status":"failed","scope":"authorized_development_only"}')
    with pytest.raises(ValueError,match='Completed development audit'):
        summarize(tmp_path,tmp_path/'missing',tmp_path/'missing',tmp_path/'missing')


def test_spread_equal_price_ticks_share_fit_bin_despite_float_subtraction():
    history=np.zeros((8,60,7))
    history[:,-1,0]=[.40,.50,.20,.30,.40,.50,.60,.70]
    history[:,-1,1]=history[:,-1,0]+np.array([.01,.01,.02,.02,.04,.04,.04,.04])
    bins,edges=spread_bins(history,np.arange(6),np.array([4,5,6,7]))
    assert len(set(bins))==1
    assert edges[-1]==3.5
