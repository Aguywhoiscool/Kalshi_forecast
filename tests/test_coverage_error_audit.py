import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.coverage_error_audit import window_masks,error_stats,load_oof,reconstruct
from src.v3_research import digest


def group(n=400):
    stamp=pd.date_range('2026-11-01',periods=n,freq='s',tz='UTC')
    close=stamp[0]+pd.Timedelta(days=1)
    return pd.DataFrame(dict(timestamp=stamp,session_id='fixture',ticker='EVENT-A',bid=.4,ask=.42,
        spread=.02,midpoint=.41,bid_depth=100.,ask_depth=100.,imbalance=0.,
        scheduled_close_time=close.isoformat(),close_time_observed_at=stamp[0].isoformat(),
        seconds_to_scheduled_close=(close-stamp).total_seconds()))


def protected():
    return dict(start=1,end=2,sessions=[])


def test_sequential_partition_and_overlapping_flags_are_distinct():
    frame=group(300)
    frame.loc[50,'ask']=.3
    frame.loc[50,'seconds_to_scheduled_close']=np.nan
    bound=frame.timestamp.iloc[-1].value+1000*10**9
    stages,flags,_=window_masks(frame,180,True,bound,protected())
    assert flags['quote_invalid'][179] and flags['timing_missing'][179]
    assert stages['excluded_quote_invalid'][179]
    assert not stages['excluded_timing_missing'][179]
    excluded=sum(int(v.sum()) for k,v in stages.items() if k.startswith('excluded_'))
    assert excluded+stages['selected_spaced'].sum()==len(frame)
    assert stages['selected_spaced'][230]


def test_gap_and_reserved_overlap_change_only_eligible_histories():
    frame=group()
    frame.loc[200:,'timestamp']+=pd.Timedelta(seconds=4)
    bound=frame.timestamp.iloc[-1].value+1000*10**9
    stages,_,_=window_masks(frame,180,False,bound,protected())
    np.testing.assert_array_equal(np.flatnonzero(stages['selected_spaced']),[179,379])
    reserved=dict(start=frame.timestamp.iloc[250].value,end=frame.timestamp.iloc[255].value,sessions=[])
    bounded,flags,_=window_masks(frame,180,False,bound,reserved)
    assert flags['protected_overlap'][299]
    assert not bounded['selected_spaced'][299]
    assert bounded['selected_spaced'][179]


@pytest.mark.parametrize('length,verified',[(60,False),(60,True),(180,False),(180,True)])
def test_future_price_or_recording_end_does_not_filter_current_input(length,verified):
    full=group(360)
    short=full.iloc[:180].copy()
    changed=full.copy()
    changed.loc[180:,'bid']=.7;changed.loc[180:,'ask']=.72;changed.loc[180:,'midpoint']=.71
    bound=full.timestamp.iloc[-1].value+1000*10**9
    a,_,_=window_masks(full,length,verified,bound,protected())
    b,_,_=window_masks(short,length,verified,bound,protected())
    c,_,_=window_masks(changed,length,verified,bound,protected())
    for key in a:
        np.testing.assert_array_equal(a[key][:180],b[key])
        np.testing.assert_array_equal(a[key][:180],c[key][:180])
    assert a['eligible_before_spacing'][179]


def test_error_metrics_and_oof_identity_guard(tmp_path):
    prediction=np.array([[1.,-3.],[90.,90.]])
    lower=np.array([[0.,0.],[10.,10.]])
    status=np.array([[0,1],[2,4]])
    stats=error_stats(prediction,np.zeros_like(prediction),lower,np.full((2,2),100.),status,['a','b'])
    assert stats['observed_mae']==2 and stats['signed_bias']==-1
    assert stats['median_absolute_error']==2 and stats['p90_absolute_error']==pytest.approx(2.8)
    assert stats['signed_paired_mae_difference']==2
    assert stats['unknown_sides']==2 and stats['rejected_sides']==1
    assert stats['observed_event_proxies']==1 and stats['low_observed_event_support']
    path=tmp_path/'oof.npz'
    np.savez(path,rows=[4,2],prediction=prediction)
    with pytest.raises(AssertionError):
        load_oof(path,np.array([2,4]))


def source_fixture(tmp_path):
    source,assignment,output=[tmp_path/n for n in ('source','assignment','output')]
    for p in (source,source/'snapshot',assignment,assignment/'development',output):p.mkdir()
    frame=group(300)
    session='fixture'
    frame.to_csv(source/'snapshot'/f'features_{session}.csv',index=False)
    (source/'protected_v2_period.json').write_text(json.dumps(protected()))
    manifest=dict(files=[dict(name=f'features_{session}.csv',sha256=digest(source/'snapshot'/f'features_{session}.csv'))])
    (source/'input_manifest.json').write_text(json.dumps(manifest))
    (source/'development_source.json').write_text(json.dumps(dict(snapshot_rows=300)))
    sessions=pd.DataFrame(dict(session_id=[session],source_rows=[300]))
    sessions.to_parquet(source/'windows.parquet',index=False)
    sessions.to_parquet(assignment/'development_sessions.parquet',index=False)
    ends=np.array([179,299]);stamps=frame.timestamp.array.as_unit('ns').asi8
    expected=pd.DataFrame(dict(session_id=session,ticker='EVENT-A',event_proxy='EVENT',series='EVENT',
        input_start=stamps[ends-179],input_end=stamps[ends],target_end=stamps[ends]+60_500_000_000,
        seconds_to_close=frame.seconds_to_scheduled_close.to_numpy()[ends]))
    expected.to_parquet(assignment/'development/metadata.parquet',index=False)
    (assignment/'segments.json').write_text(json.dumps(dict(source=str(source.resolve()),
        source_manifest_sha256=digest(source/'input_manifest.json'),development_metadata_sha256=digest(assignment/'development_sessions.parquet'),
        development_end_ns=int(stamps[-1])+1000*10**9)))
    return source,assignment,output,expected


def test_actual_reconstruction_exact_identity_and_no_future_coverage_filter(tmp_path):
    source,assignment,output,expected=source_fixture(tmp_path)
    actual=reconstruct(source,assignment,output)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
    assert actual.input_end.iloc[-1]+60_500_000_000>pd.Timestamp('2026-11-01T00:04:59Z').value
    row=pd.read_csv(output/'sequential_funnel.csv')
    assert (row.loc[(row.unit=='recorded_rows')&(row.stage=='source_rows'),'count']==300).all()


def test_actual_reconstruction_rejects_reserved_session_before_price_read(tmp_path):
    source,assignment,output,_=source_fixture(tmp_path)
    reserved=protected();reserved['sessions']=['fixture']
    (source/'protected_v2_period.json').write_text(json.dumps(reserved))
    (source/'snapshot/features_fixture.csv').unlink()
    with pytest.raises(ValueError,match='Reserved session'):
        reconstruct(source,assignment,output)
