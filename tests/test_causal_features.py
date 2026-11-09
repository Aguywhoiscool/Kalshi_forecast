import numpy as np
import pytest

from src.causal_features import (SEQUENCE_NAMES, build_features, context_indices,
    engineered_sequence, feature_schema, verified_timing)
from src.model_traces import simulate_mc


def book(length=60):
    mid=.4+np.sin(np.arange(length)*.6)*.03
    return np.column_stack([mid-.01,mid+.01,np.full(length,.02),mid,np.full(length,2.),np.full(length,3.),np.full(length,-.2)])


def test_engineered_values_and_schema_follow_known_quote_depth_changes():
    h=book();h[:,:4]=[.4,.44,.04,.42]
    h[20:,0]=.41;h[20:,2]=.03;h[20:,3]=.425
    h[30:,4]=4;h[30:,6]=1/7
    timing=(1800-np.arange(60))[None]
    seq,names=engineered_sequence(h[None],timing)
    assert seq.shape==(1,60,22) and names==SEQUENCE_NAMES
    assert seq[0,20,names.index('delta_bid')]==pytest.approx(.01)
    assert seq[0,20,names.index('quote_change_frequency_10')]==pytest.approx(1/9)
    assert seq[0,29,names.index('quote_change_frequency_10')]==0
    assert seq[0,30,names.index('delta_bid_depth')]==2
    assert seq[0,30,names.index('log1p_bid_depth')]==pytest.approx(np.log(5))
    assert seq[0,30,names.index('seconds_to_close')]==1770
    features,tab_names=build_features(h[None],timing)
    assert features.shape==(1,134) and len(tab_names)==134
    assert feature_schema()['sequence_names']==names


def test_future_row_changes_cannot_change_earlier_sequence_features():
    h=book()[None];timing=(1800-np.arange(60))[None]
    before,_=engineered_sequence(h,timing)
    changed=h.copy();changed[:,40:,:2]+=.01;changed[:,40:,3]+=.01
    after,_=engineered_sequence(changed,timing)
    np.testing.assert_array_equal(before[:,:40],after[:,:40])
    np.testing.assert_array_equal(h[0],book())


def test_missing_or_retrospectively_known_timing_is_rejected_exactly():
    stamps=np.arange(60,dtype=np.int64)*10**9+1794000000000000000
    closes=np.full(60,stamps[-1]+1800*10**9,dtype=np.int64)
    seconds=(closes-stamps)/1e9
    np.testing.assert_array_equal(verified_timing(stamps,closes,stamps,seconds),seconds)
    known=stamps.copy();known[3]+=1  # One nanosecond must not vanish into float64 rounding.
    with pytest.raises(ValueError,match='not known'):
        verified_timing(stamps,closes,known,seconds)
    with pytest.raises(ValueError,match='integer'):
        verified_timing(stamps.astype(float),closes,stamps,seconds)
    missing=seconds.copy();missing[0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        verified_timing(stamps,closes,stamps,missing)
    with pytest.raises(ValueError,match='no imputation'):
        engineered_sequence(book()[None],missing[None])
    earlier=stamps-10**9
    with pytest.raises(ValueError,match='Invalid countdown'):
        verified_timing(stamps.astype(np.uint64),earlier.astype(np.uint64),stamps.astype(np.uint64),np.full(60,float(2**64-10**9)/1e9))


def test_quote_validation_keeps_strict_positive_spread_and_bounded_imbalance():
    h=book()[None]
    h[0,0,:4]=[.4,.4000001,-.0000001,.40000005]
    with pytest.raises(ValueError,match='invalid quote'):
        engineered_sequence(h,np.full((1,60),1000.))
    h=book()[None];h[0,0,4:]=[2.,0.,1.000005]
    with pytest.raises(ValueError,match='invalid quote'):
        engineered_sequence(h,np.full((1,60),1000.))


def test_long_context_requires_same_session_continuous_past_and_matching_history():
    times=np.arange(240,dtype=np.int64)*10**9
    sessions=np.array(['s']*240)
    indices,good=context_indices(times,sessions,[179,200],180)
    assert good.tolist()==[True,True] and indices[0].tolist()==list(range(180))
    sessions[10]='other'
    _,good=context_indices(times,sessions,[179,200],180)
    assert good.tolist()==[False,True]
    times[100:]+=3*10**9
    indices,good=context_indices(times,sessions,[179,200],180)
    assert not good.any() and (indices==-1).all()
    _,missing=context_indices(np.arange(180)*10**9,np.array([None]*180),[179],180)
    assert not missing.any()
    context=book(180)[None];timing=(1800-np.arange(60))[None];h=context[:,-60:].copy()
    ordinary,names=build_features(h,timing)
    long,long_names=build_features(h,timing,context)
    np.testing.assert_array_equal(long[:,:134],ordinary)
    assert len(long_names)==136
    assert long[0,long_names.index('midpoint_momentum_180obs')]==pytest.approx(context[0,-1,3]-context[0,0,3])
    mismatch=context.copy();mismatch[0,-1,:2]+=.001;mismatch[0,-1,3]+=.001
    with pytest.raises(ValueError,match='exact matching'):
        build_features(h,timing,mismatch)


def test_mc180_version_reconstructs_real_draws_and_uses_matching_context():
    context=book(180);h=context[-60:].copy();identity={'session_id':'s','ticker':'T','input_end':123}
    baseline=simulate_mc(h,identity,paths=64,trace=True)
    long=simulate_mc(h,identity,paths=64,trace=True,context=context)
    plain=simulate_mc(h,identity,paths=64,context=context)
    assert long['predictions']==plain['predictions']
    assert long['seed']==baseline['seed'] and long['method_version']=='bootstrap180_v1'
    assert len(long['historical_increments'])==179 and len(baseline['historical_increments'])==59
    actual=np.asarray(long['paths']);draws=np.asarray(long['sampled_indices']);inc=np.diff(context[:,3])
    for step in range(60):
        expected=np.maximum(.01,np.minimum(.99,actual[:,step]+inc[draws[:,step]]))
        np.testing.assert_allclose(actual[:,step+1],expected,atol=1e-12)
    for horizon in (30,60):
        yes=(actual[:,horizon]-.01-(actual[:,1]+.01))*100-4
        no=((1-actual[:,horizon]-.01)-(1-actual[:,1]+.01))*100-4
        np.testing.assert_allclose(long['per_path_returns'][horizon],np.column_stack([yes,no]),atol=1e-10)
        np.testing.assert_allclose(long['predictions'][horizon],[yes.mean(),no.mean()])
    mismatch=context.copy();mismatch[-1,3]+=.001
    with pytest.raises(ValueError,match='exact baseline'):
        simulate_mc(h,identity,context=mismatch)
    invalid=context.copy();invalid[0,0]=-.1
    with pytest.raises(ValueError,match='invalid quote'):
        simulate_mc(h,identity,context=invalid)


def test_flat_context_remains_flat_in_both_mc_versions():
    context=np.tile([.4,.42,.02,.41,2.,3.,-.2],(180,1));h=context[-60:]
    identity={'session_id':'s','ticker':'T','input_end':1}
    for kwargs in ({},{'context':context}):
        trace=simulate_mc(h,identity,paths=32,trace=True,**kwargs)
        np.testing.assert_array_equal(np.asarray(trace['paths']),np.full((32,61),.41))
        np.testing.assert_allclose(trace['predictions'][30],[-6.,-6.])
