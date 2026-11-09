"""Offline acceptance checks; final arrays remain closed until explicit completion."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.accuracy_models import Data,load_model,raw_predict
from src.accuracy_targets import event_bootstrap
from src.audit_swing_returns import independent_bounds
from src.v3_research import FEATURES,digest,overlaps,save_json,valid_rows
from src.model_traces import simulate_mc


MODELS=("ridge","sgd","gradient_boosting","gru")
CONTROLS=("mean","decision_tree","random_forest","monte_carlo_60","monte_carlo_180")


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def check_hashes(root,manifest):
    for name,expected in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path)!=expected:
            raise ValueError('Frozen hash mismatch: '+name)


def check_partition(meta,fold):
    for name,indices in fold.items():
        ids=np.asarray(indices)
        if ids.ndim!=1 or not len(ids) or (ids<0).any() or (ids>=len(meta)).any() or len(np.unique(ids))!=len(ids):
            raise ValueError('Invalid fold indices: '+name)
    for left,right in (('fit','calibration'),('calibration','validation'),('fit','validation')):
        a,b=meta.iloc[fold[left]],meta.iloc[fold[right]]
        if set(a.session_id)&set(b.session_id) or np.intersect1d(fold[left],fold[right]).size:
            raise ValueError('Fold sessions overlap')
        if b.input_start.min()<a.target_end.max()+70*10**9:
            raise ValueError('Fold chronological embargo violated')


def expected_folds(meta):
    sessions=meta.groupby('session_id').input_start.min().sort_values()
    result=[]
    for fraction in (.4,.6,.8):
        a=int(len(sessions)*fraction);b=min(len(sessions),int(len(sessions)*(fraction+.2)+1e-8))
        train_sessions=set(sessions.index[:a]);validation_sessions=set(sessions.index[a:b])
        fit_sessions=set(sessions.index[:int(a*.8)])
        fit=np.flatnonzero(meta.session_id.isin(fit_sessions))
        train=np.flatnonzero(meta.session_id.isin(train_sessions))
        if not len(fit) or not len(train): raise ValueError('Insufficient predeclared fold sessions')
        calibration=np.flatnonzero(meta.session_id.isin(train_sessions-fit_sessions)&(meta.input_start>=meta.iloc[fit].target_end.max()+70*10**9))
        validation=np.flatnonzero(meta.session_id.isin(validation_sessions)&(meta.input_start>=meta.iloc[train].target_end.max()+70*10**9))
        result.append({'fit':fit,'calibration':calibration,'validation':validation})
    return result


def targets(data,horizon):
    lower,upper,states=independent_bounds(data.future,data.known,horizon)
    codes={'closed':0,'rejected_entry':1,'unknown_entry':2,'missing_exit_quote':3,'insufficient_exit_depth':4}
    return lower,upper,np.vectorize(codes.__getitem__)(states)


def metrics(prediction,lower,upper,status):
    prediction=np.asarray(prediction,float)
    if prediction.shape!=lower.shape or not np.isfinite(prediction).all() or (prediction<-104-1e-7).any() or (prediction>96+1e-7).any():
        raise ValueError('Invalid prediction shape or finite values')
    observed=np.isin(status,[0,1]);closed=status==0
    errors=np.abs(prediction-lower)
    distance=np.abs(prediction-np.minimum(upper,np.maximum(lower,prediction)))
    return {'observed_mae':float(errors[observed].mean()) if observed.any() else None,
        'closed_mae':float(errors[closed].mean()) if closed.any() else None,
        'observed_sides':int(observed.sum()),'closed_sides':int(closed.sum()),
        'rejected_sides':int((status==1).sum()),'total_sides':int(status.size),
        'observed_coverage':float(observed.mean()),'full_lower_bound_mae':float(errors.mean()),
        'interval_distance_mae':float(distance.mean()),'unknown_sides':int((status>=2).sum())}


def check_values(actual,expected,label):
    for key,value in expected.items():
        if key not in actual: raise ValueError('Missing '+label+'.'+key)
        if isinstance(value,dict):
            check_values(actual[key],value,label+'.'+key)
        elif value is None or isinstance(value,(str,bool)):
            if actual.get(key)!=value: raise ValueError('Mismatch '+label+'.'+key)
        elif isinstance(value,list) and any(v is None for v in value):
            if actual[key]!=value: raise ValueError('Mismatch '+label+'.'+key)
        elif isinstance(value,(list,int,float,np.number)):
            try: np.testing.assert_allclose(actual[key],value,atol=1e-7,rtol=1e-6)
            except (AssertionError,KeyError,TypeError) as error: raise ValueError('Mismatch '+label+'.'+key) from error


def prescribed_correction(raw,lower,upper,objective):
    if objective not in ('conservative','interval'): raise ValueError('Unknown objective')
    destination=lower if objective=='conservative' else np.minimum(upper,np.maximum(lower,raw))
    return np.mean(destination-raw,axis=0)


def check_identity_metadata(meta):
    if meta[['session_id','ticker','event_proxy','series']].isna().any().any():
        raise ValueError('Missing source identity metadata')
    tickers=meta.ticker.astype(str)
    if not meta.event_proxy.astype(str).eq(tickers.str.rsplit('-',n=1).str[0]).all():
        raise ValueError('Event proxies disagree with source tickers')
    if not meta.series.astype(str).eq(tickers.str.split('-').str[0]).all():
        raise ValueError('Series disagree with source tickers')


def candidate_specs(protocol):
    expected={}
    for horizon in (30,60):
        for model in CONTROLS:
            expected[f'default_{model}_{horizon}']={'model':model,'horizon':horizon,'family':'seven60','objective':'conservative','params':protocol['defaults'][model]}
        for model in MODELS:
            for family in protocol['families']:
                for objective in (('conservative','interval') if model in ('sgd','gru') else ('conservative',)):
                    expected[f'ablation_{model}_{horizon}_{family}_{objective}']={'model':model,'horizon':horizon,'family':family,'objective':objective,'params':protocol['defaults'][model]}
            for index,params in enumerate(protocol['grids'][model]):
                expected[f'search_{model}_{horizon}_{index}']={'model':model,'horizon':horizon,'params':params}
    return expected


def source_replay(data,source,scope,samples=3):
    """Replay full180 prefixes/timing and scheduled future quotes for sampled rows."""
    meta=data.meta
    check_identity_metadata(meta)
    protected=read(source/'protected_v2_period.json')
    if meta.session_id.isin(protected['sessions']).any() or overlaps(meta.input_start.to_numpy(),meta.target_end.to_numpy(),protected).any():
        raise ValueError('Reserved scope overlap')
    manifest={item['name']:item['sha256'] for item in read(source/'input_manifest.json')['files']}
    ids=np.arange(len(meta)) if samples is None else np.unique(np.linspace(0,len(meta)-1,min(samples,len(meta)),dtype=int))
    loaded={}
    for index in ids:
        row=meta.iloc[index];name=f'features_{row.session_id}.csv';path=source/'snapshot'/name
        check_hashes(source/'snapshot',{name:manifest[name]})
        if name not in loaded:
            frame=pd.read_csv(path)
            frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True,errors='coerce',format='ISO8601')
            for column in FEATURES: frame[column]=pd.to_numeric(frame[column],errors='coerce')
            loaded[name]=frame.dropna(subset=['timestamp']).drop_duplicates(['session_id','ticker','timestamp'])
        frame=loaded[name]
        group=frame.loc[(frame.session_id==row.session_id)&(frame.ticker==row.ticker)].sort_values('timestamp').reset_index(drop=True)
        clocks=group.timestamp.array.as_unit('ns').asi8
        end=int(np.searchsorted(clocks,row.input_end))
        if end<179 or end>=len(group) or clocks[end]!=row.input_end: raise ValueError('Source input identity mismatch')
        indices=np.arange(end-179,end+1)
        np.testing.assert_array_equal(data.history[index],group[FEATURES].to_numpy(float)[indices])
        if clocks[indices[0]]!=row.input_start or not ((np.diff(clocks[indices])>=.5e9)&(np.diff(clocks[indices])<=1.5e9)).all():
            raise ValueError('Source history clocks mismatch')
        closes=pd.to_datetime(group.scheduled_close_time,utc=True,format='ISO8601',errors='coerce').array.as_unit('ns').asi8[indices]
        known=pd.to_datetime(group.close_time_observed_at,utc=True,format='ISO8601',errors='coerce').array.as_unit('ns').asi8[indices]
        seconds=pd.to_numeric(group.seconds_to_scheduled_close,errors='coerce').to_numpy()[indices]
        if (known<=0).any() or (closes<=0).any() or (known>clocks[indices]).any() or not np.isfinite(seconds).all():
            raise ValueError('Timing not known in source')
        np.testing.assert_allclose((closes-clocks[indices])/1e9,seconds,atol=.001,rtol=0)
        np.testing.assert_array_equal(data.timing[index],seconds)
        desired=row.input_end+np.arange(61)*10**9
        right=np.minimum(np.searchsorted(clocks,desired),len(clocks)-1);left=np.maximum(0,np.searchsorted(clocks,desired)-1)
        chosen=np.where(np.abs(clocks[left]-desired)<=np.abs(clocks[right]-desired),left,right)
        available=(np.abs(clocks[chosen]-desired)<=.5e9)&valid_rows(group)[chosen]
        expected=np.where(available[:,None],group[['bid','ask','bid_depth','ask_depth']].to_numpy(float)[chosen],np.nan)
        np.testing.assert_array_equal(data.known[index],available)
        np.testing.assert_allclose(data.future[index],expected,equal_nan=True,atol=0,rtol=0)
    return len(ids)


def replay_model(folder,summary,data,fold,saved):
    fitted=load_model(folder,summary['model'],summary['family']);fitted['horizon']=summary['horizon']
    for name,key in (('calibration','raw_calibration'),('validation','raw_validation')):
        rows=fold[name];positions=np.unique(np.linspace(0,len(rows)-1,min(3,len(rows)),dtype=int))
        np.testing.assert_allclose(raw_sample(fitted,data,rows[positions]),saved[key][positions],atol=1e-4,rtol=1e-5)
    scaler=fitted.get('scaler')
    if fitted['kind']=='tabular' and hasattr(fitted['model'],'named_steps'):
        scaler=fitted['model'].named_steps.get('standardscaler')
    if scaler is not None:
        values=data.sequence(summary['family'])[fold['fit']].reshape(-1,scaler.n_features_in_) if fitted['kind']=='gru' else data.table(summary['family'])[fold['fit']]
        if scaler.n_samples_seen_!=len(values): raise ValueError('Scaler count is not fit-only')
        np.testing.assert_allclose(scaler.mean_,values.mean(axis=0,dtype=np.float64),atol=1e-6,rtol=1e-6)


def raw_sample(fitted,data,rows):
    if fitted['kind']!='monte_carlo': return raw_predict(fitted,data,rows)
    values=[]
    for index in rows:
        row=data.meta.iloc[index]
        identity={'session_id':row.session_id,'ticker':row.ticker,'input_end':int(row.input_end)}
        result=simulate_mc(data.history[index,-60:],identity,
            context=data.history[index] if fitted['model_name']=='monte_carlo_180' else None)
        values.append(result['predictions'][fitted['horizon']])
    return np.clip(values,-104,96)


def check_execution(trades,prediction,meta,history,lower,upper,status,model,horizon):
    scores=np.asarray(prediction).copy()
    scores[history[:,-1,5]<1,0]=-np.inf;scores[history[:,-1,4]<1,1]=-np.inf
    sides=scores.argmax(axis=1);best=scores[np.arange(len(scores)),sides]
    chosen=np.flatnonzero((best>=1)&(scores[:,0]!=scores[:,1]))
    if len(trades)!=len(chosen): raise ValueError('Trade population lost or added signals')
    blocked=set();busy={}
    states={0:'closed',1:'rejected_entry',2:'unknown_entry',3:'missing_exit_quote',4:'insufficient_exit_depth'}
    for index,trade in zip(chosen,trades.itertuples()):
        row=meta.iloc[index];side=int(sides[index]);stamp=int(row.input_end)
        if (trade.session_id,trade.ticker,trade.input_end)!=(row.session_id,row.ticker,stamp): raise ValueError('Trade source identity mismatch')
        cannot=row.ticker in blocked or stamp<=busy.get(row.ticker,-1)
        state='blocked_uncertain_or_pending' if cannot else states[status[index,side]]
        if not cannot:
            busy[row.ticker]=stamp+horizon*10**9+500000000
            if status[index,side]>=2: blocked.add(row.ticker)
        expected=[0.,0.] if cannot else [lower[index,side],upper[index,side]]
        np.testing.assert_allclose([trade.net_lower_cents,trade.net_upper_cents],expected,atol=1e-7)
        np.testing.assert_allclose(trade.predicted_net_cents,best[index],atol=1e-7)
        if trade.side!=side+1 or trade.status!=state or trade.planned_exit_time_ns!=stamp+horizon*10**9:
            raise ValueError('Trade side, state or planned clock mismatch')
    active=trades.loc[trades.status!='blocked_uncertain_or_pending'];closed=active.loc[active.status=='closed']
    return {'model':model,'horizon':horizon,'margin_cents':1.,'cost_buffer_cents':4.,
        'signals':len(trades),'blocked':len(trades)-len(active),'attempts':len(active),'closed':len(closed),
        'rejected_entries':int(active.status.eq('rejected_entry').sum()),'unknown_entries':int(active.status.eq('unknown_entry').sum()),
        'unresolved_exits':int(active.status.isin(['missing_exit_quote','insufficient_exit_depth']).sum()),
        'net_lower_total_cents':float(active.net_lower_cents.sum()),'net_upper_total_cents':float(active.net_upper_cents.sum()),
        'closed_quote_net_cents':float(closed.net_lower_cents.sum())}


def check_choices(output,protocol,summaries):
    if not (output/'choices.json').exists(): return False
    if set(summaries)!=set(candidate_specs(protocol)): raise ValueError('Incomplete predeclared candidate grid')
    choices=read(output/'choices.json')
    if digest(output/'choices.json')!=(output/'choices.sha256').read_text(): raise ValueError('Choices hash mismatch')
    for key,name in (('protocol_sha256','protocol.json'),('segments_sha256','segments.json'),('development_metadata_sha256','development/metadata.parquet')):
        if choices[key]!=digest(output/name): raise ValueError('Choices dependency hash mismatch')
    selected={};baselines={};winners={}
    for horizon in (30,60):
        defaults=[summaries[f'default_{model}_{horizon}'] for model in CONTROLS]
        for model in MODELS:
            candidates=[summaries[f'ablation_{model}_{horizon}_{family}_{objective}'] for family in protocol['families']
                for objective in (('conservative','interval') if model in ('sgd','gru') else ('conservative',))]
            chosen=min(candidates,key=lambda r:r['mean_observed_mae'])
            defaults.append(summaries[f'ablation_{model}_{horizon}_seven60_conservative'])
            tuned=[]
            for i,params in enumerate(protocol['grids'][model]):
                row=summaries[f'search_{model}_{horizon}_{i}']
                if row['params']!=params or row['family']!=chosen['family'] or row['objective']!=chosen['objective']:
                    raise ValueError('Search departed from predeclared staged choices')
                tuned.append(row)
            selected[f'{model}_{horizon}']=min(tuned,key=lambda r:r['mean_observed_mae'])
        baselines[str(horizon)]=min(defaults,key=lambda r:r['mean_observed_mae'])
        winners[str(horizon)]=min([selected[f'{m}_{horizon}'] for m in MODELS],key=lambda r:r['mean_observed_mae'])
    if choices['selections']!=selected or choices['primary_baselines']!=baselines or choices['primary_winners']!=winners:
        raise ValueError('Frozen winners do not reproduce development minima')
    return True


def audit(output,allow_final=False):
    output=Path(output).resolve();root=Path(__file__).resolve().parents[1]
    if allow_final and not (output/'final_completed.json').exists():
        raise ValueError('Final completion required before audit opens final arrays')
    protocol=read(output/'protocol.json');segments=read(output/'segments.json');source=Path(segments['source'])
    check_hashes(root/'src',protocol['implementation_sha256'])
    dependencies=read(output/'dependency_audit_manifest.json')
    check_hashes(root/'src',{name:value['sha256'] for name,value in dependencies['dependencies'].items()})
    check_hashes(output/'development',protocol['development_artifacts_sha256'])
    prior=protocol.get('frozen_old_secondary')
    if prior: check_hashes(Path(prior['run']),prior['manifest'])
    check_hashes(source,{'input_manifest.json':segments['source_manifest_sha256']})
    for scope in ('development','final'):
        check_hashes(output,{f'{scope}_sessions.parquet':segments[f'{scope}_metadata_sha256']})
    if segments['final_cutoff_ns']<segments['freshness_lower_bound_ns']+70*10**9:
        raise ValueError('Final period is not fresh and embargoed')
    for prior in segments.get('previously_opened_final_registry',[]):
        previous=Path(prior['run']);check_hashes(previous,{'final_labels_opened.json':prior['opened_marker_sha256'],'segments.json':prior['segments_sha256']})
    data=Data(output/'development');sampled=source_replay(data,source,'development')
    allowed=set(pd.read_parquet(output/'development_sessions.parquet').session_id)
    quarantined=set(pd.read_parquet(output/'final_sessions.parquet').session_id)
    if allowed&quarantined or not set(data.meta.session_id)<=allowed or data.meta.target_end.max()>segments['development_end_ns']:
        raise ValueError('Development/final quarantine violated')
    declared_folds=expected_folds(data.meta)
    folds=[]
    for index in range(3):
        fold={name:np.load(output/'folds'/str(index)/(name+'.npy')) for name in ('fit','calibration','validation')}
        for name in fold: np.testing.assert_array_equal(fold[name],declared_folds[index][name])
        check_partition(data.meta,fold);folds.append(fold)
    independent={h:targets(data,h) for h in (30,60)}
    summaries={};audited=0;replayed=0;expected_candidates=candidate_specs(protocol)
    chosen_ids={row['id'] for row in read(output/'choices.json')['selections'].values()} if (output/'choices.json').exists() else set()
    # Completed candidate summaries are written only after all fold payloads.
    for path in sorted((output/'search').glob('*/summary.json')):
        summary=read(path);folder=path.parent
        if summary['id']!=folder.name or summary['id'] not in expected_candidates:
            raise ValueError('Unregistered candidate identity')
        specification=expected_candidates[summary['id']]
        if any(summary[key]!=value for key,value in specification.items()):
            raise ValueError('Candidate parameters departed from protocol')
        if summary.get('protocol_sha256')!=digest(output/'protocol.json'):
            raise ValueError('Candidate protocol mismatch')
        horizon=summary['horizon'];lower,upper,status=independent[horizon];scores=[]
        for index,fold in enumerate(folds):
            location=folder/f'fold{index}'
            with np.load(location/'validation.npz') as saved:
                np.testing.assert_array_equal(saved['rows'],fold['validation'])
                cal,val=fold['calibration'],fold['validation']
                for key in ('raw_calibration','raw_validation'):
                    if not np.isfinite(saved[key]).all() or (saved[key]<-104-1e-7).any() or (saved[key]>96+1e-7).any():
                        raise ValueError('Saved raw predictions violate prescribed clipping')
                bias=prescribed_correction(saved['raw_calibration'],lower[cal],upper[cal],summary['objective'])
                np.testing.assert_allclose(saved['correction'],bias,atol=1e-9)
                prediction=np.clip(saved['raw_validation']+bias,-104,96)
                np.testing.assert_allclose(saved['prediction'],prediction,atol=1e-9)
                expected=metrics(prediction,lower[val],upper[val],status[val])
                check_values(read(location/'metrics.json'),expected,summary['id'])
                scores.append(expected['observed_mae'])
                if index==0 and (summary['id'] in chosen_ids or summary['id'].startswith('default_') or 'seven60_conservative' in summary['id']):
                    replay_model(location,summary,data,fold,saved);replayed+=1
                audited+=1
        np.testing.assert_allclose(summary['scores'],scores,atol=1e-8)
        np.testing.assert_allclose(summary['mean_observed_mae'],np.mean(scores),atol=1e-8)
        summaries[summary['id']]=summary
    frozen=check_choices(output,protocol,summaries)
    if (output/'refit_manifest.json').exists(): check_hashes(output,read(output/'refit_manifest.json'))
    result={'status':'passed','scope':'development','utc':datetime.now(timezone.utc).isoformat(),
        'completed_candidates':len(summaries),'fold_prediction_artifacts':audited,'saved_model_replays':replayed,
        'expected_candidates':len(expected_candidates),'all_predeclared_candidates_complete':set(summaries)==set(expected_candidates),
        'source_rows_replayed':sampled,'choices_frozen_and_reproduced':frozen,
        'final_arrays_opened_by_audit':False,'limitations':['Development source reconstruction samples three full prefixes; final scope reconstructs every final prefix',
        'Only completed candidate summaries are audited while search is running','Model replay samples three calibration/validation observations per default/selected first fold',
        'Dependency manifest was captured after search began; current bytes are reverified, not a proof against unrecorded transient edits']}
    if allow_final:
        if not (output/'final_completed.json').exists(): raise ValueError('Final completion required before audit opens final arrays')
        audit_final(output,protocol,segments,source,result,data)
        save_json(output/'audit.json',result)
    else: save_json(output/'audit_development.json',result)
    return result


def audit_final(output,protocol,segments,source,result,development_data):
    if not result['choices_frozen_and_reproduced']: raise ValueError('Final requires frozen reproduced choices')
    choices=read(output/'choices.json');started=read(output/'final_started.json');opened=read(output/'final_labels_opened.json');completed=read(output/'final_completed.json')
    choices_hash=digest(output/'choices.json')
    if any(marker['choices_sha256']!=choices_hash for marker in (started,opened,completed)):
        raise ValueError('Final marker choices mismatch')
    stamps=[pd.Timestamp(choices['frozen_utc']),pd.Timestamp(started['utc']),pd.Timestamp(opened['utc']),pd.Timestamp(completed['utc'])]
    if stamps!=sorted(stamps): raise ValueError('Final labels preceded frozen choices/refits')
    if started['refit_manifest_sha256']!=digest(output/'refit_manifest.json'): raise ValueError('Final refit manifest mismatch')
    data=Data(output/'final')
    allowed_final=set(pd.read_parquet(output/'final_sessions.parquet').session_id)
    if not set(data.meta.session_id)<=allowed_final: raise ValueError('Final identities outside frozen final sessions')
    if data.meta.event_proxy.nunique()<20 or data.meta.input_start.min()<segments['final_cutoff_ns']:
        raise ValueError('Actual final cohort does not pass fresh event gate')
    if data.meta.target_end.max()>segments['final_end_ns']: raise ValueError('Final period boundary violated')
    if opened['candidate_event_proxies']!=data.meta.event_proxy.nunique() or completed['candidates']!=len(data.meta) or completed['event_proxies']!=data.meta.event_proxy.nunique():
        raise ValueError('Final marker cohort counts mismatch')
    source_count=source_replay(data,source,'final',samples=None)
    final_metrics=read(output/'final_metrics.json');refits=read(output/'refits.json')
    development=development_data.meta
    sessions=development.groupby('session_id').input_start.min().sort_values()
    fit_sessions=set(sessions.index[:int(len(sessions)*.8)])
    fit_rows=np.flatnonzero(development.session_id.isin(fit_sessions))
    cal_rows=np.flatnonzero(~development.session_id.isin(fit_sessions)&(development.input_start>=development.iloc[fit_rows].target_end.max()+70*10**9))
    expected_names={f'{kind}_{model}_{horizon}_seed{seed}' for horizon in (30,60) for model in (*MODELS,*CONTROLS)
        for kind in (('selected','baseline') if model in MODELS else ('baseline',)) for seed in (7,42,101)}
    if len(refits)!=len(expected_names) or {r['name'] for r in refits}!=expected_names:
        raise ValueError('Incomplete or duplicated predeclared refit grid')
    old_names={f'frozen_old_{model}_{horizon}' for horizon in (30,60)
        for model in ('mean','ridge','sgd','decision_tree','random_forest','gradient_boosting','gru','monte_carlo')}
    if set(final_metrics)!=(expected_names|old_names): raise ValueError('Final metric grid differs from protocol')
    predictions={};bounds={h:targets(data,h) for h in (30,60)}
    development_bounds={h:targets(development_data,h) for h in (30,60)}
    mc_cal_cache={};scaler_audits=0;correction_replays=0
    for entry in refits:
        if entry['name']!=f"{entry['kind']}_{entry['model']}_{entry['horizon']}_seed{entry['seed']}": raise ValueError('Refit identity mismatch')
        specification=choices['selections'][f"{entry['model']}_{entry['horizon']}"] if entry['kind']=='selected' else {'family':'seven60','objective':'conservative'}
        if entry['family']!=specification['family'] or entry['objective']!=specification['objective']: raise ValueError('Refit specification departed from choices')
        prediction=np.load(output/'final'/f"{entry['name']}_prediction.npy")
        lower,upper,status=bounds[entry['horizon']]
        check_values(final_metrics[entry['name']],metrics(prediction,lower,upper,status),entry['name'])
        trades=pd.read_parquet(output/'final'/f"{entry['name']}_trades.parquet")
        economics=check_execution(trades,prediction,data.meta,data.history,lower,upper,status,entry['model'],entry['horizon'])
        check_values(final_metrics[entry['name']]['execution'],economics,'execution')
        indices=np.unique(np.linspace(0,len(data.meta)-1,min(3,len(data.meta)),dtype=int))
        fitted=load_model(output/'refits'/entry['name'],entry['model'],entry['family']);fitted['horizon']=entry['horizon']
        scaler=fitted.get('scaler')
        if fitted['kind']=='tabular' and hasattr(fitted['model'],'named_steps'):
            scaler=fitted['model'].named_steps.get('standardscaler')
        if scaler is not None:
            values=development_data.sequence(entry['family'])[fit_rows].reshape(-1,scaler.n_features_in_) if fitted['kind']=='gru' else development_data.table(entry['family'])[fit_rows]
            if scaler.n_samples_seen_!=len(values): raise ValueError('Refit scaler is not fit-only')
            np.testing.assert_allclose(scaler.mean_,values.mean(axis=0,dtype=np.float64),atol=1e-6,rtol=1e-6)
            scaler_audits+=1
        if entry['seed']==42:
            if fitted['kind']=='monte_carlo':
                mode=fitted['model_name']
                if mode not in mc_cal_cache:
                    both={h:[] for h in (30,60)}
                    for index in cal_rows:
                        row=development.iloc[index];identity={'session_id':row.session_id,'ticker':row.ticker,'input_end':int(row.input_end)}
                        values=simulate_mc(development_data.history[index,-60:],identity,
                            context=development_data.history[index] if mode=='monte_carlo_180' else None)['predictions']
                        for h in both: both[h].append(values[h])
                    mc_cal_cache[mode]={h:np.clip(values,-104,96) for h,values in both.items()}
                raw_cal=mc_cal_cache[mode][entry['horizon']]
            else: raw_cal=raw_predict(fitted,development_data,cal_rows)
            lo,hi,_=development_bounds[entry['horizon']]
            bias=prescribed_correction(raw_cal,lo[cal_rows],hi[cal_rows],entry['objective'])
            np.testing.assert_allclose(entry['correction'],bias,atol=1e-6,rtol=1e-6)
            correction_replays+=1
        expected=np.clip(raw_sample(fitted,data,indices)+entry['correction'],-104,96)
        np.testing.assert_allclose(prediction[indices],expected,atol=1e-4,rtol=1e-5)
        predictions[entry['name']]=prediction
        check_values(read(output/'refits'/entry['name']/'calibration.json'),
            {**{key:entry[key] for key in ('family','objective','correction')},'fit_count':len(fit_rows),'calibration_count':len(cal_rows)},'frozen refit calibration')
    old=Path(protocol['frozen_old_secondary']['run']);old_metrics=read(old/'metrics.json')
    for horizon in (30,60):
        lower,upper,status=bounds[horizon]
        for model in ('mean','ridge','sgd','decision_tree','random_forest','gradient_boosting','gru','monte_carlo'):
            name=f'frozen_old_{model}_{horizon}'
            prediction=np.load(output/'final'/f'{name}_prediction.npy')
            check_values(final_metrics[name],metrics(prediction,lower,upper,status),name)
            if model=='gru':
                import tensorflow as tf
                fitted={'kind':'gru','family':'seven60','model':tf.keras.models.load_model(old/f'gru_{horizon}/best.keras',compile=False),
                    'scaler':joblib.load(old/f'gru_{horizon}/scaler.joblib')}
            elif model=='monte_carlo': fitted={'kind':'monte_carlo','model_name':'monte_carlo_60','family':'seven60','horizon':horizon}
            else: fitted={'kind':'tabular','family':'seven60','model':joblib.load(old/'models'/f'{model}_{horizon}.joblib')}
            corrected=np.clip(raw_sample(fitted,data,indices)+old_metrics[f'{model}_{horizon}']['calibration_correction_cents'],-104,96)
            np.testing.assert_allclose(prediction[indices],corrected,atol=1e-4,rtol=1e-5)
    gates=read(output/'gates.json');primary=read(output/'primary_gates.json')
    if set(gates)!={f'{model}_{h}' for model in MODELS for h in (30,60)} or set(primary)!={'30','60'}:
        raise ValueError('Final promotion gate grid differs from protocol')
    seed_spread=read(output/'seed_spread.json')
    expected_spread={}
    for horizon in (30,60):
        for model in (*MODELS,*CONTROLS):
            for kind in (('selected','baseline') if model in MODELS else ('baseline',)):
                key=f'{kind}_{model}_{horizon}'
                expected_spread[key]={'seeds':[7,42,101],'primary_seed':42,
                    'observed_mae':[final_metrics[f'{key}_seed{seed}']['observed_mae'] for seed in (7,42,101)]}
    if set(seed_spread)!=set(expected_spread): raise ValueError('Seed robustness grid differs')
    check_values(seed_spread,expected_spread,'seed robustness')
    for horizon in (30,60):
        lower,_,status=bounds[horizon]
        for model in MODELS:
            gate=event_bootstrap(predictions[f'selected_{model}_{horizon}_seed42'],predictions[f'baseline_{model}_{horizon}_seed42'],lower,status,data.meta.event_proxy)
            check_values(gates[f'{model}_{horizon}'],gate,'bootstrap')
        winner=choices['primary_winners'][str(horizon)]['model'];baseline=choices['primary_baselines'][str(horizon)]['model']
        gate=event_bootstrap(predictions[f'selected_{winner}_{horizon}_seed42'],predictions[f'baseline_{baseline}_{horizon}_seed42'],lower,status,data.meta.event_proxy)
        check_values(primary[str(horizon)],{'winner':winner,'strongest_default':baseline,**gate},'primary bootstrap')
    result.update(scope='completed_final',final_arrays_opened_by_audit=True,final_candidates=len(data.meta),
        final_event_proxies=int(data.meta.event_proxy.nunique()),final_model_predictions_verified=len(refits)+16,final_source_rows_replayed=source_count)
    result.update(refit_scalers_verified=scaler_audits,primary_seed_refit_corrections_replayed=correction_replays)
    result['limitations'].append('Final bootstrap replay uses the frozen, unit-tested bootstrap implementation; does not establish global significance')
    result['limitations'].append('Refit corrections recomputed for primary seed42; robustness seeds matched frozen calibration files/counts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--allow-final',action='store_true')
    args=parser.parse_args();print(json.dumps(audit(args.output,args.allow_final),indent=2))
