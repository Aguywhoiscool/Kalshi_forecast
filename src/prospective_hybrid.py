"""Frozen prospective paper hybrid: explicit input-only readiness and once-only evaluation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
import pandas as pd
from src.causal_features import context_indices
from src.coverage_error_audit import timing_flags
from src.causal_swing_data import align_future
from src.accuracy_targets import error_metrics,event_bootstrap
from src.accuracy_audit import targets
from src.residual_accuracy_models import load,predict,nochange
from src.v3_research import FEATURES,digest,save_json,valid_rows,overlaps


def now():
    return datetime.now(timezone.utc).isoformat()


def status(output,stage,**extra):
    value=dict(utc=now(),stage=stage,**extra)
    save_json(output/'status.json',value)
    print(json.dumps(value),flush=True)
    log=output.parent/'agent_status/research.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('a',encoding='utf8') as stream:stream.write(json.dumps(value)+'\n')
    return value


def consumed_registry(artifacts,exclude=None):
    result=[]
    for marker in sorted(Path(artifacts).glob('*/final_labels_opened.json')):
        if exclude is not None and marker.parent.resolve()==Path(exclude).resolve():continue
        segment=marker.parent/'segments.json'
        if not segment.exists():raise ValueError('Consumed final lacks segment provenance')
        value=json.loads(segment.read_text())
        result.append(dict(run=str(marker.parent.resolve()),final_end_ns=int(value['final_end_ns']),
            marker_sha256=digest(marker),segments_sha256=digest(segment)))
    return result


def prospective_boundary(created_ns,registry):
    return max([int(created_ns),*[r['final_end_ns']+70_000_000_000 for r in registry]])


def verify_registry(registry):
    for entry in registry:
        run=Path(entry['run'])
        if digest(run/'final_labels_opened.json')!=entry['marker_sha256'] or digest(run/'segments.json')!=entry['segments_sha256']:
            raise ValueError('Consumed final registry provenance changed')


def prior_anchors(prior):
    prior=Path(prior)
    choices=json.loads((prior/'choices.json').read_text())
    choice_sha=digest(prior/'choices.json')
    if choice_sha!=(prior/'choices.sha256').read_text() or choices['protocol_sha256']!=digest(prior/'protocol.json'):
        raise ValueError('Prior frozen choices/protocol anchor changed')
    previous=json.loads((prior/'protocol.json').read_text())
    if previous['segments_sha256']!=digest(prior/'segments.json'):
        raise ValueError('Prior segment anchor changed')
    for name in ('final_started.json','final_labels_opened.json','final_completed.json'):
        if json.loads((prior/name).read_text())['choices_sha256']!=choice_sha:
            raise ValueError('Prior completion/consumption choices anchor changed')
    started=json.loads((prior/'final_started.json').read_text())
    if started['refit_manifest_sha256']!=digest(prior/'refit_manifest.json'):
        raise ValueError('Prior refit manifest anchor changed')
    source=Path(json.loads((prior/'segments.json').read_text())['source'])
    protection=source/'protected_v2_period.json'
    if digest(protection)!=previous['source_inputs_sha256']['protected_v2_period.json']:
        raise ValueError('Prior protected-period anchor changed')
    names=('choices.json','choices.sha256','protocol.json','segments.json','refit_manifest.json',
           'final_started.json','final_labels_opened.json','final_completed.json')
    anchors={str((prior/name).resolve()):digest(prior/name) for name in names}
    anchors[str(protection.resolve())]=digest(protection)
    return previous,protection,anchors


def manifest_digest(manifest,relative):
    canonical={}
    for key,value in manifest.items():
        name=key.replace('\\','/')
        if name in canonical:raise ValueError('Duplicate normalized manifest path')
        canonical[name]=value
    return canonical[str(relative).replace('\\','/')]


def canonical_spread(history):
    history=np.asarray(history)
    if history.ndim!=3 or history.shape[1:]!=(60,7) or not np.isfinite(history).all():
        raise ValueError('Expected finite Nx60x7 histories')
    spread=np.round((history[:,-1,1]-history[:,-1,0])*100,9)
    if (spread<=0).any():raise ValueError('Invalid spread')
    return spread


def training_threshold(history,fit_rows):
    fit_rows=np.asarray(fit_rows)
    if fit_rows.ndim!=1 or not len(fit_rows) or len(np.unique(fit_rows))!=len(fit_rows) or fit_rows.min()<0 or fit_rows.max()>=len(history):
        raise ValueError('Invalid frozen fit identities')
    return float(np.quantile(canonical_spread(history[:,-60:])[fit_rows],.75))


def route(history,gru_prediction,threshold):
    gru_prediction=np.asarray(gru_prediction,float)
    if gru_prediction.shape!=(len(history),2) or not np.isfinite(gru_prediction).all():
        raise ValueError('Invalid frozen GRU predictions')
    keep=canonical_spread(history)>=threshold
    return np.where(keep[:,None],gru_prediction,nochange(history)),keep


def verify(output):
    output=Path(output)
    protocol=json.loads((output/'protocol.json').read_text())
    if digest(output/'protocol.json')!=(output/'protocol.sha256').read_text():raise ValueError('Frozen protocol changed')
    root=Path(__file__).resolve().parent
    for name,expected in protocol['code_sha256'].items():
        if digest(root/name)!=expected:raise ValueError('Frozen code changed: '+name)
    for name,expected in protocol['model_sha256'].items():
        if digest(output/'model'/name)!=expected:raise ValueError('Frozen model changed: '+name)
    if digest(output/'protected_periods.json')!=protocol['protected_sha256']:raise ValueError('Protected periods changed')
    verify_registry(protocol['consumed_final_registry'])
    for name,expected in protocol.get('prior_anchor_sha256',{}).items():
        if digest(Path(name))!=expected:raise ValueError('Prior historical provenance changed')
    return protocol


def freeze(output,prior):
    output,prior=Path(output).resolve(),Path(prior).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():return verify(output)
    if any(output.iterdir()):raise ValueError('Use an empty new experiment directory')
    previous,protection,anchors=prior_anchors(prior)
    assignment=Path(previous['development_origin'])
    history=np.load(assignment/'development/history180.npy',mmap_mode='r')
    meta=pd.read_parquet(assignment/'development/metadata.parquet')
    for name in ('history180.npy','metadata.parquet'):
        if digest(assignment/'development'/name)!=previous['development_artifacts_sha256'][name]:
            raise ValueError('Frozen threshold development bytes changed')
    if history.shape!=(len(meta),180,7):raise ValueError('Threshold history metadata disagree')
    fit_rows=np.load(prior/'refit_fit.npy')
    sessions=meta.groupby('session_id').input_start.min().sort_values()
    expected=np.flatnonzero(meta.session_id.isin(set(sessions.index[:int(len(sessions)*.8)])))
    np.testing.assert_array_equal(fit_rows,expected)
    threshold=training_threshold(history,fit_rows)
    original=prior/'refits/gru_60_seed42'
    frozen=json.loads((prior/'refit_manifest.json').read_text())
    names=('best.keras','scaler.joblib','spec.json','calibration.json')
    for name in names:
        if digest(original/name)!=manifest_digest(frozen,f'refits/gru_60_seed42/{name}'):raise ValueError('Prior GRU refit changed')
    if json.loads((original/'spec.json').read_text())['architecture']!='gru':raise ValueError('Expected GRU')
    calibration=json.loads((original/'calibration.json').read_text())
    if calibration['fit_count']!=len(fit_rows):raise ValueError('Frozen model fit scope differs from threshold fit scope')
    correction=calibration['correction']
    if len(correction)!=2 or not np.isfinite(correction).all():raise ValueError('Invalid saved correction')
    model=output/'model';model.mkdir()
    for name in names:shutil.copyfile(original/name,model/name)
    shutil.copyfile(protection,output/'protected_periods.json')
    registry=consumed_registry(output.parent,exclude=output)
    created=now();created_ns=pd.Timestamp(created).value
    cutoff=prospective_boundary(created_ns,registry)
    code=('prospective_hybrid.py','residual_accuracy_models.py','coverage_error_audit.py','causal_features.py',
          'causal_swing_data.py','accuracy_targets.py','accuracy_audit.py','accuracy_models.py','model_benchmark.py',
          'swing_return_experiment.py','v3_research.py','audit_swing_returns.py','movement_labels.py',
          'residual_accuracy_experiment.py','model_traces.py')
    protocol=dict(created_utc=created,created_ns=created_ns,prospective_input_start_ns=cutoff,
        prospective_input_start_utc=pd.Timestamp(cutoff,tz='UTC').isoformat(),consumed_final_registry=registry,
        model_origin=str(original),model_sha256={n:digest(model/n) for n in names},correction=correction,
        prior_anchor_sha256=anchors,
        threshold_cents=threshold,threshold_quantile=.75,threshold_fit_count=len(fit_rows),
        threshold_provenance={str(p.resolve()):digest(p) for p in (prior/'refit_fit.npy',assignment/'development/history180.npy',assignment/'development/metadata.parquet')},
        code_sha256={n:digest(Path(__file__).parent/n) for n in code},protected_sha256=digest(output/'protected_periods.json'),
        routing='GRU iff round(100*(ask-bid),9)>=fitQ75; otherwise -100*(ask-bid)-4 bothsides',
        gru='Frozen residual-study gru_60_seed42; raw absolute output clipped[-104,96], saved correction then clip[-104,96]',
        cohort='Strict180 verified same-session histories; past gaps.5..1.5s;120s spacing;60.5s planned target boundary; no future availability filtering',
        evaluation=dict(entry_delay_seconds=1,horizon_seconds=60,cost_buffer_cents=4,minimum_depth_contracts=1),
        primary='Hybrid vs nochange: >=5% observed-side MAE improvement and positive paired95%1000eventproxybootstrap CI;>=20observedgroups',
        secondary='Frozen GRU descriptive comparator; no threshold/model/seed selection',
        missingness='Statuses0closed and1knownrejected observed;2/3/4 unknown retained intervals; conditional accuracy is not profitability',
        source_stability='Completed nonconflict files older30minutes; newestfile omitted; clock-only screening then SHA256 and size/mtime consistency checks',
        autonomy='Manual readiness/evaluate commands only; no service/schedule/retraining/livepromotion/trades')
    save_json(output/'protocol.json',protocol)
    (output/'protocol.sha256').write_text(digest(output/'protocol.json'),encoding='utf8')
    status(output,'waiting_for_fresh_data',threshold_cents=threshold,prospective_input_start_utc=protocol['prospective_input_start_utc'],outcomes_opened=False)
    return protocol


def snapshot(output,protocol,data_folder):
    registry=consumed_registry(output.parent,exclude=output)
    cutoff=prospective_boundary(protocol['prospective_input_start_ns'],registry)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    source=output/'sources'/stamp
    (source/'snapshot').mkdir(parents=True)
    protected=json.loads((output/'protected_periods.json').read_text())
    files,excluded,rows=[],[],[]
    paths=sorted(Path(data_folder).glob('features_*.csv'))
    stable_before=time.time()-1800
    for path in paths:
        session=path.stem.removeprefix('features_')
        before=path.stat()
        if session in protected['sessions']:
            excluded.append(dict(name=path.name,reason='protected_session'));continue
        if path==paths[-1] or before.st_mtime>=stable_before or 'sync-conflict' in path.name:
            excluded.append(dict(name=path.name,reason='recent_active_or_conflict'));continue
        clock=pd.read_csv(path,usecols=['timestamp'])
        times=pd.to_datetime(clock.timestamp,utc=True,format='ISO8601',errors='coerce').dropna()
        if len(times)<180 or times.max().value<cutoff:
            excluded.append(dict(name=path.name,reason='before_prospective_cutoff_or_short'));continue
        start,end=int(times.min().value),int(times.max().value)
        if overlaps(start,end,protected):
            excluded.append(dict(name=path.name,reason='protected_time'));continue
        if end>pd.Timestamp.now(tz='UTC').value+1_000_000_000:
            excluded.append(dict(name=path.name,reason='clock_in_future'));continue
        target=source/'snapshot'/path.name
        shutil.copyfile(path,target)
        after=path.stat();fingerprint=digest(target)
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or digest(path)!=fingerprint:
            target.unlink();excluded.append(dict(name=path.name,reason='changed_during_snapshot'));continue
        files.append(dict(name=path.name,sha256=fingerprint,size=after.st_size,mtime_ns=after.st_mtime_ns))
        rows.append(dict(session_id=session,input_start=start,target_end=end))
    save_json(source/'manifest.json',dict(created_utc=now(),files=files,excluded=excluded,
        prospective_cutoff_ns=cutoff,consumed_final_registry=registry,syncthing_remote_completion='unverified'))
    return source,rows


def input_cohort(source,rows,protocol,protected):
    manifest=json.loads((source/'manifest.json').read_text())
    cutoff=manifest['prospective_cutoff_ns']
    bound=max([cutoff,*[r['target_end'] for r in rows]])
    metadata,histories,timings=[],[],[]
    for item in manifest['files']:
        path=source/'snapshot'/item['name']
        if digest(path)!=item['sha256']:raise ValueError('Frozen source changed')
        session=path.stem.removeprefix('features_')
        frame=pd.read_csv(path)
        if not frame.session_id.eq(session).all():raise ValueError('Filename/session mismatch')
        frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True,format='ISO8601',errors='coerce')
        frame=frame.dropna(subset=['timestamp']).drop_duplicates(['session_id','ticker','timestamp'])
        for feature in FEATURES:frame[feature]=pd.to_numeric(frame[feature],errors='coerce')
        for ticker,group in frame.groupby('ticker'):
            group=group.sort_values('timestamp').reset_index(drop=True)
            times=group.timestamp.array.as_unit('ns').asi8
            missing,inconsistent,remaining=timing_flags(group)
            ends=np.arange(179,len(group))
            indices,okay=context_indices(times,group.session_id.to_numpy(),ends,180,valid_rows(group)&~missing&~inconsistent)
            ends,indices=ends[okay],indices[okay]
            chosen=[];next_time=-1
            for local,end in enumerate(ends):
                start=times[end-179];finish=times[end]+60_500_000_000
                if start<cutoff or finish>bound or overlaps(start,finish,protected):continue
                if times[end]>=next_time:chosen.append(local);next_time=times[end]+120_000_000_000
            if not chosen:continue
            ends,indices=ends[chosen],indices[chosen]
            metadata.append(pd.DataFrame(dict(session_id=session,ticker=str(ticker),event_proxy=str(ticker).rsplit('-',1)[0],
                series=str(ticker).split('-')[0],input_start=times[ends-179],input_end=times[ends],
                target_end=times[ends]+60_500_000_000,seconds_to_close=remaining[ends])))
            histories.append(group[FEATURES].to_numpy(float)[indices]);timings.append(remaining[indices])
    columns=['session_id','ticker','event_proxy','series','input_start','input_end','target_end','seconds_to_close']
    if not metadata:return pd.DataFrame(columns=columns),np.empty((0,180,7)),np.empty((0,180)),bound
    meta=pd.concat(metadata,ignore_index=True)
    order=meta.sort_values(['input_end','ticker']).index.to_numpy()
    return meta.iloc[order].reset_index(drop=True),np.concatenate(histories)[order],np.concatenate(timings)[order],bound


def readiness(output,data_folder):
    output=Path(output).resolve();protocol=verify(output)
    if (output/'final_started.json').exists():raise ValueError('Final already started; readiness cannot change its cohort')
    source,rows=snapshot(output,protocol,data_folder)
    meta,history,timing,bound=input_cohort(source,rows,protocol,json.loads((output/'protected_periods.json').read_text()))
    attempt=output/'readiness'/source.name;attempt.mkdir(parents=True)
    meta.to_parquet(attempt/'metadata.parquet',index=False)
    np.save(attempt/'history180.npy',history);np.save(attempt/'timing180.npy',timing)
    count=int(meta.event_proxy.nunique())
    record=dict(created_utc=now(),attempt=str(attempt),source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        protocol_sha256=digest(output/'protocol.json'),candidates=len(meta),event_proxies=count,
        final_cutoff_ns=json.loads((source/'manifest.json').read_text())['prospective_cutoff_ns'],final_end_ns=bound,
        input_artifacts_sha256={p.name:digest(p) for p in attempt.iterdir() if p.is_file()},outcomes_opened=False,ready=count>=20)
    save_json(attempt/'readiness.json',record);save_json(output/'latest_readiness.json',record)
    status(output,'ready_for_once_only_evaluation' if record['ready'] else 'waiting_for_fresh_data',
        candidates=len(meta),event_proxies=count,minimum_event_proxies=20,outcomes_opened=False,
        prospective_input_start_utc=pd.Timestamp(record['final_cutoff_ns'],tz='UTC').isoformat())
    return record


def once_gate(output,record):
    output=Path(output);protocol=verify(output)
    if (output/'final_started.json').exists() or (output/'final_labels_opened.json').exists() or (output/'final').exists():
        raise ValueError('Once-only evaluation already started; refusing re-entry')
    if not record['ready'] or record['event_proxies']<20:return False
    if record['protocol_sha256']!=digest(output/'protocol.json'):raise ValueError('Readiness protocol differs')
    source,attempt=Path(record['source']),Path(record['attempt'])
    if not source.is_relative_to(output/'sources') or not attempt.is_relative_to(output/'readiness'):raise ValueError('Readiness paths escape run')
    if digest(source/'manifest.json')!=record['source_manifest_sha256']:raise ValueError('Source manifest changed')
    manifest=json.loads((source/'manifest.json').read_text())
    verify_registry(manifest['consumed_final_registry'])
    for item in manifest['files']:
        path=(source/'snapshot'/item['name']).resolve()
        if not path.is_relative_to((source/'snapshot').resolve()) or digest(path)!=item['sha256']:
            raise ValueError('Frozen feature source changed')
    for name,expected in record['input_artifacts_sha256'].items():
        if digest(attempt/name)!=expected:raise ValueError('Readiness inputs changed')
    meta=pd.read_parquet(attempt/'metadata.parquet')
    cutoff=prospective_boundary(protocol['prospective_input_start_ns'],consumed_registry(output.parent,exclude=output))
    if record['final_cutoff_ns']<cutoff or len(meta)!=record['candidates'] or meta.event_proxy.nunique()!=record['event_proxies'] or meta.input_start.min()<cutoff or meta.target_end.max()>record['final_end_ns']:
        raise ValueError('Readiness cohort is stale or has invalid boundaries/counts')
    return True


def materialize_outcomes(output,record):
    output=Path(output)
    if not once_gate(output,record):return None
    verify(output)
    save_json(output/'segments.json',dict(final_cutoff_ns=record['final_cutoff_ns'],final_end_ns=record['final_end_ns'],
        source=record['source'],prospective_protocol_sha256=digest(output/'protocol.json')))
    with (output/'final_started.json').open('x',encoding='utf8') as stream:
        json.dump(dict(utc=now(),protocol_sha256=digest(output/'protocol.json'),readiness=record),stream)
    source,attempt=Path(record['source']),Path(record['attempt'])
    manifest=json.loads((source/'manifest.json').read_text())
    meta=pd.read_parquet(attempt/'metadata.parquet')
    # The first future alignment occurs after frozen input-only event gating and exclusive marker.
    save_json(output/'final_labels_opened.json',dict(utc=now(),protocol_sha256=digest(output/'protocol.json'),event_proxies=record['event_proxies']))
    future=np.full((len(meta),61,4),np.nan);known=np.zeros((len(meta),61),bool)
    for item in manifest['files']:
        path=source/'snapshot'/item['name']
        if digest(path)!=item['sha256']:raise ValueError('Frozen source changed before outcome alignment')
        frame=pd.read_csv(path);frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True,format='ISO8601',errors='coerce')
        frame=frame.dropna(subset=['timestamp']).drop_duplicates(['session_id','ticker','timestamp'])
        for feature in FEATURES:frame[feature]=pd.to_numeric(frame[feature],errors='coerce')
        session=path.stem.removeprefix('features_')
        for ticker,part in meta.loc[meta.session_id==session].groupby('ticker'):
            group=frame.loc[frame.ticker==ticker].sort_values('timestamp').reset_index(drop=True)
            times=group.timestamp.array.as_unit('ns').asi8
            aligned,_,valid=align_future(times,group[['bid','ask','bid_depth','ask_depth']].to_numpy(float),
                valid_rows(group),part.input_end.to_numpy(),np.full(len(part),60))
            future[part.index]=aligned;known[part.index]=valid
    final=output/'final';final.mkdir()
    shutil.copyfile(attempt/'metadata.parquet',final/'metadata.parquet')
    shutil.copyfile(attempt/'history180.npy',final/'history180.npy');shutil.copyfile(attempt/'timing180.npy',final/'timing180.npy')
    np.save(final/'future_quotes.npy',future);np.save(final/'future_valid.npy',known)
    return final


def evaluate(output):
    output=Path(output).resolve();protocol=verify(output)
    record=json.loads((output/'latest_readiness.json').read_text())
    if not record['ready']:
        status(output,'waiting_for_fresh_data',candidates=record['candidates'],event_proxies=record['event_proxies'],outcomes_opened=False)
        return None
    final=materialize_outcomes(output,record)
    from src.accuracy_models import Data
    data=Data(final);history=data.history[:,-60:]
    frozen=load(output/'model')
    gru=np.clip(np.clip(predict(frozen,history),-104,96)+protocol['correction'],-104,96)
    hybrid,selected=route(history,gru,protocol['threshold_cents'])
    lower,upper,outcome_status=targets(data,60)
    predictions=dict(hybrid=hybrid,nochange=nochange(history),gru=gru)
    metrics={}
    for name,prediction in predictions.items():
        np.save(final/f'{name}_prediction.npy',prediction)
        metrics[name]=error_metrics(prediction,lower,upper,outcome_status)
    np.save(final/'gru_routing_mask.npy',selected)
    gates={name:event_bootstrap(hybrid,predictions[name],lower,outcome_status,data.meta.event_proxy) for name in ('nochange','gru')}
    save_json(output/'metrics.json',metrics);save_json(output/'gates.json',gates)
    save_json(output/'final_completed.json',dict(utc=now(),protocol_sha256=digest(output/'protocol.json'),candidates=len(data.meta),
        event_proxies=int(data.meta.event_proxy.nunique()),gru_routed=int(selected.sum()),primary_gate_passes=gates['nochange']['passes_both'],promotion='none'))
    status(output,'complete',candidates=len(data.meta),primary_gate_passes=gates['nochange']['passes_both'],promotion='none')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','readiness','evaluate','status'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior',type=Path,default=Path('artifacts/residual_accuracy_20261109'))
    parser.add_argument('--data-folder',type=Path,default=Path('data/processed'))
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.output,args.prior)
    elif args.action=='readiness':readiness(args.output,args.data_folder)
    elif args.action=='evaluate':evaluate(args.output)
    else:print((args.output/'status.json').read_text())
