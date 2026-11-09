"""Acceptance replay for residual study; final arrays require completion marker."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.accuracy_audit import (read, check_hashes, check_values, expected_folds, check_partition,
    targets, metrics, source_replay, check_execution)
from src.accuracy_models import Data,load_model,raw_predict
from src.accuracy_targets import event_bootstrap
from src.residual_accuracy_models import load,predict,inputs,nochange,ARCHITECTURES
from src.residual_accuracy_experiment import identifier,freeze_protocol,baseline_oof,diagnostic_schema
from src.v3_research import digest,save_json


def extended_metrics(prediction,lower,upper,status):
    value=metrics(prediction,lower,upper,status)
    observed=np.isin(status,[0,1])
    errors=(prediction-lower)[observed]
    value.update(signed_bias_cents=float(errors.mean()) if len(errors) else None,
        median_absolute_error_cents=float(np.median(np.abs(errors))) if len(errors) else None,
        p90_absolute_error_cents=float(np.quantile(np.abs(errors),.9)) if len(errors) else None)
    return value


def check_final_registry(refits,metric_keys):
    expected_refits={f'{a}_{h}_seed{s}' for a in ARCHITECTURES for h in (30,60) for s in (7,42,101)}
    expected_keys=expected_refits|{f'ensemble_{h}_seed{s}' for h in (30,60) for s in (7,42,101)}|{f'{c}_{h}' for c in ('nochange','current_gru','strongest_default') for h in (30,60)}
    if set(metric_keys)!=expected_keys:
        raise ValueError('Missing expected30 final model/seed/control metrics')
    if len(refits)!=18 or {e['name'] for e in refits}!=expected_refits:
        raise ValueError('Expected18 unique frozen refits')
    for entry in refits:
        if entry['architecture'] not in ARCHITECTURES or entry['horizon'] not in (30,60) or entry['seed'] not in (7,42,101) or entry['name']!=f"{entry['architecture']}_{entry['horizon']}_seed{entry['seed']}":
            raise ValueError('Refit identity mismatch')


def check_execution_registry(execution):
    expected={f'{a}_{h}_seed42' for a in (*ARCHITECTURES,'ensemble') for h in (30,60)}|{f'{c}_{h}' for c in ('nochange','current_gru','strongest_default') for h in (30,60)}
    if set(execution)!=expected:
        raise ValueError('Missing expected14 execution records')


def replay(folder,history,fit_rows,cal_rows,lower,upper,saved=None):
    fitted=load(folder)
    x=inputs(history,fitted['spec']['architecture'])[fit_rows]
    expected=x.reshape(-1,x.shape[-1])
    if fitted['scaler'].n_samples_seen_!=len(expected):
        raise ValueError('Scaler fit count differs')
    np.testing.assert_allclose(fitted['scaler'].mean_,expected.mean(axis=0,dtype=np.float64),atol=1e-6,rtol=1e-6)
    raw=np.clip(predict(fitted,history[cal_rows]),-104,96)
    absolute=predict(fitted,history[cal_rows])
    residual=(absolute-np.minimum(upper[cal_rows],np.maximum(lower[cal_rows],absolute)))/100
    magnitude=np.abs(residual)
    loss=fitted['spec']['loss']
    delta=.01 if loss=='huber1' else .05
    value=np.mean(magnitude if loss=='l1' else np.where(magnitude<=delta,.5*residual**2,delta*(magnitude-.5*delta)))
    epochs=pd.read_csv(folder/'epochs.csv')
    if not 1<=len(epochs)<=30:
        raise ValueError('Saved checkpoint does not reproduce minimum calibration objective')
    np.testing.assert_allclose(value,epochs.val_loss.min(),atol=1e-8,rtol=1e-5)
    bias=np.mean(np.minimum(upper[cal_rows],np.maximum(lower[cal_rows],raw))-raw,axis=0)
    if saved is not None:
        np.testing.assert_allclose(raw,saved['raw_calibration'],atol=1e-4,rtol=1e-5)
        np.testing.assert_allclose(bias,saved['correction'],atol=1e-5,rtol=1e-5)
        calculated=np.clip(np.clip(predict(fitted,history[saved['rows']]),-104,96)+bias,-104,96)
        np.testing.assert_allclose(calculated,saved['prediction'],atol=1e-4,rtol=1e-5)
    return fitted,bias


def audit(output,allow_final=False):
    output=Path(output).resolve()
    if allow_final and not (output/'final_completed.json').exists():
        raise ValueError('Completed final marker required before any final reads')
    protocol=read(output/'protocol.json')
    previous=Path(protocol['development_origin'])
    # This recomputes every source/dependency/control hash and refuses changes.
    freeze_protocol(output,previous)
    data=Data(output/'development')
    history=data.history[:,-60:]
    folds=expected_folds(data.meta)
    for index,fold in enumerate(folds):
        check_partition(data.meta,fold)
        for name,rows in fold.items():
            np.testing.assert_array_equal(rows,np.load(output/'folds'/str(index)/f'{name}.npy'))
    old_source=Path(read(previous/'segments.json')['source'])
    source_count=source_replay(data,old_source,'development',samples=6)
    summaries={}
    replays=0
    for horizon in (30,60):
        lower,upper,status=targets(data,horizon)
        for spec in protocol['search'][str(horizon)]:
            name=identifier(horizon,spec)
            path=output/'search'/name/'summary.json'
            if not path.exists():
                continue
            summary=read(path)
            if summary['id']!=name or summary['horizon']!=horizon or summary['spec']!=spec or summary['protocol_sha256']!=digest(output/'protocol.json'):
                raise ValueError('Candidate configuration differs')
            scores=[]
            for index,fold in enumerate(folds):
                folder=path.parent/f'fold{index}'
                saved=np.load(folder/'validation.npz')
                if read(folder/'spec.json')!=spec:
                    raise ValueError('Saved model specification differs')
                np.testing.assert_array_equal(saved['rows'],fold['validation'])
                val=fold['validation']
                expected=extended_metrics(saved['prediction'],lower[val],upper[val],status[val])
                check_values(read(folder/'metrics.json'),expected,'foldmetrics')
                scores.append(expected['observed_mae'])
                if read(folder/'diagnostic_schema.json')!=diagnostic_schema(history,data.meta,fold['fit']):
                    raise ValueError('Diagnostic bins are not fit-only')
                # One actual saved fold for every architecture/target/loss/horizon.
                if index==0:
                    replay(folder,history,fold['fit'],fold['calibration'],lower,upper,saved)
                    replays+=1
            np.testing.assert_allclose(summary['scores'],scores,atol=1e-7)
            np.testing.assert_allclose(summary['mean_observed_mae'],np.mean(scores),atol=1e-7)
            summaries[name]=summary
    choices_path=output/'choices.json'
    if choices_path.exists():
        expected_names={identifier(h,s) for h in (30,60) for s in protocol['search'][str(h)]}
        if set(summaries)!=expected_names:
            raise ValueError('Missing expected36candidate summaries/108fold payloads')
        choices=read(choices_path)
        if digest(choices_path)!=(output/'choices.sha256').read_text():
            raise ValueError('Frozen choices changed')
        for key,path in [('protocol_sha256','protocol.json'),('segments_sha256','segments.json'),
                         ('development_metadata_sha256','development/metadata.parquet')]:
            if choices[key]!=digest(output/path):
                raise ValueError('Frozen choices dependency mismatch')
        for horizon in (30,60):
            selected={arch:min([r for r in summaries.values() if r['horizon']==horizon and r['spec']['architecture']==arch],
                key=lambda r:r['mean_observed_mae']) for arch in ARCHITECTURES}
            if selected!=choices['selections'][str(horizon)]:
                raise ValueError('Architecture selection is not dev minimum')
            lower,upper,status=targets(data,horizon)
            ensemble_scores=[]
            for index,fold in enumerate(folds):
                member=[np.load(output/'search'/selected[a]['id']/f'fold{index}'/'validation.npz') for a in ARCHITECTURES]
                for saved in member:
                    np.testing.assert_array_equal(saved['rows'],fold['validation'])
                expected=np.mean([saved['prediction'] for saved in member],axis=0)
                saved=np.load(output/'search'/f'ensemble_{horizon}'/f'fold{index}'/'validation.npz')
                np.testing.assert_array_equal(saved['rows'],fold['validation'])
                np.testing.assert_allclose(saved['prediction'],expected,atol=1e-7)
                v=fold['validation']
                ensemble_scores.append(metrics(expected,lower[v],upper[v],status[v])['observed_mae'])
            ensemble=dict(id=f'ensemble_{horizon}',spec={'architecture':'ensemble'},horizon=horizon,
                scores=ensemble_scores,mean_observed_mae=float(np.mean(ensemble_scores)))
            if min([*selected.values(),ensemble],key=lambda r:r['mean_observed_mae'])!=choices['primary_winners'][str(horizon)]:
                raise ValueError('Primary winner is not prefinal dev minimum')
            controls=baseline_oof(previous,data,folds,horizon)
            strongest=min([v for k,v in controls.items() if k!='current_gru'],key=lambda r:r['mean_observed_mae']).copy()
            strongest.pop('predictions')
            if strongest!=choices['primary_baselines'][str(horizon)]:
                raise ValueError('Strongest default baseline mismatch')
    result=dict(status='passed',scope='development',candidate_summaries=len(summaries),
        expected_candidate_summaries=36,expected_fold_evaluations=108,actual_model_replays=replays,
        source_prefix_replays=source_count,final_arrays_opened=False)
    if allow_final:
        if not choices_path.exists():
            raise ValueError('Final requires frozen choices')
        segments=read(output/'segments.json')
        for entry in segments['previously_opened_final_registry']:
            if segments['final_cutoff_ns']<entry['final_end_ns']+70*10**9:
                raise ValueError('Final overlaps consumed prior outcomes')
        check_hashes(output,read(output/'refit_manifest.json'))
        completed=read(output/'final_completed.json')
        started=read(output/'final_started.json')
        opened=read(output/'final_labels_opened.json')
        if any(marker['choices_sha256']!=digest(choices_path) for marker in (completed,started,opened)):
            raise ValueError('Final marker choices differ')
        if started['refit_manifest_sha256']!=digest(output/'refit_manifest.json'):
            raise ValueError('Frozen refit manifest differs')
        if not (pd.Timestamp(choices['frozen_utc'])<=pd.Timestamp(started['utc'])<=pd.Timestamp(opened['utc'])<=pd.Timestamp(completed['utc'])):
            raise ValueError('Final chronology mismatch')
        final=Data(output/'final')
        if final.meta.input_start.min()<segments['final_cutoff_ns'] or final.meta.target_end.max()>segments['final_end_ns'] or final.meta.event_proxy.nunique()<20:
            raise ValueError('Final eligibility boundary/count failure')
        if not set(final.meta.session_id)<=set(pd.read_parquet(output/'final_sessions.parquet').session_id):
            raise ValueError('Final sessions differ from frozen identities')
        if completed['candidates']!=len(final.meta) or completed['event_proxies']!=final.meta.event_proxy.nunique() or opened['candidate_event_proxies']!=final.meta.event_proxy.nunique():
            raise ValueError('Final marker cohort counts differ')
        source_count+=source_replay(final,Path(segments['source']),'final',samples=6)
        fit_rows=np.load(output/'refit_fit.npy');cal_rows=np.load(output/'refit_calibration.npy')
        sessions=data.meta.groupby('session_id').input_start.min().sort_values()
        fit_sessions=set(sessions.index[:int(len(sessions)*.8)])
        expected_fit=np.flatnonzero(data.meta.session_id.isin(fit_sessions))
        expected_cal=np.flatnonzero(~data.meta.session_id.isin(fit_sessions)&(data.meta.input_start>=data.meta.iloc[expected_fit].target_end.max()+70*10**9))
        np.testing.assert_array_equal(fit_rows,expected_fit)
        np.testing.assert_array_equal(cal_rows,expected_cal)
        if set(data.meta.iloc[fit_rows].session_id)&set(data.meta.iloc[cal_rows].session_id) or data.meta.iloc[cal_rows].input_start.min()<data.meta.iloc[fit_rows].target_end.max()+70*10**9:
            raise ValueError('Refit embargo violation')
        final_metrics=read(output/'final_metrics.json')
        predictions={name:np.load(output/'final'/f'{name}_prediction.npy') for name in final_metrics}
        refits=read(output/'refits.json')
        check_final_registry(refits,final_metrics)
        for entry in refits:
            folder=output/'refits'/entry['name']
            expected_spec=choices['selections'][str(entry['horizon'])][entry['architecture']]['spec']
            if read(folder/'spec.json')!=expected_spec:
                raise ValueError('Refit model specification differs')
            scaler=joblib.load(folder/'scaler.joblib')
            x=inputs(history,entry['architecture'])[fit_rows]
            x=x.reshape(-1,x.shape[-1])
            if scaler.n_samples_seen_!=len(x):
                raise ValueError('Refit scaler is not fit-only')
            np.testing.assert_allclose(scaler.mean_,x.mean(axis=0,dtype=np.float64),atol=1e-6,rtol=1e-6)
            calibration=read(folder/'calibration.json')
            if calibration['fit_count']!=len(fit_rows) or calibration['calibration_count']!=len(cal_rows):
                raise ValueError('Refit calibration population differs')
            np.testing.assert_allclose(calibration['correction'],entry['correction'],atol=1e-7)
            if entry['seed']!=42:
                continue
            lower,upper,_=targets(data,entry['horizon'])
            fitted,bias=replay(output/'refits'/entry['name'],history,fit_rows,cal_rows,lower,upper)
            np.testing.assert_allclose(bias,entry['correction'],atol=1e-5)
            expected=np.clip(np.clip(predict(fitted,final.history[:,-60:]),-104,96)+bias,-104,96)
            np.testing.assert_allclose(predictions[entry['name']],expected,atol=1e-4,rtol=1e-5)
            replays+=1
        execution=read(output/'execution.json')
        check_execution_registry(execution)
        gates=read(output/'primary_gates.json')
        for horizon in (30,60):
            lower,upper,status=targets(final,horizon)
            prior_refits=read(previous/'refits.json')
            strongest=choices['primary_baselines'][str(horizon)]['name']
            for control,prior_name in [('current_gru',f'selected_gru_{horizon}_seed42'),
                ('strongest_default',None if strongest=='nochange' else f"baseline_{strongest.removeprefix('default_')}_{horizon}_seed42")]:
                if prior_name is None:
                    expected=nochange(final.history[:,-60:])
                else:
                    prior_entry=next(e for e in prior_refits if e['name']==prior_name)
                    fitted=load_model(previous/'refits'/prior_name,prior_entry['model'],prior_entry['family'])
                    fitted['horizon']=horizon
                    expected=np.clip(raw_predict(fitted,final,np.arange(len(final.meta)))+prior_entry['correction'],-104,96)
                np.testing.assert_allclose(predictions[f'{control}_{horizon}'],expected,atol=1e-4,rtol=1e-5)
            for name,prediction in predictions.items():
                if f'_{horizon}_' not in name and not name.endswith(f'_{horizon}'):
                    continue
                check_values(final_metrics[name],extended_metrics(prediction,lower,upper,status),'finalmetrics')
                if name in execution:
                    expected=check_execution(pd.read_parquet(output/'final'/f'{name}_trades.parquet'),prediction,
                        final.meta,final.history[:,-60:],lower,upper,status,name,horizon)
                    check_values(execution[name],expected,'execution')
            for seed in (7,42,101):
                np.testing.assert_allclose(predictions[f'ensemble_{horizon}_seed{seed}'],
                    np.mean([predictions[f'{a}_{horizon}_seed{seed}'] for a in ARCHITECTURES],axis=0),atol=1e-7)
            np.testing.assert_allclose(predictions[f'nochange_{horizon}'],nochange(final.history[:,-60:]),atol=1e-7)
            arch=choices['primary_winners'][str(horizon)]['spec']['architecture']
            if gates[str(horizon)]['primary']!=arch:
                raise ValueError('Gate primary differs from frozen choice')
            expected={name:event_bootstrap(predictions[f'{arch}_{horizon}_seed42'],predictions[f'{name}_{horizon}'],lower,status,final.meta.event_proxy)
                      for name in ('current_gru','strongest_default')}
            check_values(gates[str(horizon)]['comparisons'],expected,'bootstrap')
            if gates[str(horizon)]['passes_both_comparators']!=all(v['passes_both'] for v in expected.values()):
                raise ValueError('Dual-comparator gate mismatch')
        result.update(scope='completed_final',final_arrays_opened=True,final_candidates=len(final.meta),
            final_event_proxies=int(final.meta.event_proxy.nunique()),actual_model_replays=replays,
            source_prefix_replays=source_count,final_metric_records=len(final_metrics),choices_sha256=digest(choices_path))
    save_json(output/('audit.json' if allow_final else 'audit_development.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-final',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit(args.output,args.allow_final),indent=2))
