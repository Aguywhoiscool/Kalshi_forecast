"""Input-only development coverage reconstruction and matched OOF error audit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.v3_research import FEATURES,digest,save_json,valid_rows,overlaps
from src.accuracy_audit import targets,expected_folds,check_partition
from src.accuracy_models import Data
from src.residual_accuracy_models import nochange
from src.residual_accuracy_experiment import diagnostic_schema


def write_status(output,stage,**extra):
    value=dict(stage=stage,**extra)
    save_json(output/'status.json',value)
    print(json.dumps(value),flush=True)
    log=output.parent/'agent_status/research.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('a',encoding='utf8') as stream:
        stream.write(json.dumps(value)+'\n')


class Funnel:
    def __init__(self):
        self.counts=defaultdict(int)
        self.sessions=defaultdict(set)
        self.events=defaultdict(set)

    def add(self,name,mask,sessions,events):
        mask=np.asarray(mask,bool)
        self.counts[name]+=int(mask.sum())
        self.sessions[name].update(np.asarray(sessions)[mask].tolist())
        self.events[name].update(np.asarray(events)[mask].tolist())

    def records(self,unit,variant=''):
        return [dict(unit=unit,variant=variant,stage=name,count=count,
            sessions=len(self.sessions[name]),event_proxies=len(self.events[name])) for name,count in self.counts.items()]


def timing_flags(frame):
    n=len(frame)
    required={'scheduled_close_time','close_time_observed_at','seconds_to_scheduled_close'}
    if not required<=set(frame.columns):
        return np.ones(n,bool),np.zeros(n,bool),np.full(n,np.nan)
    close=pd.to_datetime(frame.scheduled_close_time,utc=True,errors='coerce',format='ISO8601').array.as_unit('ns').asi8
    observed=pd.to_datetime(frame.close_time_observed_at,utc=True,errors='coerce',format='ISO8601').array.as_unit('ns').asi8
    stamp=frame.timestamp.array.as_unit('ns').asi8
    remaining=pd.to_numeric(frame.seconds_to_scheduled_close,errors='coerce').to_numpy()
    missing=(close<=0)|(observed<=0)|~np.isfinite(remaining)
    inconsistent=~missing&((observed>stamp)|(remaining<0)|(np.abs((close-stamp)/1e9-remaining)>.001))
    return missing,inconsistent,remaining


def any_past(flag,length):
    cumulative=np.r_[0,np.cumsum(flag,dtype=np.int64)]
    ends=np.arange(len(flag))
    return cumulative[ends+1]-cumulative[np.maximum(0,ends-length+1)]>0


def window_masks(group,length,verified,bound,protected):
    """Never reads future prices; every timestamped row is a potential input end."""
    times=group.timestamp.array.as_unit('ns').asi8
    n=len(times)
    end=np.arange(n)
    start=np.maximum(0,end-length+1)
    missing,inconsistent,remaining=timing_flags(group)
    gap=np.r_[False,(np.diff(times)<.5e9)|(np.diff(times)>1.5e9)]
    # A bad transition at the first row of a window is outside that window.
    gap_counts=np.r_[0,np.cumsum(gap,dtype=np.int64)]
    gaps=(gap_counts[end+1]-gap_counts[start+1]>0)|(np.abs((times-times[start])/1e9-(length-1))>1.5)
    flags={'short_history':end<length-1,'quote_invalid':any_past(~valid_rows(group),length),
        'timing_missing':any_past(missing,length),'timing_inconsistent':any_past(inconsistent,length),
        'history_gaps_or_elapsed':gaps,
        'protected_overlap':overlaps(times[start],times+60_500_000_000,protected),
        'segment_bounds':times+60_500_000_000>bound}
    survivor=np.ones(n,bool)
    stages={'potential_input_ends':survivor.copy()}
    for name,flag in flags.items():
        enforced=verified or name not in ('timing_missing','timing_inconsistent')
        removed=survivor&flag if enforced else np.zeros(n,bool)
        stages['excluded_'+name]=removed
        survivor&=~removed
    stages['eligible_before_spacing']=survivor.copy()
    selected=np.zeros(n,bool)
    next_time=-1
    for row in np.flatnonzero(survivor):
        if times[row]>=next_time:
            selected[row]=True
            next_time=times[row]+120_000_000_000
    flags['spacing']=survivor&~selected
    stages['excluded_spacing']=flags['spacing']
    stages['selected_spaced']=selected
    return stages,flags,remaining


def reconstruct(source,assignment,output):
    segments=json.loads((assignment/'segments.json').read_text())
    if segments['source']!=str(source.resolve()) or segments['source_manifest_sha256']!=digest(source/'input_manifest.json'):
        raise ValueError('Authorized source provenance differs')
    development_path=assignment/'development_sessions.parquet'
    if digest(development_path)!=segments['development_metadata_sha256']:
        raise ValueError('Authorized development assignment changed')
    sessions=pd.read_parquet(development_path)
    protected=json.loads((source/'protected_v2_period.json').read_text())
    manifest=json.loads((source/'input_manifest.json').read_text())
    files={entry['name']:entry for entry in manifest['files']}
    universe=pd.read_parquet(source/'windows.parquet')
    source_summary=json.loads((source/'development_source.json').read_text())
    save_json(output/'source_universe.json',dict(snapshot_files=len(files),
        snapshot_rows_metadata=source_summary['snapshot_rows'],eligible_source_sessions=len(universe),
        authorized_development_sessions=len(sessions),
        outside_authorized_sessions_metadata_only=len(universe)-len(sessions),
        outside_authorized_snapshot_files_metadata_only=len(files)-len(sessions),
        outside_authorized_csvs_read=0,
        authorized_development_rows_metadata=int(sessions.source_rows.sum()),
        source_manifest_sha256=digest(source/'input_manifest.json'),assignment_sha256=digest(development_path)))
    row_funnel,row_flags=Funnel(),Funnel()
    funnels={f'{length}_{mode}':Funnel() for length in (60,180) for mode in ('quote_only','verified_timing')}
    overlaps_funnels={key:Funnel() for key in funnels}
    selected_pieces=[]
    group_records=[]
    for number,session in enumerate(sessions.session_id):
        if session in protected['sessions']:
            raise ValueError('Reserved session in authorized development')
        name=f'features_{session}.csv'
        path=source/'snapshot'/name
        if name not in files or digest(path)!=files[name]['sha256']:
            raise ValueError('Frozen authorized source bytes changed')
        frame=pd.read_csv(path)
        if not frame.session_id.eq(session).all():
            raise ValueError('Filename/session mismatch')
        frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True,errors='coerce',format='ISO8601')
        for column in FEATURES:
            frame[column]=pd.to_numeric(frame[column],errors='coerce')
        event=frame.ticker.fillna('__missing__').astype(str).str.rsplit('-',n=1).str[0].to_numpy()
        sid=np.full(len(frame),session)
        duplicate=frame.duplicated(['session_id','ticker','timestamp']).to_numpy()
        flags=dict(timestamp_invalid=frame.timestamp.isna().to_numpy(),duplicate_timestamp=duplicate,
            ticker_missing=frame.ticker.isna().to_numpy(),quote_invalid=~valid_rows(frame))
        missing,inconsistent,_=timing_flags(frame)
        flags.update(timing_missing=missing,timing_inconsistent=inconsistent)
        surviving=np.ones(len(frame),bool)
        row_funnel.add('source_rows',surviving,sid,event)
        for key,flag in flags.items():
            row_flags.add(key,flag,sid,event)
            removed=surviving&flag
            row_funnel.add('excluded_'+key,removed,sid,event)
            surviving&=~removed
        row_funnel.add('sequential_survivors',surviving,sid,event)
        # Do not remove quote/timing-invalid rows before forming histories.
        frame=frame.loc[frame.timestamp.notna()&frame.ticker.notna()].drop_duplicates(['session_id','ticker','timestamp'])
        for ticker,group in frame.groupby('ticker'):
            group=group.sort_values('timestamp').reset_index(drop=True)
            n=len(group)
            sid=np.full(n,session)
            event=np.full(n,str(ticker).rsplit('-',1)[0])
            for length in (60,180):
                for mode in ('quote_only','verified_timing'):
                    variant=f'{length}_{mode}'
                    stages,flags,remaining=window_masks(group,length,mode=='verified_timing',segments['development_end_ns'],protected)
                    for key,mask in stages.items():
                        funnels[variant].add(key,mask,sid,event)
                    for key,mask in flags.items():
                        overlaps_funnels[variant].add(key,mask,sid,event)
                    selected=np.flatnonzero(stages['selected_spaced'])
                    group_records.append(dict(variant=variant,session_id=session,ticker=ticker,
                        event_proxy=event[0],potential_input_ends=n,eligible_before_spacing=int(stages['eligible_before_spacing'].sum()),
                        selected_spaced=len(selected)))
                    if length==180 and mode=='verified_timing' and len(selected):
                        times=group.timestamp.array.as_unit('ns').asi8
                        selected_pieces.append(pd.DataFrame(dict(session_id=session,ticker=str(ticker),event_proxy=event[0],
                            series=str(ticker).split('-')[0],input_start=times[selected-179],input_end=times[selected],
                            target_end=times[selected]+60_500_000_000,seconds_to_close=remaining[selected])))
        if number%200==0:
            write_status(output,'coverage_reconstruction',authorized_files_read=number+1,total=len(sessions))
    reconstructed=pd.concat(selected_pieces,ignore_index=True).sort_values(['input_end','ticker']).reset_index(drop=True)
    expected=pd.read_parquet(assignment/'development/metadata.parquet')
    pd.testing.assert_frame_equal(reconstructed,expected,check_dtype=False,check_exact=True)
    reconstructed.to_parquet(output/'reproduced_metadata.parquet',index=False)
    save_json(output/'identity_reproduction.json',dict(status='passed',exact_rows=len(expected),
        sessions=int(expected.session_id.nunique()),event_proxies=int(expected.event_proxy.nunique()),
        expected_metadata_sha256=digest(assignment/'development/metadata.parquet'),all_columns_exact=True))
    records=row_funnel.records('recorded_rows')
    records+=sum([f.records('candidate_input_ends',key) for key,f in funnels.items()],[])
    pd.DataFrame(records).to_csv(output/'sequential_funnel.csv',index=False)
    overlap_records=row_flags.records('recorded_rows')
    overlap_records+=sum([f.records('candidate_input_ends',key) for key,f in overlaps_funnels.items()],[])
    pd.DataFrame(overlap_records).to_csv(output/'overlapping_flags.csv',index=False)
    pd.DataFrame(group_records).to_csv(output/'group_coverage.csv',index=False)
    summary=[]
    for variant,funnel in funnels.items():
        excluded=sum(v for k,v in funnel.counts.items() if k.startswith('excluded_'))
        if excluded+funnel.counts['selected_spaced']!=funnel.counts['potential_input_ends']:
            raise ValueError('Sequential window funnel does not reconcile')
        summary.append(dict(variant=variant,potential_input_ends=funnel.counts['potential_input_ends'],
            eligible_before_spacing=funnel.counts['eligible_before_spacing'],selected_spaced=funnel.counts['selected_spaced'],
            sessions=len(funnel.sessions['selected_spaced']),event_proxies=len(funnel.events['selected_spaced'])))
    if sum(v for k,v in row_funnel.counts.items() if k.startswith('excluded_'))+row_funnel.counts['sequential_survivors']!=row_funnel.counts['source_rows']:
        raise ValueError('Sequential row funnel does not reconcile')
    pd.DataFrame(summary).to_csv(output/'coverage_comparison.csv',index=False)
    return reconstructed


def error_stats(prediction,baseline,lower,upper,status,groups):
    observed=np.isin(status,[0,1])
    errors=(prediction-lower)[observed]
    paired=(np.abs(prediction-lower)-np.abs(baseline-lower))[observed]
    events=np.repeat(np.asarray(groups)[:,None],2,axis=1)[observed]
    count=len(np.unique(events))
    return dict(candidates=len(prediction),total_sides=int(status.size),observed_sides=int(observed.sum()),
        closed_sides=int((status==0).sum()),rejected_sides=int((status==1).sum()),unknown_sides=int((status>=2).sum()),
        observed_event_proxies=count,low_observed_event_support=count<20,
        observed_mae=float(np.abs(errors).mean()) if len(errors) else None,
        signed_bias=float(errors.mean()) if len(errors) else None,
        median_absolute_error=float(np.median(np.abs(errors))) if len(errors) else None,
        p90_absolute_error=float(np.quantile(np.abs(errors),.9)) if len(errors) else None,
        signed_paired_mae_difference=float(paired.mean()) if len(paired) else None,
        full_interval_distance_mae=float(np.abs(prediction-np.clip(prediction,lower,upper)).mean()))


def load_oof(path,rows):
    saved=np.load(path)
    np.testing.assert_array_equal(saved['rows'],rows)
    if saved['prediction'].shape!=(len(rows),2) or not np.isfinite(saved['prediction']).all():
        raise ValueError('Invalid OOF predictions')
    return saved['prediction']


def matched_oof(study,assignment,output):
    # Authorized development outcomes are opened only after exact identity pass.
    if json.loads((output/'identity_reproduction.json').read_text())['status']!='passed':
        raise ValueError('Exact development identity reproduction is required')
    data=Data(assignment/'development')
    history=data.history[:,-60:]
    choices=json.loads((study/'choices.json').read_text())
    if digest(study/'choices.json')!=(study/'choices.sha256').read_text():
        raise ValueError('Frozen OOF selection changed')
    if choices['development_metadata_sha256']!=digest(assignment/'development/metadata.parquet'):
        raise ValueError('Study OOF development identities differ')
    folds=expected_folds(data.meta)
    seen=set()
    identities=[]
    for index,fold in enumerate(folds):
        check_partition(data.meta,fold)
        for name,rows in fold.items():
            np.testing.assert_array_equal(rows,np.load(study/'folds'/str(index)/f'{name}.npy'))
        rows=fold['validation']
        if seen.intersection(rows):
            raise ValueError('OOF observation counted twice')
        seen.update(rows)
        identities.append(data.meta.iloc[rows].assign(original_row=rows,fold=index))
    pd.concat(identities).to_parquet(output/'oof_identities.parquet',index=False)
    records,associations,paired_rows=[],[],[]
    for horizon in (30,60):
        lower,upper,status=targets(data,horizon)
        winner=choices['primary_winners'][str(horizon)]['id']
        all_predictions=defaultdict(list)
        all_ids=[]
        for index,fold in enumerate(folds):
            rows=fold['validation']
            baseline=nochange(history[rows])
            predictions={'nochange':baseline}
            for label,candidate in [('winner',winner),('ensemble',f'ensemble_{horizon}')]:
                predictions[label]=load_oof(study/'search'/candidate/f'fold{index}'/'validation.npz',rows)
            schema=diagnostic_schema(history,data.meta,fold['fit'])
            save_json(output/f'fold{index}_diagnostic_schema.json',schema)
            spread=(history[rows,-1,1]-history[rows,-1,0])*100
            vol=np.diff(history[rows,:,3],axis=1).std(axis=1)*100
            seconds=data.meta.iloc[rows].seconds_to_close.to_numpy()
            labels={key:np.searchsorted(schema['quantile_edges'][key],value,side='right').astype(str)
                for key,value in [('spread',spread),('volatility',vol),('time_to_close',seconds)]}
            labels['series']=np.where(data.meta.iloc[rows].series.isin(schema['training_series']),data.meta.iloc[rows].series,'__unseen__')
            entry=data.future[rows,1]
            exit=data.future[rows,horizon]
            valid_pair=data.known[rows,1]&data.known[rows,horizon]
            movement=np.abs((exit[:,0]+exit[:,1]-entry[:,0]-entry[:,1])*50)
            labels['future_midpoint_movement']=np.select([~valid_pair,movement<1e-9,movement<1],['unknown','unchanged','positive_under1c'],default='at_least1c')
            labels['entry_outcome_availability']=np.select([(status[rows]>=2).any(axis=1),(status[rows]==1).any(axis=1)],['unknown_side','known_rejection'],default='both_closed')
            for model,prediction in predictions.items():
                all_predictions[model].append(prediction)
                records.append(dict(horizon=horizon,fold=index,model=model,dimension='all',group='all',
                    **error_stats(prediction,baseline,lower[rows],upper[rows],status[rows],data.meta.iloc[rows].event_proxy)))
                for dimension,values in labels.items():
                    for group in np.unique(values):
                        keep=values==group
                        records.append(dict(horizon=horizon,fold=index,model=model,dimension=dimension,group=str(group),
                            postprediction_diagnostic=dimension in ('future_midpoint_movement','entry_outcome_availability'),
                            **error_stats(prediction[keep],baseline[keep],lower[rows][keep],upper[rows][keep],status[rows][keep],data.meta.iloc[rows].event_proxy.to_numpy()[keep])))
                for local,row in enumerate(rows):
                    for side in (0,1):
                        paired_rows.append(dict(horizon=horizon,original_row=int(row),fold=index,model=model,side=side,
                            event_proxy=data.meta.iloc[row].event_proxy,
                            status=int(status[row,side]),prediction_cents=float(prediction[local,side]),
                            baseline_cents=float(baseline[local,side]),lower_cents=float(lower[row,side]),upper_cents=float(upper[row,side]),
                            model_minus_baseline_absolute_error_cents=float(abs(prediction[local,side]-lower[row,side])-abs(baseline[local,side]-lower[row,side])),
                            **{f'{dimension}_group':str(values[local]) for dimension,values in labels.items()}))
            # Algebraic quote-return decomposition for closed sides only.
            for side in (0,1):
                closed=status[rows,side]==0
                sign=1 if side==0 else -1
                delta_mid=(exit[:,0]+exit[:,1]-entry[:,0]-entry[:,1])*50
                current_mid=history[rows,-1,3]
                entry_mid=(entry[:,0]+entry[:,1])/2
                entry_shift=sign*(entry_mid-current_mid)*100
                future_shift=sign*delta_mid
                entry_spread=(entry[:,1]-entry[:,0])*100
                exit_spread=(exit[:,1]-exit[:,0])*100
                spread_term=spread-(entry_spread+exit_spread)/2
                expected_error=-future_shift+entry_spread/2+exit_spread/2-spread
                np.testing.assert_allclose((baseline[:,side]-lower[rows,side])[closed],expected_error[closed],atol=1e-7)
                for local in np.flatnonzero(closed):
                    associations.append(dict(horizon=horizon,fold=index,original_row=int(rows[local]),side=side,
                        event_proxy=data.meta.iloc[rows[local]].event_proxy,
                        signed_baseline_error_cents=float(expected_error[local]),
                        future_midpoint_component_cents=float(-future_shift[local]),
                        future_spread_component_cents=float(-spread_term[local]),
                        entry_midpoint_shift_diagnostic_cents=float(entry_shift[local]),
                        interpretation='Exact closed quote-return algebra; associations, not causal attribution'))
            all_ids.extend(rows)
        rows=np.asarray(all_ids)
        baseline=np.concatenate(all_predictions['nochange'])
        for model,blocks in all_predictions.items():
            records.append(dict(horizon=horizon,fold='pooled_unique',model=model,dimension='all',group='all',
                **error_stats(np.concatenate(blocks),baseline,lower[rows],upper[rows],status[rows],data.meta.iloc[rows].event_proxy)))
    pd.DataFrame(records).to_csv(output/'oof_error_groups.csv',index=False)
    pd.DataFrame(paired_rows).to_parquet(output/'oof_side_errors.parquet',index=False)
    pd.DataFrame(associations).to_csv(output/'closed_return_decomposition.csv',index=False)
    return len(seen)


def artifact_report(output):
    identity=json.loads((output/'identity_reproduction.json').read_text())
    coverage=pd.read_csv(output/'coverage_comparison.csv')
    errors=pd.read_csv(output/'oof_error_groups.csv')
    pooled=errors.loc[(errors.fold=='pooled_unique')&(errors.dimension=='all')]
    lines=['# Development coverage and error audit','',
        f"Exact prior development identity reconstruction passed: {identity['exact_rows']:,} observations, {identity['sessions']:,} sessions and {identity['event_proxies']:,} event proxies.",'',
        '## Input-only coverage','',
        '| History/metadata rule | Eligible ends before spacing | Spaced observations | Sessions | Event proxies |',
        '|---|---:|---:|---:|---:|']
    for row in coverage.itertuples():
        lines.append(f'| {row.variant} | {row.eligible_before_spacing:,} | {row.selected_spaced:,} | {row.sessions:,} | {row.event_proxies:,} |')
    lines+=['','These counterfactual coverage counts use only past inputs, planned timestamp boundaries and the same120-second spacing. They do not compare predictive performance on different populations. No data was imputed, stitched, relabeled or filtered by future price availability.',
        '', '## Matched development OOF errors','',
        '| Horizon | Model | Observed MAE(cents) | Signed bias | Median error | P90 error | Model minus nochange MAE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in pooled.itertuples():
        lines.append(f'| {row.horizon}s | {row.model} | {row.observed_mae:.6f} | {row.signed_bias:.6f} | {row.median_absolute_error:.6f} | {row.p90_absolute_error:.6f} | {row.signed_paired_mae_difference:.6f} |')
    lines+=['','Negative paired differences favor the model. Pooled observations appear once per horizon/model; YES/NO sides remain distinct outcomes. Unknown intervals and known rejected entries are counted separately. Group files mark fewer than20 observed event proxies as low support. Future-movement and availability groups are postprediction diagnostics, never input filters.',
        '', 'The sequential funnel partitions exclusions; overlapping flags deliberately count a problematic row/window more than once. Session/event counts are unique within each stage and must not be added across stages. The row funnel is descriptive: invalid quote/timing rows stay in their original positions while window eligibility is checked, preventing accidental stitching.',
        '', 'Closed quote-return baseline errors are decomposed algebraically into midpoint and spread components. Their associations do not identify a causal effect. Recorded rows, overlapping candidate ends, spaced observations, sessions and event proxies are different units; none is a claim of independent sample size.',
        '', 'No reserved or final outcome arrays were opened. No model was trained or promoted, and neither recordings nor collection settings were changed.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def run(source,assignment,study,output):
    output.mkdir(parents=True,exist_ok=False)
    write_status(output,'starting_input_only_coverage',no_training=True,no_final_outcomes=True)
    reconstructed=reconstruct(source.resolve(),assignment.resolve(),output)
    write_status(output,'exact_identity_reproduced',observations=len(reconstructed))
    count=matched_oof(study.resolve(),assignment.resolve(),output)
    artifact_report(output)
    save_json(output/'audit.json',dict(status='passed',exact_development_rows=len(reconstructed),unique_oof_rows=count,
        scope='authorized_development_only',final_outcome_arrays_opened=False,
        source_manifest_sha256=digest(source/'input_manifest.json'),study_choices_sha256=digest(study/'choices.json')))
    write_status(output,'complete',exact_development_rows=len(reconstructed),unique_oof_rows=count,final_outcomes_read=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/accuracy_source_20261109'))
    parser.add_argument('--assignment',type=Path,default=Path('artifacts/accuracy_experiment_20261109'))
    parser.add_argument('--study',type=Path,default=Path('artifacts/residual_accuracy_20261109'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.assignment,args.study,args.output)
