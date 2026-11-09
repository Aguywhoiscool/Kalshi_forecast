"""Pooled descriptive diagnostics and provenance for a completed coverage audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.accuracy_models import Data
from src.coverage_error_audit import error_stats
from src.v3_research import digest,save_json


def movement_groups(future,known,horizon):
    entry,exit=future[:,1],future[:,horizon]
    good=known[:,1]&known[:,horizon]
    # Normalize binary floating arithmetic at a 1e-9-cent tolerance; no quote is filled.
    movement=np.round(np.abs((exit[:,0]+exit[:,1]-entry[:,0]-entry[:,1])*50),9)
    return np.select([~good,movement==0,movement<1],['unknown','unchanged','positive_under1c'],default='at_least1c')


def spread_bins(history,fit_rows,rows):
    spread=np.round((history[:,-1,1]-history[:,-1,0])*100,9)
    edges=np.unique(np.quantile(spread[fit_rows],[.25,.5,.75]))
    return np.searchsorted(edges,spread[rows],side='right').astype(str),edges.tolist()


def pool_side_groups(frame):
    dimensions=[c.removesuffix('_group') for c in frame if c.endswith('_group')]
    records=[]
    for (horizon,model),block in frame.groupby(['horizon','model']):
        for dimension in dimensions:
            for group,part in block.groupby(f'{dimension}_group'):
                part=part.sort_values(['original_row','side'])
                rows=part.original_row.to_numpy()
                if len(part)%2 or not np.array_equal(part.side.to_numpy(),np.tile([0,1],len(part)//2)) or not np.array_equal(rows[::2],rows[1::2]) or len(np.unique(rows))!=len(part)//2:
                    raise ValueError('Pooled side rows are duplicated or unpaired')
                arrays=[part[column].to_numpy().reshape(-1,2) for column in ('prediction_cents','baseline_cents','lower_cents','upper_cents','status')]
                records.append(dict(horizon=int(horizon),model=model,dimension=dimension,group=str(group),
                    fold='pooled_unique',postprediction_diagnostic=dimension in ('future_midpoint_movement','entry_outcome_availability'),
                    **error_stats(*arrays,part.event_proxy.to_numpy()[::2])))
    return pd.DataFrame(records)


def summarize(output,source,assignment,study):
    accepted=json.loads((output/'audit.json').read_text())
    if accepted['status']!='passed' or accepted['scope']!='authorized_development_only':
        raise ValueError('Completed development audit required')
    if digest(assignment/'development/metadata.parquet')!=json.loads((output/'identity_reproduction.json').read_text())['expected_metadata_sha256']:
        raise ValueError('Development identity artifact changed')
    data=Data(assignment/'development')
    frame=pd.read_parquet(output/'oof_side_errors.parquet')
    if frame.duplicated(['horizon','model','original_row','side']).any():
        raise ValueError('OOF side observation appears twice')
    for horizon in (30,60):
        mask=frame.horizon==horizon
        labels=movement_groups(data.future,data.known,horizon)
        frame.loc[mask,'future_midpoint_movement_group']=labels[frame.loc[mask,'original_row'].to_numpy()]
    for fold in sorted(frame.fold.unique()):
        mask=frame.fold==fold
        fit_rows=np.load(study/'folds'/str(fold)/'fit.npy')
        bins,edges=spread_bins(data.history[:,-60:],fit_rows,frame.loc[mask,'original_row'].to_numpy())
        frame.loc[mask,'spread_group']=bins
        schema=json.loads((output/f'fold{fold}_diagnostic_schema.json').read_text())
        schema['quantile_edges']['spread']=edges
        schema['spread_rounding']='9decimal cents before fitquantiles/binassignment; quotevalues unchanged'
        save_json(output/f'canonical_fold{fold}_schema.json',schema)
    frame.to_parquet(output/'pooled_side_errors.parquet',index=False)
    pooled=pool_side_groups(frame)
    pooled.to_csv(output/'pooled_error_groups.csv',index=False)
    # Preserve the first-pass output and publish threshold-normalized fold diagnostics.
    original_groups=pd.read_csv(output/'oof_error_groups.csv')
    canonical=[original_groups.loc[~original_groups.dimension.isin(['future_midpoint_movement','spread'])]]
    for fold,part in frame.groupby('fold'):
        part=part.drop(columns=[c for c in part if c.endswith('_group') and c not in ('future_midpoint_movement_group','spread_group')])
        corrected=pool_side_groups(part)
        corrected['fold']=fold
        canonical.append(corrected)
    pd.concat(canonical,ignore_index=True).to_csv(output/'canonical_fold_error_groups.csv',index=False)
    decomposition=pd.read_csv(output/'closed_return_decomposition.csv')
    records=[]
    for horizon,part in decomposition.groupby('horizon'):
        total=part.signed_baseline_error_cents.to_numpy()
        mid=part.future_midpoint_component_cents.to_numpy()
        spread=part.future_spread_component_cents.to_numpy()
        np.testing.assert_allclose(total,mid+spread,atol=1e-7,rtol=0)
        def correlation(a,b):
            return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
        records.append(dict(horizon=int(horizon),closed_sides=len(part),
            mean_absolute_baseline_error=float(np.abs(total).mean()),
            mean_absolute_midpoint_component=float(np.abs(mid).mean()),
            mean_absolute_spread_component=float(np.abs(spread).mean()),
            signed_midpoint_component_mean=float(mid.mean()),signed_spread_component_mean=float(spread.mean()),
            pearson_absolute_error_midpoint=correlation(np.abs(total),np.abs(mid)),
            pearson_absolute_error_spread=correlation(np.abs(total),np.abs(spread)),
            interpretation='Closed-outcome algebraic association; absolute components can cancel and are not causal shares'))
    pd.DataFrame(records).to_csv(output/'decomposition_summary.csv',index=False)
    choices=json.loads((study/'choices.json').read_text())
    provenance={'scope':'authorized_development_only; no final outcome arrays',
        'source_metadata':{name:digest(source/name) for name in ('input_manifest.json','protected_v2_period.json','windows.parquet','development_source.json')},
        'assignment_metadata':{name:digest(assignment/name) for name in ('segments.json','development_sessions.parquet')},
        'development_artifacts':{p.name:digest(p) for p in sorted((assignment/'development').glob('*')) if p.is_file()},
        'study_metadata':{name:digest(study/name) for name in ('choices.json','choices.sha256','protocol.json')},
        'folds':{str(p.relative_to(study)):digest(p) for p in sorted((study/'folds').rglob('*.npy'))},
        'oof_payloads':{str(path.relative_to(study)):digest(path) for h in (30,60)
            for candidate in (choices['primary_winners'][str(h)]['id'],f'ensemble_{h}') for i in range(3)
            for path in [study/'search'/candidate/f'fold{i}'/'validation.npz']},
        'implementation':{name:digest(Path(__file__).parent/name) for name in ('coverage_error_audit.py','coverage_error_summary.py')},
        'numerical_groups':'Absolute midpoint changes and currentspread normalized to9decimal cents for threshold/fitquantile arithmetic only; no quote imputation',
        'pooled_groups':'Fit-quantile rank bins and series vocabulary remain fold-specific; pooled groups summarize ranks, not one common numerical threshold'}
    save_json(output/'provenance.json',provenance)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('artifacts/accuracy_source_20261109'))
    parser.add_argument('--assignment',type=Path,default=Path('artifacts/accuracy_experiment_20261109'))
    parser.add_argument('--study',type=Path,default=Path('artifacts/residual_accuracy_20261109'))
    args=parser.parse_args()
    print(json.dumps(summarize(args.output,args.source,args.assignment,args.study),indent=2))
