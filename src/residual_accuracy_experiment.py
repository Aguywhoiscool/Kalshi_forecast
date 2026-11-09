"""Versioned residual/direct neural study; old final outcomes are never reused."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from src.accuracy_data import declare_segments, prepare, walkforward
from src.accuracy_models import Data, load_model, raw_predict
from src.accuracy_targets import correction, error_metrics, event_bootstrap
from src.accuracy_experiment import progress
from src.residual_accuracy_models import ARCHITECTURES, LOSSES, nochange, fit, load, predict
from src.swing_return_experiment import return_bounds, evaluate
from src.v3_research import digest, save_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def specs(horizon, prior_choices):
    chosen = prior_choices['primary_winners'][str(horizon)]['params']
    return [dict(architecture=architecture, target=target, loss=loss,
        layers=chosen['layers'] if architecture=='gru' else [32,32,32] if architecture=='tcn' else [128,64],
        dropout=chosen['dropout'] if architecture=='gru' else .1,
        network_output_units='dollars', interval_huber_delta_cents=1 if loss=='huber1' else 5 if loss=='huber5' else None,
        baseline_units='cents;nochange=-100*(ask-bid)-4;residual bounds subtract before dollar scaling')
        for architecture, target, loss in itertools.product(ARCHITECTURES, ('direct', 'residual'), LOSSES)]


def identifier(horizon, spec):
    return f"{spec['architecture']}_{horizon}_{spec['target']}_{spec['loss']}"


def richer_metrics(prediction, lower, upper, status):
    result = error_metrics(prediction, lower, upper, status)
    mask = np.isin(status, [0, 1])
    errors = (prediction-lower)[mask]
    result.update(signed_bias_cents=float(errors.mean()) if len(errors) else None,
        median_absolute_error_cents=float(np.median(np.abs(errors))) if len(errors) else None,
        p90_absolute_error_cents=float(np.quantile(np.abs(errors), .9)) if len(errors) else None)
    return result


def diagnostic_schema(history, meta, fit_rows):
    variables = {'spread': (history[:, -1, 1]-history[:, -1, 0])*100,
        'volatility': np.diff(history[:, :, 3], axis=1).std(axis=1)*100,
        'time_to_close': meta.seconds_to_close.to_numpy()}
    return {'quantile_edges': {key: np.unique(np.quantile(value[fit_rows], [.25, .5, .75])).tolist()
            for key, value in variables.items()},
            'training_series': sorted(meta.iloc[fit_rows].series.unique().tolist()),
            'definition': 'Numeric quartile edges and series vocabulary from fit rows only; availability is postprediction diagnostic, never input.'}


def diagnostics(prediction, lower, upper, status, history, meta, schema, name):
    variables = {'spread': (history[:, -1, 1]-history[:, -1, 0])*100,
        'volatility': np.diff(history[:, :, 3], axis=1).std(axis=1)*100,
        'time_to_close': meta.seconds_to_close.to_numpy()}
    labels = {key: np.searchsorted(schema['quantile_edges'][key], value, side='right').astype(str)
              for key, value in variables.items()}
    labels['series'] = np.where(meta.series.isin(schema['training_series']), meta.series, '__unseen__')
    result = []
    for dimension, values in labels.items():
        for group in np.unique(values):
            mask = values == group
            result.append(dict(name=name, dimension=dimension, group=str(group), rows=int(mask.sum()),
                **richer_metrics(prediction[mask], lower[mask], upper[mask], status[mask])))
    for group in range(5):
        side_mask = status == group
        if side_mask.any():
            metrics = richer_metrics(prediction[side_mask], lower[side_mask], upper[side_mask], status[side_mask])
            distance = np.abs(prediction-np.clip(prediction, lower, upper))[side_mask]
            result.append(dict(name=name, dimension='availability', group=str(group), sides=int(side_mask.sum()),
                selected_interval_distance_mae=float(distance.mean()), **metrics))
    return result


def setup(output, source, previous):
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'segments.json').exists():
        segments = declare_segments(source, output, Path(json.loads((previous/'segments.json').read_text())['source']))
        registry = []
        for marker in sorted(output.parent.glob('*/final_labels_opened.json')):
            path = marker.parent/'segments.json'
            if not path.exists():
                raise ValueError('Consumed final marker lacks segment provenance')
            value = json.loads(path.read_text())
            registry.append(dict(run=str(marker.parent.resolve()), final_end_ns=value['final_end_ns'],
                opened_marker_sha256=digest(marker), segments_sha256=digest(path)))
        lower = max([segments['freshness_lower_bound_ns'], *[r['final_end_ns'] for r in registry]])
        cutoff = max(segments['final_cutoff_ns'], lower+70*10**9)
        if cutoff != segments['final_cutoff_ns']:
            sessions = pd.read_parquet(source/'windows.parquet')
            final = sessions.loc[sessions.input_start >= cutoff]
            development = sessions.loc[sessions.target_end <= cutoff-70*10**9]
            final.to_parquet(output/'final_sessions.parquet',index=False)
            development.to_parquet(output/'development_sessions.parquet',index=False)
            proxies = set()
            for session in final.session_id:
                tickers = pd.read_csv(source/'snapshot'/f'features_{session}.csv',usecols=['ticker']).ticker.dropna().unique()
                proxies.update(str(t).rsplit('-',1)[0] for t in tickers)
            segments.update(final_cutoff_ns=cutoff, development_end_ns=cutoff-70*10**9,
                final_end_ns=int(final.target_end.max()) if len(final) else cutoff,
                final_sessions=len(final), development_sessions=len(development), final_event_proxies=len(proxies),
                mode='final_eligible' if len(proxies)>=20 else 'development_only',
                development_metadata_sha256=digest(output/'development_sessions.parquet'),
                final_metadata_sha256=digest(output/'final_sessions.parquet'))
        segments.update(previously_opened_final_registry=registry, freshness_lower_bound_ns=lower)
        save_json(output/'segments.json',segments)
    segments = json.loads((output/'segments.json').read_text())
    if segments['source'] != str(source.resolve()) or segments['source_manifest_sha256'] != digest(source/'input_manifest.json'):
        raise ValueError('Existing frozen source differs')
    for marker in sorted(output.parent.glob('*/final_labels_opened.json')):
        if marker.parent.resolve() == output.resolve():
            continue
        consumed = json.loads((marker.parent/'segments.json').read_text())['final_end_ns']
        if segments['final_cutoff_ns'] < consumed+70*10**9:
            raise ValueError('Final is not fresh after consumed prior final')
    if not (output/'development').exists():
        shutil.copytree(previous/'development', output/'development')
    expected = {p.name:digest(p) for p in (previous/'development').glob('*') if p.is_file()}
    actual = {p.name:digest(p) for p in (output/'development').glob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Copied matched development cohort changed')


def freeze_protocol(output, previous):
    root = Path(__file__).resolve().parents[1]
    source = Path(json.loads((output/'segments.json').read_text())['source'])
    old_choices = json.loads((previous/'choices.json').read_text())
    value = {'created_utc': utc(), 'development_origin': str(previous.resolve()),
        'scope': 'Same6054 development cohort; no prior final used for research or evaluation. Fresh final strictly after every consumed accuracy final+70s.',
        'search': {str(h): specs(h, old_choices) for h in (30, 60)},
        'training': {'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 256, 'max_epochs': 30, 'patience': 5,
                     'search_seed': 42, 'refit_seeds': [7, 42, 101], 'primary_seed': 42},
        'calibration': 'Reconstruct absolute cents from residual baseline first, clip[-104,96], one-pass mean interval-projection residual over calibration, then clip absolute predictions[-104,96]. Same rule for every loss.',
        'architecture': {'gru': 'Prior development-selected per-horizon layers/dropout;60x7',
            'tcn': '3residual causalConv1D blocks32channels,kernel3,dilations1/2/4,dropout.1,globalaveragepool;60x7',
            'mlp': '50summaries from60x7;Dense128/64ReLU,dropout.1'},
        'selection': 'Minimum average3fold observed-side MAE per architecture. Equal-weight3architecture ensemble from exactsame OOF rows; choose primary among3architectures+ensemble bydevMAE. OOF architecture selection optimism acknowledged.',
        'baseline': 'Nochange = -100*(currentask-currentbid)-4 bothsides; strongest of nochange and9prior default OOF baselines selected onexactsame3fold rows. Frozen prior selectedGRU also mandatory comparator.',
        'gates': 'Primaryseed42>=5%relativeMAE and paired1000eventproxybootstrap95%lowerCI>0 versus BOTH frozenpriorGRU andstrongestdevbaseline;>=20observedgroups. Unadjustedexploratory, notpowerproof. Noautomaticpromotion.',
        'development_artifacts_sha256': {p.name: digest(p) for p in sorted((output/'development').glob('*')) if p.is_file()},
        'segments_sha256': digest(output/'segments.json'),
        'source_inputs_sha256': {name:digest(source/name) for name in ('input_manifest.json','windows.parquet','protected_v2_period.json')},
        'segment_metadata_sha256': {name:digest(output/name) for name in ('development_sessions.parquet','final_sessions.parquet')},
        'prior_choices_sha256': digest(previous/'choices.json'),
        'prior_control_files_sha256': {name:digest(previous/name) for name in ('refits.json','choices.json','choices.sha256','protocol.json','segments.json')},
        'dependencies_sha256': {name: digest(root/'src'/name) for name in (
            'residual_accuracy_models.py', 'residual_accuracy_experiment.py', 'accuracy_data.py', 'accuracy_models.py',
            'accuracy_targets.py', 'accuracy_experiment.py', 'causal_features.py', 'causal_swing_data.py',
            'swing_return_experiment.py', 'model_benchmark.py', 'model_traces.py', 'v3_research.py')},
        'prior_artifacts_sha256': {str(p.relative_to(previous)): digest(p) for base in ('search', 'refits')
            for p in sorted((previous/base).rglob('*')) if p.is_file()}}
    path = output/'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        value['created_utc'] = old['created_utc']
        if value != old:
            raise ValueError('Frozen protocol/dependency changed; use new run')
    else:
        save_json(path, value)
    return value


def calibrated(fitted, history, rows, bias):
    return np.clip(np.clip(predict(fitted, history[rows]), -104, 96)+bias, -104, 96)


def ensemble_oof(members, rows):
    if len(members) != 3 or any(not np.array_equal(member['rows'],rows) for member in members):
        raise ValueError('Ensemble OOF row identities differ')
    arrays = [member['prediction'] for member in members]
    if any(a.shape != (len(rows),2) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid ensemble predictions')
    return np.mean(arrays,axis=0)


def crossvalidate(output, data, folds, spec, horizon):
    name = identifier(horizon, spec)
    summary_path = output/'search'/name/'summary.json'
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result['protocol_sha256'] != digest(output/'protocol.json') or result['spec'] != spec:
            raise ValueError('Cached search identity changed')
        return result
    lower, upper, status = return_bounds(data.future, data.known, horizon)
    history = data.history[:, -60:]
    scores = []
    for index, fold in enumerate(folds):
        progress(output, 'residual_search', candidate=name, fold=index+1, total_fits=108)
        folder = output/'search'/name/f'fold{index}'
        fitted = fit(history, lower, upper, fold['fit'], fold['calibration'], spec, 42, folder)
        raw_cal = np.clip(predict(fitted, history[fold['calibration']]), -104, 96)
        bias = correction(raw_cal, lower[fold['calibration']], upper[fold['calibration']], 'interval')
        val = fold['validation']
        prediction = calibrated(fitted, history, val, bias)
        metrics = richer_metrics(prediction, lower[val], upper[val], status[val])
        scores.append(metrics['observed_mae'])
        np.savez_compressed(folder/'validation.npz', rows=val, prediction=prediction,
            raw_calibration=raw_cal, correction=bias)
        schema = diagnostic_schema(history, data.meta, fold['fit'])
        save_json(folder/'diagnostic_schema.json', schema)
        save_json(folder/'diagnostics.json', diagnostics(prediction, lower[val], upper[val], status[val],
            history[val], data.meta.iloc[val], schema, name))
        save_json(folder/'metrics.json', metrics)
    result = dict(id=name, spec=spec, horizon=horizon, scores=scores,
        mean_observed_mae=float(np.mean(scores)), protocol_sha256=digest(output/'protocol.json'))
    save_json(summary_path, result)
    return result


def baseline_oof(previous, data, folds, horizon):
    old_choices = json.loads((previous/'choices.json').read_text())
    names = [('nochange', None)]
    names += [(f'default_{m}', f'default_{m}_{horizon}') for m in ('mean','decision_tree','random_forest','monte_carlo_60','monte_carlo_180')]
    names += [(f'default_{m}', f'ablation_{m}_{horizon}_seven60_conservative') for m in ('ridge','sgd','gradient_boosting','gru')]
    names += [('current_gru', old_choices['selections'][f'gru_{horizon}']['id'])]
    lower, upper, status = return_bounds(data.future, data.known, horizon)
    result = {}
    for name, candidate in names:
        predictions, scores = [], []
        for index, fold in enumerate(folds):
            rows = fold['validation']
            if candidate is None:
                prediction = nochange(data.history[rows, -60:])
            else:
                saved = np.load(previous/'search'/candidate/f'fold{index}'/'validation.npz')
                if not np.array_equal(saved['rows'], rows):
                    raise ValueError('Previous baseline OOF cohort differs')
                prediction = saved['prediction']
            predictions.append(prediction)
            scores.append(richer_metrics(prediction, lower[rows], upper[rows], status[rows])['observed_mae'])
        result[name] = dict(name=name, candidate=candidate, scores=scores,
            mean_observed_mae=float(np.mean(scores)), predictions=predictions)
    return result


def run(output, source, previous):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    output, source, previous = output.resolve(), source.resolve(), previous.resolve()
    if (output/'choices.json').exists() or (output/'final_started.json').exists():
        raise ValueError('Choices already frozen; no adaptive rerun')
    setup(output, source, previous)
    protocol = freeze_protocol(output, previous)
    data = Data(output/'development')
    history = data.history[:, -60:]
    folds = walkforward(data.meta)
    for index, fold in enumerate(folds):
        folder = output/'folds'/str(index)
        folder.mkdir(parents=True, exist_ok=True)
        for key, rows in fold.items():
            np.save(folder/f'{key}.npy', rows)
    selections, winners, baselines = {}, {}, {}
    for horizon in (30, 60):
        results = [crossvalidate(output, data, folds, spec, horizon) for spec in protocol['search'][str(horizon)]]
        chosen = {arch: min([r for r in results if r['spec']['architecture'] == arch], key=lambda r:r['mean_observed_mae']) for arch in ARCHITECTURES}
        selections[str(horizon)] = chosen
        lower, upper, status = return_bounds(data.future, data.known, horizon)
        ensemble_scores = []
        for index, fold in enumerate(folds):
            members = [np.load(output/'search'/chosen[arch]['id']/f'fold{index}'/'validation.npz') for arch in ARCHITECTURES]
            prediction = ensemble_oof(members,fold['validation'])
            folder = output/'search'/f'ensemble_{horizon}'/f'fold{index}'
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'validation.npz', rows=fold['validation'], prediction=prediction)
            ensemble_scores.append(richer_metrics(prediction, lower[fold['validation']], upper[fold['validation']], status[fold['validation']])['observed_mae'])
        ensemble = dict(id=f'ensemble_{horizon}', spec={'architecture':'ensemble'}, horizon=horizon,
            scores=ensemble_scores, mean_observed_mae=float(np.mean(ensemble_scores)))
        winners[str(horizon)] = min([*chosen.values(), ensemble], key=lambda r:r['mean_observed_mae'])
        old = baseline_oof(previous, data, folds, horizon)
        baselines[str(horizon)] = min([v for k,v in old.items() if k != 'current_gru'], key=lambda r:r['mean_observed_mae'])
        save_json(output/f'baseline_oof_{horizon}.json', {k:{a:b for a,b in v.items() if a != 'predictions'} for k,v in old.items()})
        for name, entry in old.items():
            for index, predictions in enumerate(entry['predictions']):
                folder = output/'controls'/name/str(horizon)/f'fold{index}'
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/'validation.npz', rows=folds[index]['validation'], prediction=predictions)
        del baselines[str(horizon)]['predictions']
    save_json(output/'choices.json', dict(frozen_utc=utc(), selections=selections, primary_winners=winners,
        primary_baselines=baselines, protocol_sha256=digest(output/'protocol.json'),
        segments_sha256=digest(output/'segments.json'), development_metadata_sha256=digest(output/'development/metadata.parquet')))
    (output/'choices.sha256').write_text(digest(output/'choices.json'), encoding='utf-8')
    sessions = data.meta.groupby('session_id').input_start.min().sort_values()
    fit_sessions = set(sessions.index[:int(len(sessions)*.8)])
    fit_rows = data.meta.index[data.meta.session_id.isin(fit_sessions)].to_numpy()
    cal_rows = data.meta.index[~data.meta.session_id.isin(fit_sessions) & (data.meta.input_start >= data.meta.iloc[fit_rows].target_end.max()+70*10**9)].to_numpy()
    np.save(output/'refit_fit.npy', fit_rows)
    np.save(output/'refit_calibration.npy', cal_rows)
    save_json(output/'diagnostic_schema.json', diagnostic_schema(history, data.meta, fit_rows))
    refits = []
    for horizon in (30, 60):
        lower, upper, _ = return_bounds(data.future, data.known, horizon)
        for arch in ARCHITECTURES:
            spec = selections[str(horizon)][arch]['spec']
            for seed in (7, 42, 101):
                name = f'{arch}_{horizon}_seed{seed}'
                progress(output, 'residual_refit', name=name)
                folder = output/'refits'/name
                fitted = fit(history, lower, upper, fit_rows, cal_rows, spec, seed, folder)
                raw_cal = np.clip(predict(fitted, history[cal_rows]), -104, 96)
                bias = correction(raw_cal, lower[cal_rows], upper[cal_rows], 'interval')
                save_json(folder/'calibration.json', dict(correction=bias.tolist(), fit_count=len(fit_rows), calibration_count=len(cal_rows)))
                refits.append(dict(name=name, horizon=horizon, architecture=arch, seed=seed, correction=bias.tolist()))
    save_json(output/'refits.json', refits)
    save_json(output/'refit_manifest.json', {str(p.relative_to(output)):digest(p) for p in sorted((output/'refits').rglob('*')) if p.is_file()})
    segments = json.loads((output/'segments.json').read_text())
    if segments['mode'] != 'final_eligible':
        progress(output, 'development_only', reason='Fewer than20 timestamp-only fresh final eventproxies; no final outcomes read')
        return
    freeze_protocol(output, previous)
    with (output/'final_started.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(utc=utc(), choices_sha256=digest(output/'choices.json'), refit_manifest_sha256=digest(output/'refit_manifest.json')), stream)
    if prepare(source, output, 'final') is None:
        progress(output, 'development_only', reason='Fewer than20 actual common180 fresh final eventproxies; no future labels read')
        return
    freeze_protocol(output, previous)
    final = Data(output/'final')
    final_history = final.history[:, -60:]
    predictions, metrics, diagnostic_rows, economics = {}, {}, [], {}
    schema = json.loads((output/'diagnostic_schema.json').read_text())
    prior_refits = json.loads((previous/'refits.json').read_text())
    for entry in refits:
        predictions[entry['name']] = calibrated(load(output/'refits'/entry['name']), final_history, np.arange(len(final.meta)), entry['correction'])
    for horizon in (30, 60):
        for seed in (7, 42, 101):
            predictions[f'ensemble_{horizon}_seed{seed}'] = np.mean([predictions[f'{a}_{horizon}_seed{seed}'] for a in ARCHITECTURES], axis=0)
        predictions[f'nochange_{horizon}'] = nochange(final_history)
        for control, old_name in [('current_gru', f'selected_gru_{horizon}_seed42'),
            ('strongest_default', None if baselines[str(horizon)]['name'] == 'nochange' else f"baseline_{baselines[str(horizon)]['name'].removeprefix('default_')}_{horizon}_seed42")]:
            if old_name is None:
                predictions[f'{control}_{horizon}'] = nochange(final_history)
            else:
                entry = next(e for e in prior_refits if e['name'] == old_name)
                fitted = load_model(previous/'refits'/old_name, entry['model'], entry['family'])
                fitted['horizon'] = horizon
                predictions[f'{control}_{horizon}'] = np.clip(raw_predict(fitted, final, np.arange(len(final.meta)))+entry['correction'], -104, 96)
        lower, upper, status = return_bounds(final.future, final.known, horizon)
        for name, prediction in predictions.items():
            if f'_{horizon}_' not in name and not name.endswith(f'_{horizon}'):
                continue
            metrics[name] = richer_metrics(prediction, lower, upper, status)
            np.save(output/'final'/f'{name}_prediction.npy', prediction)
            if 'seed42' in name or '_seed' not in name:
                diagnostic_rows.extend(diagnostics(prediction, lower, upper, status, final_history, final.meta, schema, name))
                trades, execution = evaluate(name, horizon, prediction, final_history, final.meta, final.future, final.known, 1., 4.)
                economics[name] = execution
                trades.to_parquet(output/'final'/f'{name}_trades.parquet', index=False)
    gates = {}
    for horizon in (30, 60):
        lower, _, status = return_bounds(final.future, final.known, horizon)
        arch = winners[str(horizon)]['spec']['architecture']
        primary = predictions[f'{arch}_{horizon}_seed42']
        comparisons = {name:event_bootstrap(primary, predictions[f'{name}_{horizon}'], lower, status, final.meta.event_proxy)
            for name in ('current_gru', 'strongest_default')}
        gates[str(horizon)] = dict(primary=arch, comparisons=comparisons,
            passes_both_comparators=all(v['passes_both'] for v in comparisons.values()))
    save_json(output/'final_metrics.json', metrics)
    save_json(output/'execution.json', economics)
    save_json(output/'primary_gates.json', gates)
    save_json(output/'seed_spread.json', {f'{arch}_{h}':dict(seeds=[7,42,101],primary_seed=42,
        observed_mae=[metrics[f'{arch}_{h}_seed{s}']['observed_mae'] for s in (7,42,101)],
        interpretation='Robustness only; no best-seed selection and not independent event samples')
        for h,arch in itertools.product((30,60),(*ARCHITECTURES,'ensemble'))})
    pd.DataFrame(diagnostic_rows).to_csv(output/'subgroup_diagnostics.csv', index=False)
    save_json(output/'final_completed.json', dict(utc=utc(), choices_sha256=digest(output/'choices.json'),
        candidates=len(final.meta), event_proxies=int(final.meta.event_proxy.nunique()), promotion='none'))
    progress(output, 'complete', final_rows=len(final.meta), event_proxies=int(final.meta.event_proxy.nunique()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('artifacts/accuracy_experiment_20261109'))
    args = parser.parse_args()
    run(args.output, args.source, args.previous)
