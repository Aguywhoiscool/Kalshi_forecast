import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.residual_accuracy_models import architecture, fit, load, predict, nochange, loss_function
from src.residual_accuracy_experiment import specs, identifier, diagnostic_schema, diagnostics, richer_metrics


def histories(n=12):
    h = np.zeros((n, 60, 7))
    h[:, :, 0] = .40+np.arange(n)[:, None]*.001
    h[:, :, 1] = h[:, :, 0]+.02
    h[:, :, 2] = .02
    h[:, :, 3] = (h[:, :, 0]+h[:, :, 1])/2
    h[:, :, 4:6] = 100
    return h


@pytest.mark.parametrize('loss,expected,gradient', [('huber1',.000375,.005),('huber5',.001375,.025),('l1',.04,.5)])
def test_actual_tensor_loss_and_gradient(loss, expected, gradient):
    import tensorflow as tf
    prediction = tf.Variable([[.08, .03]], dtype=tf.float32)
    bounds = tf.constant([[0., -.1, 0., .1]])
    with tf.GradientTape() as tape:
        value = tf.reduce_mean(loss_function(loss)(bounds, prediction))
    np.testing.assert_allclose(value.numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(tape.gradient(value, prediction).numpy(), [[gradient, 0]], rtol=1e-6)


@pytest.mark.parametrize('arch,target', [('gru','direct'),('tcn','residual'),('mlp','residual')])
def test_real_fit_reload_fit_only_scaling_and_residual_reconstruction(tmp_path, arch, target):
    h = histories()
    lo = np.tile([-6.,-6.], (12,1))
    hi = lo.copy()
    hi[2] = 20
    spec = dict(architecture=arch,target=target,loss='huber1',layers=[4,3],dropout=.1)
    fitted = fit(h,lo,hi,np.arange(8),np.arange(8,10),spec,42,tmp_path,epochs=2)
    actual = predict(fitted,h[10:])
    np.testing.assert_allclose(actual,predict(load(tmp_path),h[10:]),rtol=1e-6)
    expected_count = 8 if arch == 'mlp' else 8*60
    assert fitted['scaler'].n_samples_seen_ == expected_count
    if target == 'residual':
        from src.residual_accuracy_models import inputs
        x = inputs(h[10:],arch)
        scaled = fitted['scaler'].transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
        network = fitted['model'](scaled.astype('float32'),training=False).numpy()*100
        np.testing.assert_allclose(actual,network+nochange(h[10:]),rtol=1e-6)
    assert len(pd.read_csv(tmp_path/'epochs.csv')) == 2


def test_tcn_intermediate_prefix_is_causal():
    import tensorflow as tf
    model = architecture('tcn',(60,7),[4,3],.1)
    block = next(layer for layer in reversed(model.layers) if isinstance(layer,tf.keras.layers.Activation))
    prefix = tf.keras.Model(model.input,block.output)
    h = histories(1).astype('float32')
    changed = h.copy()
    changed[:,30:] += 1000
    np.testing.assert_allclose(prefix(h,training=False).numpy()[:,:30],prefix(changed,training=False).numpy()[:,:30],atol=1e-6)
    convs = [l for l in model.layers if isinstance(l,tf.keras.layers.Conv1D) and l.kernel_size==(3,)]
    assert [l.dilation_rate for l in convs] == [(1,),(2,),(4,)]
    assert all(l.padding=='causal' for l in convs)


def test_exact_registry_and_nochange_and_fit_defined_diagnostic_bins():
    old = {'primary_winners':{'30':{'params':{'layers':[64,32],'dropout':.1}},'60':{'params':{'layers':[96,48],'dropout':.3}}}}
    assert sum(len(specs(h,old))*3 for h in (30,60)) == 108
    assert len({identifier(h,s) for h in (30,60) for s in specs(h,old)}) == 36
    h = histories(4)
    np.testing.assert_allclose(nochange(h),-6)
    meta = pd.DataFrame(dict(seconds_to_close=[10.,20.,999.,1000.],series=['a','b','new','new']))
    schema = diagnostic_schema(h,meta,np.array([0,1]))
    assert schema['quantile_edges']['time_to_close'] == [12.5,15.,17.5]
    assert schema['training_series'] == ['a','b']
    lo = np.full((4,2),-6.)
    status = np.array([[0,1],[2,3],[4,0],[1,0]])
    records = diagnostics(nochange(h),lo,lo+2,status,h,meta,schema,'test')
    assert any(r.get('group')=='__unseen__' for r in records)
    metrics = richer_metrics(nochange(h),lo,lo+2,status)
    assert metrics['observed_sides']==5 and metrics['signed_bias_cents']==pytest.approx(0)
    assert metrics['p90_absolute_error_cents']==pytest.approx(0)


def test_actual_ensemble_rejects_permuted_oof_identities():
    from src.residual_accuracy_experiment import ensemble_oof
    rows = np.array([4,8])
    members = [dict(rows=rows.copy(),prediction=np.full((2,2),v)) for v in (1.,2.,6.)]
    np.testing.assert_allclose(ensemble_oof(members,rows),3)
    members[1]['rows'] = rows[::-1]
    with pytest.raises(ValueError,match='identities'):
        ensemble_oof(members,rows)


def test_existing_setup_rejects_consumed_non_accuracy_final(tmp_path):
    from src.residual_accuracy_experiment import setup
    from src.v3_research import digest
    source,output,prior,consumed = [tmp_path/name for name in ('source','new','prior','residual_previous')]
    for folder in (source,output,prior,consumed):
        folder.mkdir()
    (source/'input_manifest.json').write_text('{}')
    (output/'segments.json').write_text(json.dumps(dict(source=str(source.resolve()),
        source_manifest_sha256=digest(source/'input_manifest.json'), final_cutoff_ns=100)))
    (consumed/'segments.json').write_text(json.dumps(dict(final_end_ns=100)))
    (consumed/'final_labels_opened.json').write_text('{}')
    with pytest.raises(ValueError,match='not fresh'):
        setup(output,source,prior)


@pytest.mark.parametrize('tamper', ['prior_control','protected_source'])
def test_frozen_protocol_rejects_comparator_or_protection_tamper(tmp_path,tamper):
    from src.residual_accuracy_experiment import freeze_protocol
    output,prior,source = [tmp_path/name for name in ('output','prior','source')]
    for folder in (output,prior,source,output/'development'):
        folder.mkdir()
    (output/'segments.json').write_text(json.dumps(dict(source=str(source))))
    for name in ('development_sessions.parquet','final_sessions.parquet'):
        (output/name).write_bytes(b'fixedmetadata')
    for name in ('input_manifest.json','windows.parquet','protected_v2_period.json'):
        (source/name).write_bytes(b'fixedsource')
    (prior/'choices.json').write_text(json.dumps({'primary_winners':{str(h):{'params':{'layers':[4,3],'dropout':.1}} for h in (30,60)}}))
    for name in ('refits.json','choices.sha256','protocol.json','segments.json'):
        (prior/name).write_text('{}')
    freeze_protocol(output,prior)
    target = prior/'refits.json' if tamper=='prior_control' else source/'protected_v2_period.json'
    target.write_text('changed')
    with pytest.raises(ValueError,match='Frozen protocol'):
        freeze_protocol(output,prior)
