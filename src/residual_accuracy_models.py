"""Reloadable direct/residual neural return predictors; no split selection."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model_benchmark import summarize


ARCHITECTURES = ('gru', 'tcn', 'mlp')
LOSSES = ('huber1', 'huber5', 'l1')


def nochange(history):
    """Forecast-time constant-book roundtrip, cents, both YES and NO sides."""
    h = np.asarray(history)
    if h.ndim != 3 or h.shape[1:] != (60, 7) or not np.isfinite(h).all():
        raise ValueError('Expected finite Nx60x7 history')
    spread = (h[:, -1, 1]-h[:, -1, 0])*100
    if (spread < 0).any():
        raise ValueError('Crossed book')
    return np.repeat((-spread-4)[:, None], 2, axis=1)


def loss_function(name):
    import tensorflow as tf
    if name not in LOSSES:
        raise ValueError('Unknown interval loss')
    delta = .01 if name == 'huber1' else .05
    def loss(bounds, prediction):
        lo, hi = bounds[:, :2], bounds[:, 2:]
        residual = prediction-tf.clip_by_value(prediction, lo, hi)
        absolute = tf.abs(residual)
        values = absolute if name == 'l1' else tf.where(absolute <= delta,
            .5*tf.square(residual), delta*(absolute-.5*delta))
        return tf.reduce_mean(values, axis=-1)
    return loss


def architecture(name, shape, layers, dropout):
    import tensorflow as tf
    x = inputs = tf.keras.Input(shape=shape)
    if name == 'gru':
        x = tf.keras.layers.GRU(layers[0], return_sequences=True)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.GRU(layers[1])(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif name == 'tcn':
        for dilation in (1, 2, 4):
            residual = tf.keras.layers.Conv1D(32, 1)(x)
            x = tf.keras.layers.Conv1D(32, 3, padding='causal', dilation_rate=dilation,
                                      activation='relu')(x)
            x = tf.keras.layers.Dropout(.1)(x)
            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif name == 'mlp':
        for width in (128, 64):
            x = tf.keras.layers.Dense(width, activation='relu')(x)
            x = tf.keras.layers.Dropout(.1)(x)
    else:
        raise ValueError('Unknown architecture')
    return tf.keras.Model(inputs, tf.keras.layers.Dense(2)(x))


def inputs(history, arch):
    return summarize(history) if arch == 'mlp' else np.asarray(history, dtype=np.float32)


def fit(history, lower, upper, fit_rows, calibration_rows, spec, seed, folder, epochs=30):
    import tensorflow as tf
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if spec['architecture'] not in ARCHITECTURES or spec['loss'] not in LOSSES or spec['target'] not in ('direct', 'residual'):
        raise ValueError('Unknown model specification')
    fit_rows, calibration_rows = np.asarray(fit_rows), np.asarray(calibration_rows)
    if not len(fit_rows) or not len(calibration_rows) or np.intersect1d(fit_rows, calibration_rows).size:
        raise ValueError('Empty or overlapping fit/calibration rows')
    for rows in (fit_rows, calibration_rows):
        if rows.ndim != 1 or rows.min() < 0 or rows.max() >= len(history) or len(np.unique(rows)) != len(rows):
            raise ValueError('Invalid row identity')
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (len(history), 2) or upper.shape != lower.shape or not np.isfinite([lower, upper]).all() or (lower > upper).any():
        raise ValueError('Invalid interval bounds')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    x = inputs(history, spec['architecture'])
    scaler = StandardScaler().fit(x[fit_rows].reshape(-1, x.shape[-1]))
    def scaled(rows):
        return scaler.transform(x[rows].reshape(-1, x.shape[-1])).reshape(x[rows].shape).astype(np.float32)
    baseline = nochange(history) if spec['target'] == 'residual' else np.zeros_like(lower)
    y = np.concatenate([(lower-baseline)/100, (upper-baseline)/100], axis=1).astype(np.float32)
    model = architecture(spec['architecture'], x.shape[1:], spec['layers'], spec['dropout'])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss=loss_function(spec['loss']))
    model.fit(scaled(fit_rows), y[fit_rows], validation_data=(scaled(calibration_rows), y[calibration_rows]),
        batch_size=256, epochs=epochs, verbose=0, callbacks=[
            tf.keras.callbacks.CSVLogger(str(folder/'epochs.csv')),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.ModelCheckpoint(str(folder/'best.keras'), monitor='val_loss', save_best_only=True)])
    joblib.dump(scaler, folder/'scaler.joblib')
    (folder/'spec.json').write_text(json.dumps(spec), encoding='utf-8')
    return load(folder)


def load(folder):
    import tensorflow as tf
    folder = Path(folder)
    return {'model': tf.keras.models.load_model(folder/'best.keras', compile=False),
            'scaler': joblib.load(folder/'scaler.joblib'), 'spec': json.loads((folder/'spec.json').read_text())}


def predict(fitted, history):
    x = inputs(history, fitted['spec']['architecture'])
    x = fitted['scaler'].transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    result = np.concatenate([fitted['model'](x[i:i+256], training=False).numpy()*100 for i in range(0, len(x), 256)])
    if fitted['spec']['target'] == 'residual':
        result = result+nochange(history)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite absolute predictions')
    return result
