"""Shared TensorFlow protocol; model reuse, finite checks and validation selection."""
import gc
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from kssd.models import build_model
from kssd.tf_metrics import dice_loss, make_metrics
from kssd.metrics import binary_metrics, metrics_from_counts
from kssd.reproducibility import set_seeds
from strain.config import *
from strain.data import save


def loss(truth, prediction):
    # Dice maintains foreground sensitivity; BCE penalizes false positives.
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(truth, prediction), axis=(1, 2))
    return dice_loss(truth, prediction) + .25 * bce


def compile_model(architecture, size, precision='float32'):
    tf.keras.backend.clear_session()
    gc.collect()
    set_seeds(SEED)
    tf.keras.mixed_precision.set_global_policy(precision)
    model = build_model(architecture, (size, size, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss,
                  metrics=make_metrics(THRESHOLD), jit_compile=False)
    return model


def evaluate(model, arrays, ids):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    rows = []
    x, y = arrays
    for offset in range(0, len(x), BATCH):
        prediction = model(x[offset:offset+BATCH], training=False).numpy()
        if prediction.shape != y[offset:offset+BATCH].shape:
            raise ValueError('Model output-mask shape mismatch')
        for name, truth, prob in zip(ids[offset:offset+BATCH], y[offset:offset+BATCH], prediction):
            result = binary_metrics(truth, prob, THRESHOLD)
            rows.append(dict(id=name, **result))
            for key in counts:
                counts[key] += result[key]
    return dict(micro=metrics_from_counts(**counts), counts=counts, per_image=rows,
                macro={k:float(np.mean([r[k] for r in rows])) for k in metrics_from_counts(**counts)})


def dataset(arrays, training=False):
    ds = tf.data.Dataset.from_tensor_slices(arrays)
    if training:
        ds = ds.shuffle(len(arrays[0]), SEED, reshuffle_each_iteration=True)
        def augment(x, y):
            # Concatenation guarantees spatially identical transforms for the pair.
            pair = tf.concat([x, y], -1)
            pair = tf.image.random_flip_left_right(pair, seed=SEED)
            pair = tf.image.random_flip_up_down(pair, seed=SEED+1)
            return pair[..., :1], pair[..., 1:]
        ds = ds.map(augment, num_parallel_calls=1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.experimental_deterministic = True
    return ds.batch(BATCH).with_options(options).prefetch(1)


class Deadline(tf.keras.callbacks.Callback):
    def __init__(self, end):
        super().__init__()
        self.end, self.last_start, self.epoch_seconds = end, None, []
        self.hit = False
    def on_epoch_begin(self, epoch, logs=None):
        self.last_start = time.monotonic()
    def on_train_batch_end(self, batch, logs=None):
        if time.monotonic() >= self.end:
            self.hit = True
            self.model.stop_training = True
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.monotonic() - self.last_start)
        if time.monotonic() + max(self.epoch_seconds[-3:])*1.3 >= self.end:
            self.hit = True
            self.model.stop_training = True


def pilot(architecture, arrays, output, end):
    measurements = []
    # Equal batch size and precision; real backward passes at both resolutions.
    for size in [256, 512]:
        if time.monotonic() > end-120:
            raise TimeoutError('Insufficient pilot time')
        start = time.monotonic()
        model = compile_model(architecture, size)
        init = time.monotonic()-start
        x, y = arrays['train'][0][:2], arrays['train'][1][:2]
        if size == 512:
            x = tf.image.resize(x, (size,size), method='bilinear').numpy()
            y = tf.image.resize(y, (size,size), method='nearest').numpy()
        tf.config.experimental.reset_memory_stats('GPU:0')
        times = []
        for _ in range(3):
            start = time.monotonic()
            values = model.train_on_batch(x, y, return_dict=True)
            times.append(time.monotonic()-start)
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite pilot loss/metrics')
        start = time.monotonic()
        values = model.test_on_batch(x, y, return_dict=True)
        validation = time.monotonic()-start
        prediction = model(x, training=False).numpy()
        assert prediction.shape == y.shape and np.isfinite(prediction).all()
        measurements.append(dict(size=size, batch=2, precision='float32', init_seconds=init,
            step_seconds=times, steady_step_seconds=float(np.mean(times[1:])),
            validation_batch_seconds=validation, memory=tf.config.experimental.get_memory_info('GPU:0')))
        del model
    speedup = measurements[1]['steady_step_seconds']/measurements[0]['steady_step_seconds']
    result = dict(architecture=architecture, measurements=measurements, measured_speedup=speedup,
        note='512 timing uses upsampled cached train pairs; compute comparison only, not accuracy comparison')
    save(output/'pilot.json', result)
    return result


def train_model(architecture, arrays, splits, output, checkpoint, end, timing):
    start = time.monotonic()
    model = compile_model(architecture, 256)
    # Fresh initialization discards all pilot training updates.
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    estimated = timing['measurements'][0]['steady_step_seconds']*2 * np.ceil(len(splits['train'])/BATCH)
    epochs = min(MAX_EPOCHS, max(1, int((end-time.monotonic()-60)/max(estimated*1.5,1))))
    deadline = Deadline(end-30)
    callbacks = [deadline, tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(output/'history.csv')),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_hard_mask_dice',
            mode='max', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_hard_mask_dice', mode='max', patience=PATIENCE,
            start_from_epoch=EARLY_STOPPING_WARMUP),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=6, min_lr=1e-5)]
    fit_start = time.monotonic()
    history = model.fit(dataset(arrays['train'], True), validation_data=dataset(arrays['validation']),
        epochs=epochs, callbacks=callbacks, verbose=2)
    duration = time.monotonic()-fit_start
    save(output/'history.json', history.history)
    if not checkpoint.exists() or not all(np.isfinite(v).all() for v in history.history.values()):
        raise ValueError('No finite selected checkpoint')
    model.load_weights(checkpoint)
    before = model(arrays['validation'][0][:1], training=False).numpy()
    model.load_weights(checkpoint)
    np.testing.assert_array_equal(before, model(arrays['validation'][0][:1], training=False).numpy())
    selected = int(np.argmax(history.history['val_hard_mask_dice']))+1
    result = dict(architecture=architecture, status='time_limited' if deadline.hit else 'complete',
        epochs_completed=len(history.epoch), allocated_epochs=epochs, best_validation_epoch=selected,
        best_validation_dice=float(history.history['val_hard_mask_dice'][selected-1]),
        training_including_validation_seconds=duration, epoch_seconds=deadline.epoch_seconds,
        checkpoint=str(checkpoint), checkpoint_reload_verified=True,
        loss='per-image Dice + 0.25 mean pixel binary cross entropy', optimizer='Adam 0.001; plateau LR',
        precision='float32', batch_size=BATCH, input_shape=[256,256,1], seed=SEED, threshold=THRESHOLD,
        early_stopping_warmup=EARLY_STOPPING_WARMUP)
    if time.monotonic() < end:
        evaluation_start = time.monotonic()
        result['test'] = evaluate(model, arrays['test'], splits['test'])
        result['evaluation_seconds'] = time.monotonic()-evaluation_start
    else:
        result['test'] = None
        result['test_unavailable_reason'] = 'deadline'
    result['total_model_seconds'] = time.monotonic()-start
    result['memory'] = tf.config.experimental.get_memory_info('GPU:0')
    save(output/'result.json', result)
    del model
    return result
