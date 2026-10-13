import gc
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from kssd.models import build_model
from kssd.reproducibility import set_seeds
from kssd.tf_metrics import make_metrics,dice_loss
from strain.config import SEED
from strain.data import save
from strain.train import loss as current_loss
from strain.targeted.diagnose import predictions,evaluate_probabilities,sweep

MODELS=['unetpp','transunet','modified_unet','unet3plus']


def fp_loss(y,p):
    y=tf.cast(y,tf.float32);p=tf.cast(p,tf.float32)
    axes=(1,2,3)
    tp=tf.reduce_sum(y*p,axes);fp=tf.reduce_sum((1-y)*p,axes);fn=tf.reduce_sum(y*(1-p),axes)
    tversky=1-(tp+1e-6)/(tp+.7*fp+.3*fn+1e-6)
    bce=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,p),axis=(1,2))
    return tversky+.25*bce


def compile_model(name,checkpoint,kind,lr):
    if name not in MODELS:raise ValueError('Unknown architecture')
    if kind not in ['dice_bce','fp_tversky_bce']:raise ValueError('Unknown loss')
    tf.keras.backend.clear_session();gc.collect();set_seeds(SEED)
    tf.keras.mixed_precision.set_global_policy('float32')
    model=build_model(name,(256,256,1));model.load_weights(checkpoint)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
        loss=current_loss if kind=='dice_bce' else fp_loss,metrics=make_metrics(.5),jit_compile=False)
    return model


def dataset(arrays,batch,training=False):
    ds=tf.data.Dataset.from_tensor_slices(arrays)
    if training:
        ds=ds.shuffle(len(arrays[0]),SEED,reshuffle_each_iteration=True)
        def augment(x,y):
            pair=tf.concat([x,y],-1)
            pair=tf.image.random_flip_left_right(pair,seed=SEED)
            pair=tf.image.random_flip_up_down(pair,seed=SEED+1)
            return pair[...,:1],pair[...,1:]
        ds=ds.map(augment,num_parallel_calls=2,deterministic=True)
    options=tf.data.Options();options.threading.private_threadpool_size=2
    options.experimental_deterministic=True
    return ds.batch(batch).with_options(options).prefetch(1)


class Timing(tf.keras.callbacks.Callback):
    def __init__(self,end):
        super().__init__();self.end=end;self.epochs=[];self.validation=[];self.full=[];self.hit=False
    def on_epoch_begin(self,epoch,logs=None):
        self.start=time.monotonic();self.batches=0
    def on_test_begin(self,logs=None):self.test_start=time.monotonic()
    def on_test_end(self,logs=None):self.validation.append(time.monotonic()-self.test_start)
    def on_train_batch_end(self,batch,logs=None):
        self.batches+=1
        if time.monotonic()>self.end:
            self.hit=True;self.model.stop_training=True
    def on_epoch_end(self,epoch,logs=None):
        self.epochs.append(time.monotonic()-self.start)
        self.full.append(self.batches==self.params['steps'])
        if time.monotonic()+max(self.epochs[-3:])*1.25>self.end:
            self.hit=True;self.model.stop_training=True


def pilot(name,checkpoint,arrays,output,end):
    start=time.monotonic();model=compile_model(name,checkpoint,'dice_bce',3e-4)
    init=time.monotonic()-start;rows=[];best=None
    for batch in [4,8,16]:
        if time.monotonic()>end-120:break
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
            x,y=arrays['train'][0][:batch],arrays['train'][1][:batch]
            times=[]
            for _ in range(3):
                start=time.monotonic();value=model.train_on_batch(x,y,return_dict=True)
                times.append(time.monotonic()-start)
                assert all(np.isfinite(v) for v in value.values())
            start=time.monotonic()
            # Full validation timing includes device synchronization and batching.
            for i in range(0,len(arrays['validation'][0]),batch):
                p=model(arrays['validation'][0][i:i+batch],training=False).numpy()
                assert p.shape==arrays['validation'][1][i:i+batch].shape and np.isfinite(p).all()
            validation=time.monotonic()-start
            estimate=np.ceil(len(arrays['train'][0])/batch)*float(np.mean(times[1:]))+validation
            row=dict(batch=batch,steps_seconds=times,validation_seconds=validation,
                estimated_epoch_seconds=float(estimate),memory=tf.config.experimental.get_memory_info('GPU:0'))
            rows.append(row)
            if best is None or estimate<best['estimated_epoch_seconds']:best=row
        except (tf.errors.ResourceExhaustedError,AssertionError) as error:
            rows.append(dict(batch=batch,error=repr(error)))
            break
    if best is None:raise RuntimeError('No stable batch size')
    save(output/'pilot.json',dict(architecture=name,initialization_seconds=init,measurements=rows,
        selected_batch=best['batch'],estimated_epoch_seconds=best['estimated_epoch_seconds'],
        precision='float32',pilot_updates_discarded=True))
    del model;tf.keras.backend.clear_session();gc.collect()
    return best


def fit_stage(name,arrays,ids,output,checkpoint,initial,kind,lr,batch,epochs,end,
              baseline_dice,offset=0,short=False,fallback=None,baseline_epoch=0):
    output.mkdir(parents=True,exist_ok=True);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();model=compile_model(name,initial,kind,lr)
    timer=Timing(end-30)
    callbacks=[timer,tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(output/'history.csv')),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_hard_mask_dice',mode='max',
            save_weights_only=True,save_best_only=True,initial_value_threshold=baseline_dice)]
    if not short:
        callbacks += [tf.keras.callbacks.EarlyStopping(monitor='val_hard_mask_dice',mode='max',
            patience=12,start_from_epoch=10),tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_hard_mask_dice',mode='max',factor=.5,patience=8,min_lr=5e-5)]
    fit_start=time.monotonic()
    history=model.fit(dataset(arrays['train'],batch,True),validation_data=dataset(arrays['validation'],batch),
        epochs=epochs,callbacks=callbacks,verbose=2)
    duration=time.monotonic()-fit_start
    save(output/'history.json',history.history)
    assert all(np.isfinite(v).all() for v in history.history.values())
    latest=checkpoint.with_name(checkpoint.stem+'.latest.weights.h5')
    model.save_weights(latest)
    better=checkpoint.exists()
    selected=checkpoint if better else (fallback or initial)
    model.load_weights(selected)
    p=predictions(model,arrays['validation'][0])
    val=evaluate_probabilities(arrays['validation'][1],p,ids['validation'],.5)
    best_epoch=(int(np.argmax(history.history['val_hard_mask_dice']))+1+offset) if better else baseline_epoch
    result=dict(architecture=name,loss=kind,learning_rate=lr,batch_size=batch,
        epochs_completed=sum(timer.full),partial_epochs=sum(not v for v in timer.full),
        status='time_limited' if timer.hit else 'complete',best_epoch=best_epoch,
        selected_checkpoint=str(selected),latest_checkpoint=str(latest),validation=val,
        fit_seconds=duration,training_seconds=duration-sum(timer.validation),validation_seconds=sum(timer.validation),
        epoch_seconds=timer.epochs,total_seconds=time.monotonic()-started,offset=offset,
        history_path=str(output/'history.json'))
    save(output/'stage.json',result)
    del model;tf.keras.backend.clear_session();gc.collect()
    return result
