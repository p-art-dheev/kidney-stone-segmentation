"""Run nested group CV: inner selection checkpoint, one outer-fold evaluation."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from kssd.config import ExperimentConfig
from kssd.data import load_pair,read_grayscale
from kssd.validate_data import digest
from kssd.metrics import binary_metrics,metrics_from_counts
from kssd.models import build_model
from kssd.reproducibility import environment,set_seeds
from kssd.splits import validate_splits
from kssd.tf_metrics import dice_loss,make_metrics


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def dataset(root, ids, config, training=False, seed=42):
    root=Path(root)
    def read(name):
        name=name.decode('utf-8')
        x,y=load_pair(root/'image'/name,root/'label'/name,config.mask_threshold)
        if x.shape!=(config.input_size,config.input_size,1):raise ValueError('Input shape mismatch; resizing is disabled')
        return x,y
    def load(name):
        x,y=tf.numpy_function(read,[name],[tf.float32,tf.float32])
        x.set_shape((config.input_size,config.input_size,1));y.set_shape(x.shape)
        return x,y
    ds=tf.data.Dataset.from_tensor_slices(ids)
    if training:ds=ds.shuffle(len(ids),seed=seed,reshuffle_each_iteration=True)
    ds=ds.map(load,num_parallel_calls=2,deterministic=True)
    if config.augmentation:
        if training:
            from kssd.augmentation import augment_dataset,AugmentationParameters
            ds=augment_dataset(ds,seed,AugmentationParameters(**config.augmentation_parameters))
    options=tf.data.Options();options.experimental_deterministic=True;options.threading.private_threadpool_size=2
    return ds.batch(config.batch_size,drop_remainder=False).with_options(options).prefetch(1)


def evaluate(model,root,ids,config):
    rows=[];counts=dict(tp=0,fp=0,fn=0,tn=0);durations=[]
    for names in [ids[i:i+config.batch_size] for i in range(0,len(ids),config.batch_size)]:
        pairs=[load_pair(Path(root)/'image'/n,Path(root)/'label'/n,config.mask_threshold) for n in names]
        images=np.stack([p[0] for p in pairs])
        start=time.perf_counter();predictions=model(images,training=False).numpy()
        durations.append(time.perf_counter()-start)
        for name,pair,pred in zip(names,pairs,predictions):
            result=binary_metrics(pair[1],pred,config.threshold)
            rows.append({'id':name,'stone_area':int(pair[1].sum()),**result})
            for key in counts:counts[key]+=result[key]
    metrics=metrics_from_counts(**counts)
    macro={key:float(np.mean([r[key] for r in rows])) for key in metrics}
    # First batch includes warm-up and is excluded from latency, never from metrics.
    timed_images=sum(len(ids[i:i+config.batch_size]) for i in range(config.batch_size,len(ids),config.batch_size))
    latency=sum(durations[1:])/timed_images if timed_images else None
    return {'counts':counts,'micro':metrics,'macro':macro,'per_image':rows,
            'inference_seconds_per_image_excluding_first_batch':latency,
            'latency_scope':'model forward + device synchronization; excludes disk/preprocessing'}


def run_fold(root,manifest,splits,config,fold_number,output,evaluate_outer=True):
    validate_splits(splits,manifest)
    if config.n_folds!=splits['n_folds'] or config.seed!=splits['seed'] or config.mask_threshold!=manifest['mask_threshold']:
        raise ValueError('Configuration differs from audited preprocessing/split protocol')
    for sample in manifest['samples']:
        image=read_grayscale(Path(root)/'image'/sample['id'])
        mask=read_grayscale(Path(root)/'label'/sample['id'])
        if digest(image)!=sample['image_sha256'] or digest(mask)!=sample['mask_sha256']:
            raise ValueError(f"Source pixels changed since audit: {sample['id']}")
    fold=next(f for f in splits['folds'] if f['fold']==fold_number)
    output=Path(output)/f'fold-{fold_number}'
    if output.exists():raise ValueError(f'Output already exists: {output}; choose a new run directory')
    output.mkdir(parents=True)
    tf.keras.mixed_precision.set_global_policy(config.precision)
    tf.keras.backend.clear_session();gc.collect();set_seeds(fold['seed'])
    model=build_model(config.architecture,(config.input_size,config.input_size,1))
    optimizer=(tf.keras.optimizers.Adam if config.optimizer=='adam' else tf.keras.optimizers.SGD)(learning_rate=config.learning_rate)
    if config.loss=='dice':loss=dice_loss
    else:
        from kssd.losses import FocalTverskyLoss
        loss=FocalTverskyLoss()
    model.compile(optimizer=optimizer,loss=loss,metrics=make_metrics(config.threshold),jit_compile=False)
    if int(optimizer.iterations)!=0:raise RuntimeError('Optimizer was not freshly initialized')
    initial_hash=hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    write_json(output/'provenance.json',{'config':config.to_dict(),'fold':fold,'dataset_fingerprint':splits['dataset_fingerprint'],
                                      'environment':environment(),'parameters':model.count_params(),'initial_weights_sha256':initial_hash,
                                      'optimizer_class':type(model.optimizer).__name__,
                                      'checkpoint_policy':'max inner selection hard_mask_dice; earliest epoch wins ties',
                                      'status':'running'})
    checkpoint=output/'best.weights.h5'
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_hard_mask_dice',mode='max',save_best_only=True,save_weights_only=True),
               tf.keras.callbacks.CSVLogger(str(output/'history.csv')),tf.keras.callbacks.TerminateOnNaN()]
    if config.early_stopping_patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_hard_mask_dice',mode='max',patience=config.early_stopping_patience))
    start=time.perf_counter()
    history=model.fit(dataset(root,fold['train'],config,True,fold['seed']),
                      validation_data=dataset(root,fold['selection'],config),epochs=config.epochs,callbacks=callbacks,verbose=2)
    duration=time.perf_counter()-start
    optimizer_updates=int(model.optimizer.iterations.numpy())
    write_json(output/'history.json',history.history)
    if not checkpoint.exists() or not all(np.isfinite(v).all() for v in history.history.values()):
        raise RuntimeError('Training did not produce finite histories and a checkpoint')
    model.load_weights(checkpoint)
    selected_epoch=int(np.argmax(history.history['val_hard_mask_dice']))+1
    if not evaluate_outer:
        result={'status':'selection_only','selected_epoch':selected_epoch,'training_seconds':duration,
                'epochs_completed':len(history.epoch),'configuration':config.to_dict()}
        result['optimizer_updates']=optimizer_updates
        write_json(output/'selection_run.json',result)
        provenance=json.loads((output/'provenance.json').read_text(encoding='utf-8'))
        provenance.update({'status':'selection_only','selected_epoch':selected_epoch})
        write_json(output/'provenance.json',provenance)
        del model;tf.keras.backend.clear_session();gc.collect()
        return result
    result=evaluate(model,root,fold['test'],config)
    result.update({'architecture':config.architecture,'fold':fold_number,'seed':fold['seed'],
                   'selected_epoch':selected_epoch,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'parameters':model.count_params(),'training_seconds':duration,'epochs_completed':len(history.epoch),
                   'outer_test_count':len(fold['test']),'inner_selection_count':len(fold['selection']),
                   'configuration':config.to_dict(),'dataset_fingerprint':splits['dataset_fingerprint'],'status':'complete'})
    result['optimizer_updates']=optimizer_updates
    write_json(output/'evaluation.json',result)
    provenance=json.loads((output/'provenance.json').read_text(encoding='utf-8'))
    provenance.update({'status':'complete','selected_epoch':selected_epoch,'checkpoint_sha256':result['checkpoint_sha256']})
    write_json(output/'provenance.json',provenance)
    del model;tf.keras.backend.clear_session();gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',default='reports/dataset/manifest.json');parser.add_argument('--splits',default='reports/splits.json')
    parser.add_argument('--fold',type=int)
    parser.add_argument('--selection-only',action='store_true',help='Runtime pilot without looking at outer-test predictions')
    parser.add_argument('--require-gpu',action='store_true',help='Fail before creating a run if TensorFlow cannot see a GPU')
    args=parser.parse_args()
    gpus=tf.config.list_physical_devices('GPU')
    if args.require_gpu and not gpus:raise RuntimeError('GPU required but not visible; training was not started')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    config=ExperimentConfig(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'));splits=json.loads(Path(args.splits).read_text(encoding='utf-8'))
    folds=[args.fold] if args.fold else list(range(1,config.n_folds+1))
    for fold in folds:
        result=run_fold(args.data,manifest,splits,config,fold,args.output,evaluate_outer=not args.selection_only)
        print(json.dumps({'fold':fold,'micro':result.get('micro'),'training_seconds':result['training_seconds']}))


if __name__=='__main__':main()
