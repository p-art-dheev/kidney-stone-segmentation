"""Read frozen caches; diagnostics and threshold selection never access test data."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from kssd.data import read_grayscale
from kssd.models import build_model
from kssd.metrics import binary_metrics, metrics_from_counts
from kssd.reproducibility import set_seeds
from strain.data import save
from strain.train import loss
from strain.config import SEED, BATCH

BASE = Path('strain/results/experiment256_final_20261013')
MODELS = ['unetpp','transunet']
THRESHOLDS = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995,.999]


def load_cached(parts=('train','validation'), allow_test=False):
    if 'test' in parts and not allow_test:
        raise ValueError('Test access is forbidden during diagnosis/tuning')
    split = json.loads((BASE/'splits.json').read_text())['splits']
    audit = json.loads((BASE/'preprocessing.json').read_text())
    arrays = {}
    for part in parts:
        pairs = [np.load(Path(audit['cache'])/(Path(n).stem+'.npz')) for n in split[part]]
        x = np.stack([p['image'] for p in pairs])[...,None].astype('float32')/255
        y = np.stack([p['mask'] for p in pairs])[...,None].astype('float32')
        assert x.shape == y.shape and x.shape[1:] == (256,256,1)
        assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
        assert np.isin(y,[0,1]).all()
        arrays[part] = x,y
    return arrays, split


def model_from_checkpoint(name, checkpoint):
    if name not in MODELS:
        raise ValueError('Only the two targeted architectures are permitted')
    tf.keras.backend.clear_session()
    gc.collect()
    set_seeds(SEED)
    tf.keras.mixed_precision.set_global_policy('float32')
    model = build_model(name, (256,256,1))
    model.load_weights(checkpoint)
    return model


def predictions(model, x):
    p = np.concatenate([model(x[i:i+BATCH],training=False).numpy() for i in range(0,len(x),BATCH)])
    assert p.shape == x.shape and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    return p


def evaluate_probabilities(y, p, ids, threshold):
    rows, total = [], dict(tp=0,fp=0,fn=0,tn=0)
    for name, truth, prob in zip(ids,y,p):
        row = dict(id=name, **binary_metrics(truth,prob,threshold))
        rows.append(row)
        for k in total:
            total[k] += row[k]
    return dict(threshold=threshold, counts=total, micro=metrics_from_counts(**total),
        per_image=rows, truth_foreground_percent=float(y.mean()*100),
        predicted_foreground_percent=float((p>=threshold).mean()*100),
        false_positive_rate=total['fp']/(total['fp']+total['tn']) if total['fp']+total['tn'] else 0)


def sweep(y,p,ids):
    rows = [evaluate_probabilities(y,p,ids,t) for t in THRESHOLDS]
    # Highest pooled validation Dice, then closest to common 0.5, then lower threshold.
    best = max(rows,key=lambda r:(r['micro']['hard_mask_dice'],-abs(r['threshold']-.5),-r['threshold']))
    return rows,best


def distribution(p,y):
    quantiles = [0,.01,.1,.5,.9,.95,.99,.999,1]
    result = {}
    for name, values in [('all',p.ravel()),('true_foreground',p[y==1]),('true_background',p[y==0])]:
        result[name] = dict(mean=float(values.mean()), quantiles=dict(zip(map(str,quantiles),
            np.quantile(values,quantiles).tolist()))) if values.size else None
    result['histogram'] = dict(counts=np.histogram(p,bins=np.linspace(0,1,21))[0].tolist(),
        bins=np.linspace(0,1,21).tolist())
    return result


def examples(model_name, root, ids, x,y,p,threshold, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = list(range(min(3,len(ids))))
    fig, axes = plt.subplots(len(selected),5,figsize=(17,4*len(selected)))
    for row,index in enumerate(selected):
        image,truth,prob = x[index,...,0],y[index,...,0],p[index,...,0]
        native = read_grayscale(Path(root)/'image'/ids[index])
        pred = prob >= threshold
        pictures = [native,truth,prob,pred,image]
        titles = ['Original CT','Resized ground truth','Probability','Prediction','GT green / prediction red']
        for col,(picture,title) in enumerate(zip(pictures,titles)):
            ax = axes[row,col]
            ax.imshow(picture,cmap='inferno' if col==2 else 'gray',
                vmin=0,vmax=255 if col==0 else 1)
            if col==4:
                rgba=np.zeros((*truth.shape,4));rgba[...,1]=truth;rgba[...,0]=pred
                rgba[...,3]=np.maximum(truth,pred)*.7
                ax.imshow(rgba)
            ax.set_title(title+'\n'+ids[index]);ax.axis('off')
    fig.suptitle(model_name+f' | validation | threshold {threshold:g}')
    fig.tight_layout();fig.savefig(output/'examples.png',dpi=150);plt.close(fig)


def diagnose(name, arrays, split, root, output):
    output.mkdir(parents=True,exist_ok=True)
    old = json.loads((BASE/name/'result.json').read_text())
    # Only configuration/checkpoint/history fields are used, not stored test metrics.
    checkpoint = Path(old['checkpoint'])
    model = model_from_checkpoint(name, checkpoint)
    result = dict(architecture=name, checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        parameters=model.count_params(), input_shape=list(model.input_shape), output_shape=list(model.output_shape),
        output_head=[dict(name=l.name,type=type(l).__name__,config=l.get_config()) for l in model.layers[-3:]],
        transformer_layers=[dict(name=l.name,type=type(l).__name__,config=l.get_config())
            for l in model.layers if 'patch' in l.name or 'ViT' in l.name or 'token' in l.name],
        layers=[dict(name=l.name,type=type(l).__name__) for l in model.layers],
        no_pretrained_backbone=True, no_deep_supervision=True)
    for part in ['train','validation']:
        x,y = arrays[part]
        p = predictions(model,x)
        result[part] = dict(common=evaluate_probabilities(y,p,split[part],.5),
            probabilities=distribution(p,y), loss=float(np.mean(loss(y,p).numpy())))
        if part == 'validation':
            thresholds,best = sweep(y,p,split[part])
            result[part].update(threshold_sweep=thresholds,best_calibrated=best)
            examples(name,root,split[part],x,y,p,best['threshold'],output)
            np.savez_compressed(output/'probabilities.npz', probabilities=p)
    history = json.loads((BASE/name/'history.json').read_text())
    result['previous_history_summary'] = dict(initial_lr=history['learning_rate'][0],
        final_lr=history['learning_rate'][-1], first_floor_epoch=next((i+1 for i,v in enumerate(history['learning_rate'])
        if v <= 1.01e-5),None), training_loss_initial=history['loss'][0], training_loss_final=history['loss'][-1],
        validation_loss_initial=history['val_loss'][0],validation_loss_final=history['val_loss'][-1])
    # Independent flattened count check agrees with the per-image accumulator.
    common = result['validation']['common']
    flat = binary_metrics(arrays['validation'][1],p,.5)
    assert all(flat[k] == common['counts'][k] for k in ['tp','fp','fn','tn'])
    result['independent_metric_check_passed']=True
    save(output/'diagnosis.json',result)
    print(json.dumps(dict(architecture=name, validation_common=common['micro'],
        validation_calibrated=result['validation']['best_calibrated']['micro'],
        selected_threshold=result['validation']['best_calibrated']['threshold'],
        truth_percent=common['truth_foreground_percent'], prediction_percent=common['predicted_foreground_percent'])),flush=True)
    del model
    return result
