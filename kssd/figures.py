"""Generate figures only from complete, matched, checkpoint-verified evaluations."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from kssd.data import load_pair,read_grayscale
from kssd.validate_data import digest
from kssd.metrics import binary_metrics
from kssd.reporting import METRICS,aggregate,close,compare_protocols


def select_cases(rows):
    if not rows:raise ValueError('No evaluated images')
    criteria={'highest_dice':lambda r:(-r['hard_mask_dice'],r['id']),
              'lowest_dice':lambda r:(r['hard_mask_dice'],r['id']),
              'largest_fn':lambda r:(-r['fn'],r['id']),
              'largest_fp':lambda r:(-r['fp'],r['id']),
              'smallest_area':lambda r:(r['stone_area'],r['id'])}
    result={key:sorted(rows,key=rank)[0] for key,rank in criteria.items()}
    result['middle_dice']=sorted(rows,key=lambda r:(r['hard_mask_dice'],r['id']))[len(rows)//2]
    return result


def verify_prediction(truth,prob,row,threshold):
    actual=binary_metrics(truth,prob,threshold)
    for key,value in actual.items():
        if key in ['tp','fp','fn','tn']:
            if value!=row[key]:raise ValueError('Restored prediction counts differ from evaluation')
        else:close(value,row[key])
    return actual


def prediction_figure(image,truth,prob,row,threshold,path,title):
    import matplotlib.pyplot as plt
    from scipy.ndimage import label
    verify_prediction(truth,prob,row,threshold)
    x=image[...,0];y=truth[...,0].astype(bool);p=prob[...,0]>=threshold
    colors=np.zeros((*y.shape,4),dtype=np.float32)
    colors[y&p]=[1,1,0,.8];colors[y&~p]=[1,0,0,.8];colors[~y&p]=[0,1,1,.8]
    components,n=label(y)
    if n:
        sizes=np.bincount(components.ravel());sizes[0]=0
        center=np.argwhere(components==sizes.argmax()).mean(axis=0).astype(int)
    else:center=np.array(y.shape)//2
    lo=np.maximum(center-64,0);hi=np.minimum(lo+128,y.shape);lo=np.maximum(hi-128,0)
    box=(slice(lo[0],hi[0]),slice(lo[1],hi[1]))
    fig,axes=plt.subplots(1,5,figsize=(17,4))
    for ax,values,caption in zip(axes[:3],[x,y,p],['CT image','Ground truth','Binary prediction']):
        ax.imshow(values,cmap='gray',vmin=0,vmax=1);ax.set_title(caption)
    axes[3].imshow(x,cmap='gray',vmin=0,vmax=1);axes[3].imshow(colors);axes[3].set_title('Yellow TP · red FN · cyan FP')
    axes[4].imshow(x[box],cmap='gray',vmin=0,vmax=1);axes[4].imshow(colors[box]);axes[4].set_title('Largest GT component: 128px crop')
    for ax in axes:ax.axis('off')
    fig.suptitle(f"{title} | {row['id']} | Dice {row['hard_mask_dice']:.4f} | threshold >= {threshold} | TP/FP/FN {row['tp']}/{row['fp']}/{row['fn']}",fontsize=13)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def comparison_plot(summaries,path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharey=True)
    lower=min(s['fold_mean'][m]-s['fold_sample_std'][m] for s in summaries for m in METRICS)
    upper=max(s['fold_mean'][m]+s['fold_sample_std'][m] for s in summaries for m in METRICS)
    for ax,metric in zip(axes,METRICS):
        means=[s['fold_mean'][metric] for s in summaries];sd=[s['fold_sample_std'][metric] for s in summaries]
        ax.bar(range(len(summaries)),means,yerr=sd,capsize=4,color=['#466a91','#65a188','#ba8248','#8b679f'])
        ax.set_xticks(range(len(summaries)),[s['architecture'] for s in summaries],rotation=30,ha='right',fontsize=9)
        ax.set_title(metric.replace('hard_mask_',''));ax.set_ylim(min(0,lower-.03),max(1,upper+.03));ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Outer-fold pooled-pixel score')
    fig.suptitle('Five-fold mean ± sample SD; threshold 0.5 (not confidence intervals)')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def training_plot(root,path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for fold in range(1,6):
        history=json.loads((Path(root)/f'fold-{fold}'/'history.json').read_text())
        epochs=np.arange(1,len(history['loss'])+1)
        for ax,metric in zip(axes,['loss','hard_mask_dice']):
            line=ax.plot(epochs,history[metric],label=f'F{fold} training')[0]
            ax.plot(epochs,history['val_'+metric],linestyle='--',color=line.get_color(),label=f'F{fold} inner selection')
            ax.set_xlabel('Epoch');ax.set_ylabel(metric.replace('_',' '));ax.grid(alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(ncol=2,fontsize=8)
    fig.suptitle(f'{Path(root).name}: training solid, inner selection dashed; outer test excluded')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def generate(root,data,manifest,splits,output,architectures=None):
    from kssd.models import ARCHITECTURES,build_model
    from kssd.reproducibility import set_seeds
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    architectures=ARCHITECTURES if architectures is None else architectures
    if not architectures or len(set(architectures))!=len(architectures) or any(a not in ARCHITECTURES for a in architectures):
        raise ValueError('Invalid architecture selection')
    summaries=[aggregate(Path(root)/architecture,manifest,splits) for architecture in architectures]
    compare_protocols(summaries)
    # Fixed comparison figure caption and all model evaluation thresholds must agree.
    if any(s['configuration']['threshold']!=.5 for s in summaries):raise ValueError('Figures currently require threshold 0.5')
    output=Path(output)
    if output.exists():raise ValueError('Figure output exists; choose a new directory')
    output.mkdir(parents=True)
    index={'status':'running','selection':'All deterministic criteria retained, including poor best cases and zero-error maxima',
           'checkpoint_hashes':{s['architecture']:[f['checkpoint_sha256'] for f in s['fold_metrics']] for s in summaries},
           'comparison_values':{s['architecture']:{'fold_mean':s['fold_mean'],'fold_sample_std':s['fold_sample_std']} for s in summaries},'cases':[]}
    def save_index(): (output/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    save_index();comparison_plot(summaries,output/'comparison.png')
    sources={sample['id']:sample for sample in manifest['samples']}
    for summary in summaries:
        architecture=summary['architecture'];run=Path(root)/architecture;training_plot(run,output/f'{architecture}-curves.png')
        rows=[];sample_folds={}
        for fold in splits['folds']:
            evaluation=json.loads((run/f"fold-{fold['fold']}"/'evaluation.json').read_text())
            rows.extend(evaluation['per_image'])
            sample_folds.update({row['id']:fold for row in evaluation['per_image']})
        chosen=select_cases(rows)
        for fold in splits['folds']:
            local={name:row for name,row in chosen.items() if sample_folds[row['id']]['fold']==fold['fold']}
            if not local:continue
            tf.keras.backend.clear_session();gc.collect();set_seeds(fold['seed'])
            model=build_model(architecture)
            model.load_weights(run/f"fold-{fold['fold']}"/'best.weights.h5')
            for criterion,row in local.items():
                # Preserve original evaluation batch shape/order, avoiding changed
                # floating-point kernels at the threshold when using batch size one.
                position=fold['test'].index(row['id']);batch=summary['configuration']['batch_size']
                start=(position//batch)*batch;names=fold['test'][start:start+batch];pairs=[]
                for sample_id in names:
                    image_path=Path(data)/'image'/sample_id;mask_path=Path(data)/'label'/sample_id
                    if digest(read_grayscale(image_path))!=sources[sample_id]['image_sha256'] or digest(read_grayscale(mask_path))!=sources[sample_id]['mask_sha256']:
                        raise ValueError('Figure source pixels changed since audit')
                    pairs.append(load_pair(image_path,mask_path,summary['configuration']['mask_threshold']))
                probabilities=model(np.stack([pair[0] for pair in pairs]),training=False).numpy()
                image,truth=pairs[position-start];probability=probabilities[position-start]
                name=f'{architecture}-{criterion}-{Path(row["id"]).stem}.png'
                prediction_figure(image,truth,probability,row,.5,output/name,f'{architecture} · fold {fold["fold"]} · {criterion}')
                index['cases'].append({'architecture':architecture,'fold':fold['fold'],'criterion':criterion,'file':name,**row});save_index()
            del model
    index['status']='complete';save_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--architectures',nargs='+',help='Defaults to all four; use modified_unet for the controlled loss run')
    parser.add_argument('--manifest',default='reports/dataset/manifest.json');parser.add_argument('--splits',default='reports/splits.json')
    args=parser.parse_args()
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
    generate(args.root,args.data,json.loads(Path(args.manifest).read_text()),json.loads(Path(args.splits).read_text()),args.output,args.architectures)


if __name__=='__main__':main()
