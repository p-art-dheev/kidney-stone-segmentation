"""GPU Attention U-Net CV: fold one is retained as pilot, then folds 2–5.

No performance-based pilot gate, repeated fold training, or second outer inference.
"""
import argparse
import hashlib
import shutil
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from kssd.config import ExperimentConfig
from kssd.train import run_fold,write_json
from kssd.final_audit import read,verify_saved_fold
from kssd.reporting import aggregate,compare_protocols
from kssd.data import load_pair
from kssd.figures import select_cases,prediction_figure,training_plot,comparison_plot,verify_prediction


def verify_masks(folder,data,result,threshold):
    folder=Path(folder);expected={f"{Path(row['id']).stem}.png" for row in result['per_image']}
    if {p.name for p in folder.glob('*.png')}!=expected:raise ValueError('Saved mask coverage mismatch')
    for row in result['per_image']:
        _,truth=load_pair(Path(data)/'image'/row['id'],Path(data)/'label'/row['id'])
        prediction=np.asarray(Image.open(folder/f"{Path(row['id']).stem}.png"))
        if prediction.shape!=truth.shape[:2] or not set(np.unique(prediction)).issubset({0,255}):raise ValueError('Invalid binary prediction mask')
        verify_prediction(truth,(prediction/255.)[...,None],row,threshold)
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.png'))}


def finish(run,data,manifest,splits,baseline,output):
    summary=aggregate(run,manifest,splits);base=aggregate(baseline,manifest,splits)
    compare_protocols([base,summary]);write_json(run/'summary.json',summary)
    differences={key:float(summary['fold_mean'][key]-base['fold_mean'][key]) for key in base['fold_mean']}
    comparison={'status':'complete','scope':'Matched 10-epoch grouped CV; descriptive comparison, no significance or clinical claim',
                'modified_unet':base,'attention_unet':summary,'mean_difference_attention_minus_modified':differences,
                'provenance_note':'Architecture/gate and binary-mask persistence were added after baseline; original baseline records/weights are reused, not retrained. Computational source hashes differ and are retained.'}
    write_json(output/'comparison.json',comparison)
    figures=output/'figures';figures.mkdir(exist_ok=False)
    comparison_plot([base,summary],figures/'comparison.png');training_plot(run,figures/'attention_unet-curves.png')
    rows=[];mapping={}
    for fold in splits['folds']:
        local=read(run/f"fold-{fold['fold']}"/'evaluation.json')['per_image']
        rows.extend(local);mapping.update({r['id']:fold['fold'] for r in local})
    cases=[]
    for criterion,row in select_cases(rows).items():
        fold=mapping[row['id']];image,truth=load_pair(Path(data)/'image'/row['id'],Path(data)/'label'/row['id'])
        pred=(np.asarray(Image.open(run/f'fold-{fold}'/'predictions'/f"{Path(row['id']).stem}.png"))/255.)[...,None]
        name=f"attention_unet-{criterion}-{Path(row['id']).stem}.png"
        prediction_figure(image,truth,pred,row,.5,figures/name,f'Attention U-Net · fold {fold} · {criterion}')
        cases.append({'architecture':'attention_unet','fold':fold,'criterion':criterion,'file':name,**row})
    write_json(figures/'index.json',{'status':'complete','prediction_source':'Binary masks saved during the original single outer evaluation; no repeated outer forward pass',
                                   'cases':cases,'comparison_values':{s['architecture']:{'fold_mean':s['fold_mean'],'fold_sample_std':s['fold_sample_std']} for s in [base,summary]},
                                   'checkpoint_hashes':{s['architecture']:[f['checkpoint_sha256'] for f in s['fold_metrics']] for s in [base,summary]}})
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--config',default='configs/10epoch/attention_unet.json')
    parser.add_argument('--baseline',required=True,help='Existing completed Modified U-Net run, including weights')
    parser.add_argument('--export',help='Optional compact evidence export directory; never exports H5 weights')
    args=parser.parse_args();gpus=tf.config.list_physical_devices('GPU')
    if not gpus:raise RuntimeError('GPU required; experiment not started')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    config=ExperimentConfig(**read(args.config));manifest=read('reports/dataset/manifest.json');splits=read('reports/splits.json')
    baseline=aggregate(args.baseline,manifest,splits)
    if config.architecture!='attention_unet':raise ValueError('Attention experiment requires attention_unet')
    if {k:v for k,v in config.to_dict().items() if k!='architecture'}!={k:v for k,v in baseline['configuration'].items() if k!='architecture'}:
        raise ValueError('Configuration differs from verified Modified U-Net protocol')
    output=Path(args.output)
    if output.exists():raise ValueError('Output exists; preserve it and choose a new root')
    output.mkdir(parents=True);run=output/'attention_unet';completed=[];started=time.time()
    status={'status':'running','architecture':'attention_unet','pilot_fold':1,'pilot_reused_in_cv':True,'outer_evaluations_per_fold':1,'completed_folds':completed,'configuration':config.to_dict()}
    def save_status():write_json(output/'experiment_status.json',status)
    save_status()
    try:
        for fold in splits['folds']:
            number=fold['fold'];status['active_fold']=number;save_status()
            print(f'ATTENTION_FOLD_START {number}',flush=True)
            tf.config.experimental.reset_memory_stats('GPU:0')
            result=run_fold(args.data,manifest,splits,config,number,run,save_predictions=True)
            record=verify_saved_fold(run/f'fold-{number}',fold,manifest,splits,run/f'fold-{number}'/'best.weights.h5')
            hashes=verify_masks(run/f'fold-{number}'/'predictions',args.data,result,config.threshold)
            record['gpu_peak_allocated_bytes']=tf.config.experimental.get_memory_info('GPU:0')['peak']
            record['saved_prediction_sha256']=hashes
            write_json(run/f'fold-{number}'/'verification.json',record);completed.append(number)
            status['pilot_passed']=True;status['last_fold_micro']=result['micro'];save_status()
            if args.export:
                destination=Path(args.export)/'attention_unet'/f'fold-{number}';destination.mkdir(parents=True,exist_ok=True)
                for name in ['evaluation.json','history.json','history.csv','provenance.json','verification.json']:
                    shutil.copy2(run/f'fold-{number}'/name,destination/name)
                shutil.copytree(run/f'fold-{number}'/'predictions',destination/'predictions',dirs_exist_ok=True)
            print(f'ATTENTION_FOLD_VERIFIED {number} {result["micro"]} seconds={result["training_seconds"]:.2f}',flush=True)
        finish(run,args.data,manifest,splits,Path(args.baseline),output)
        status.update({'status':'complete','active_fold':None,'wall_seconds':time.time()-started});save_status()
        if args.export:
            export=Path(args.export);shutil.copy2(run/'summary.json',export/'attention_unet/summary.json')
            for name in ['comparison.json','experiment_status.json']:shutil.copy2(output/name,export/name)
            shutil.copytree(output/'figures',export/'figures',dirs_exist_ok=True)
        print('ATTENTION_FIVE_FOLD_COMPLETE',flush=True)
    except BaseException as exc:
        status.update({'status':'failed','error':f'{type(exc).__name__}: {exc}','wall_seconds':time.time()-started});save_status()
        raise


if __name__=='__main__':main()
