"""Validate completed Attention records and saved masks without TensorFlow/inference."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from kssd.final_audit import read,sha256,verify_saved_fold
from kssd.reporting import aggregate,compare_protocols,close
from kssd.figures import select_cases
from kssd.data import read_grayscale
from kssd.metrics import binary_metrics
from kssd.validate_data import digest


def validate(records,manifest,splits,baseline,checkpoints=None,data=None,figures=None):
    records=Path(records);summary=aggregate(records,manifest,splits,verify_checkpoints=False)
    if summary!=read(records/'summary.json'):raise ValueError('Attention summary mismatch')
    compare_protocols([baseline,summary])
    if summary['architecture']!='attention_unet':raise ValueError('Expected Attention U-Net records')
    folds=[];rows=[];mapping={};mask_total=0;samples={s['id']:s for s in manifest['samples']}
    for fold in splits['folds']:
        folder=records/f"fold-{fold['fold']}"
        checkpoint=Path(checkpoints)/f"fold-{fold['fold']}"/'best.weights.h5' if checkpoints else None
        verified=verify_saved_fold(folder,fold,manifest,splits,checkpoint)
        result=read(folder/'evaluation.json');hashes=read(folder/'verification.json')['saved_prediction_sha256']
        expected={f"{Path(r['id']).stem}.png" for r in result['per_image']}
        if set(hashes)!=expected or {p.name for p in (folder/'predictions').glob('*.png')}!=expected:
            raise ValueError('Prediction mask coverage mismatch')
        for row in result['per_image']:
            path=folder/'predictions'/f"{Path(row['id']).stem}.png"
            if sha256(path)!=hashes[path.name]:raise ValueError('Prediction mask checksum mismatch')
            with Image.open(path) as image:prediction=np.asarray(image)
            if prediction.shape!=(512,512) or prediction.dtype!=np.uint8 or not set(np.unique(prediction)).issubset({0,255}):
                raise ValueError('Invalid saved binary mask')
            if np.count_nonzero(prediction)!=row['tp']+row['fp']:raise ValueError('Predicted foreground count mismatch')
            if data:
                raw=read_grayscale(Path(data)/'label'/row['id'])
                if digest(raw)!=samples[row['id']]['mask_sha256']:raise ValueError('Ground truth source pixels changed')
                truth=(raw>127)[...,None]
                for key,value in binary_metrics(truth,(prediction/255.)[...,None],.5).items():close(row[key],value)
            mask_total+=1
        rows.extend(result['per_image']);mapping.update({r['id']:fold['fold'] for r in result['per_image']});folds.append(verified)
    figure_hashes={}
    if figures:
        figures=Path(figures);index=read(figures/'index.json')
        if index['status']!='complete' or len(index['cases'])!=6:raise ValueError('Incomplete Attention figures')
        for s in [baseline,summary]:
            if index['comparison_values'][s['architecture']]!={k:s[k] for k in ['fold_mean','fold_sample_std']}:
                raise ValueError('Figure comparison values mismatch')
            if index['checkpoint_hashes'][s['architecture']]!=[f['checkpoint_sha256'] for f in s['fold_metrics']]:
                raise ValueError('Figure checkpoint identities mismatch')
        chosen=select_cases(rows)
        if {c['criterion'] for c in index['cases']}!=set(chosen):raise ValueError('Missing figure criteria')
        for case in index['cases']:
            row=chosen[case['criterion']]
            if case['id']!=row['id'] or case['fold']!=mapping[row['id']] or any(case.get(k)!=v for k,v in row.items()):
                raise ValueError('Figure case mismatch')
            if not (figures/case['file']).is_file():raise ValueError('Missing case figure')
        for name in ['comparison.png','attention_unet-curves.png']:
            if not (figures/name).is_file():raise ValueError('Missing comparison/curve figure')
        figure_hashes={p.name:sha256(p) for p in figures.glob('*') if p.is_file()}
    return {'status':'verified','scope':'Saved records and masks; no training or repeated outer inference',
            'checkpoint_verification':'all five verified' if checkpoints else 'not performed; local weights required',
            'ground_truth_mask_verification':'all saved masks recomputed against decoded ground truth' if data else 'not performed; raw data required (saved hashes and foreground totals checked)',
            'saved_binary_masks':mask_total,'folds':folds,'summary':summary,'figure_files_sha256':figure_hashes,
            'mean_difference_attention_minus_modified':{k:summary['fold_mean'][k]-baseline['fold_mean'][k] for k in baseline['fold_mean']}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',default='reports/experiments/attention10fp32_20261012/attention_unet')
    parser.add_argument('--baseline',default='reports/experiments/baseline10fp32_final/modified_unet/summary.json')
    parser.add_argument('--checkpoints');parser.add_argument('--data');parser.add_argument('--output',required=True)
    parser.add_argument('--figures',default='reports/experiments/attention10fp32_20261012/figures')
    args=parser.parse_args();result=validate(args.records,read('reports/dataset/manifest.json'),read('reports/splits.json'),read(args.baseline),args.checkpoints,args.data,args.figures)
    import json
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'masks':result['saved_binary_masks'],'checkpoint_verification':result['checkpoint_verification'],'ground_truth_mask_verification':result['ground_truth_mask_verification']}))


if __name__=='__main__':main()
