"""Strict aggregation of complete outer-fold evaluations from one fixed protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from kssd.metrics import metrics_from_counts
from kssd.splits import validate_splits

METRICS=['hard_mask_dice','hard_mask_iou','hard_mask_precision','hard_mask_recall']


def close(actual,expected):
    if not np.isclose(actual,expected,rtol=0,atol=1e-12):raise ValueError(f'Metric does not match counts: {actual} != {expected}')


def verify_evaluation(result,fold,manifest):
    if result['status']!='complete':raise ValueError('Incomplete fold evaluation')
    rows=result['per_image'];expected=set(fold['test']);ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=expected or result['outer_test_count']!=len(expected):
        raise ValueError('Evaluation IDs differ from outer test split')
    samples={s['id']:s for s in manifest['samples']};counts={k:0 for k in ['tp','fp','fn','tn']}
    for row in rows:
        local={k:row[k] for k in counts}
        if any(not isinstance(v,int) or v<0 for v in local.values()):raise ValueError('Invalid confusion counts')
        if row['tp']+row['fn']!=samples[row['id']]['stone_area']:raise ValueError('Ground truth count mismatch')
        if sum(local.values())!=int(np.prod(samples[row['id']]['shape'])):raise ValueError('Pixel count mismatch')
        for key,value in metrics_from_counts(**local).items():close(row[key],value)
        for key in counts:counts[key]+=row[key]
    if result['counts']!=counts:raise ValueError('Fold counts differ from per-image counts')
    for key,value in metrics_from_counts(**counts).items():close(result['micro'][key],value)
    for key in METRICS:close(result['macro'][key],float(np.mean([row[key] for row in rows])))
    if not 1<=result['selected_epoch']<=result['epochs_completed']:raise ValueError('Invalid checkpoint epoch')
    return True


def aggregate(run_root,manifest,splits,verify_checkpoints=True):
    validate_splits(splits,manifest);root=Path(run_root);results=[];common=None
    for fold in splits['folds']:
        path=root/f"fold-{fold['fold']}"/'evaluation.json'
        if not path.is_file():raise ValueError(f'Missing completed fold: {path}')
        result=json.loads(path.read_text(encoding='utf-8'));verify_evaluation(result,fold,manifest)
        if result['fold']!=fold['fold'] or result['seed']!=fold['seed'] or result['dataset_fingerprint']!=splits['dataset_fingerprint']:
            raise ValueError('Fold provenance mismatch')
        if result['configuration']['mask_threshold']!=manifest['mask_threshold']:raise ValueError('Mask protocol mismatch')
        if common is None:common=result['configuration']
        elif common!=result['configuration']:raise ValueError('Configurations differ across folds')
        if verify_checkpoints:
            weights=path.parent/'best.weights.h5'
            if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest()!=result['checkpoint_sha256']:
                raise ValueError('Missing or changed selected checkpoint')
            history=json.loads((path.parent/'history.json').read_text())
            if result['selected_epoch']!=int(np.argmax(history['val_hard_mask_dice']))+1:
                raise ValueError('Checkpoint epoch is not earliest maximum inner Dice')
        results.append(result)
    all_counts={key:sum(r['counts'][key] for r in results) for key in ['tp','fp','fn','tn']}
    return {'status':'complete','architecture':common['architecture'],'configuration':common,
            'dataset_fingerprint':splits['dataset_fingerprint'],'n_folds':len(results),
            'fold_metrics':[{key:r[key] for key in ['fold','seed','selected_epoch','micro','macro','counts','parameters','training_seconds','checkpoint_sha256','epochs_completed']} for r in results],
            'fold_mean':{key:float(np.mean([r['micro'][key] for r in results])) for key in METRICS},
            'fold_sample_std':{key:float(np.std([r['micro'][key] for r in results],ddof=1)) for key in METRICS},
            'pooled_counts':all_counts,'pooled_micro':metrics_from_counts(**all_counts),
            'total_training_seconds':sum(r['training_seconds'] for r in results),
            'interpretation':'Measured outer content-group CV under recorded budget; no patient-independent or clinical claim'}


def compare_protocols(summaries):
    if not summaries:raise ValueError('No results to compare')
    expected={k:v for k,v in summaries[0]['configuration'].items() if k!='architecture'}
    fingerprint=summaries[0]['dataset_fingerprint']
    for summary in summaries:
        actual={k:v for k,v in summary['configuration'].items() if k!='architecture'}
        if actual!=expected or summary['dataset_fingerprint']!=fingerprint:
            raise ValueError('Architecture comparison has mismatched protocols')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',default='reports/dataset/manifest.json');parser.add_argument('--splits',default='reports/splits.json')
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text());splits=json.loads(Path(args.splits).read_text())
    result=aggregate(args.run,manifest,splits);Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['fold_mean'],indent=2))


if __name__=='__main__':main()
