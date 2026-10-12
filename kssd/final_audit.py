"""Audit saved results without training; optional local checkpoint verification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from kssd.reporting import aggregate,compare_protocols,verify_evaluation
from kssd.splits import validate_splits
from kssd.figures import select_cases

ARCHITECTURES=['modified_unet','unetpp','unet3plus','transunet']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def invalid(value):raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def verify_saved_fold(folder,fold,manifest,splits,checkpoint=None):
    folder=Path(folder);result=read(folder/'evaluation.json')
    verify_evaluation(result,fold,manifest)
    config=result['configuration'];history=read(folder/'history.json');provenance=read(folder/'provenance.json')
    if result['epochs_completed']!=config['epochs'] or any(len(v)!=config['epochs'] or not np.isfinite(v).all() for v in history.values()):
        raise ValueError('Fixed-budget history mismatch')
    selected=int(np.argmax(history['val_hard_mask_dice']))+1
    if selected!=result['selected_epoch'] or provenance.get('selected_epoch')!=selected:
        raise ValueError('Selected checkpoint is not earliest inner-validation maximum')
    if provenance['status']!='complete' or provenance['config']!=config or provenance['fold']!=fold:
        raise ValueError('Provenance configuration or fold mismatch')
    if result['dataset_fingerprint']!=splits['dataset_fingerprint'] or provenance['dataset_fingerprint']!=splits['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    if result['fold']!=fold['fold'] or result['seed']!=fold['seed'] or config['mask_threshold']!=manifest['mask_threshold']:
        raise ValueError('Saved split, seed or mask protocol mismatch')
    actual_precision=provenance['environment'].get('precision_policy')
    if actual_precision!=config.get('precision','float32'):
        raise ValueError('Actual precision differs from configured precision')
    expected_optimizer='LossScaleOptimizer' if actual_precision=='mixed_float16' else {'adam':'Adam','sgd':'SGD'}[config['optimizer']]
    if provenance.get('optimizer_class')!=expected_optimizer:
        raise ValueError('Actual optimizer differs from configured optimizer')
    if not provenance.get('initial_weights_sha256') or provenance['environment'].get('deterministic_ops') is not True:
        raise ValueError('Missing deterministic initialization evidence')
    checksum=result['checkpoint_sha256']
    if len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum) or provenance.get('checkpoint_sha256')!=checksum:
        raise ValueError('Checkpoint identity mismatch')
    if checkpoint is not None and (not Path(checkpoint).is_file() or sha256(checkpoint)!=checksum):
        raise ValueError('Selected checkpoint missing or checksum mismatch')
    if config['precision']=='float32':
        expected_updates=config['epochs']*((len(fold['train'])+config['batch_size']-1)//config['batch_size'])
        if result.get('optimizer_updates')!=expected_updates:raise ValueError('Optimizer update budget mismatch')
    return {'fold':fold['fold'],'selected_epoch':selected,'checkpoint_sha256':checksum,
            'checkpoint_verified':checkpoint is not None,'train_count':len(fold['train']),
            'selection_count':len(fold['selection']),'outer_count':len(fold['test']),
            'counts':result['counts'],'micro':result['micro'],'macro':result['macro'],
            'training_seconds':result['training_seconds'],'optimizer_updates':result.get('optimizer_updates'),
            'parameters':result['parameters'],'record_sha256':{p.name:sha256(p) for p in folder.glob('*') if p.is_file()}}


def verify_figures(folder,summaries,records_root):
    folder=Path(folder);index=read(folder/'index.json')
    if index['status']!='complete':raise ValueError('Incomplete figure generation')
    cases=index['cases']
    for summary in summaries:
        architecture=summary['architecture']
        if index['checkpoint_hashes'][architecture]!=[f['checkpoint_sha256'] for f in summary['fold_metrics']]:
            raise ValueError('Figure checkpoint identities mismatch')
        expected={'fold_mean':summary['fold_mean'],'fold_sample_std':summary['fold_sample_std']}
        if index['comparison_values'][architecture]!=expected:raise ValueError('Figure comparison values mismatch')
        rows=[];mapping={}
        for fold in range(1,6):
            result=read(Path(records_root)/architecture/f'fold-{fold}'/'evaluation.json')
            rows.extend(result['per_image']);mapping.update({r['id']:(fold,r) for r in result['per_image']})
        chosen=select_cases(rows);local=[c for c in cases if c['architecture']==architecture]
        if len(local)!=len(chosen) or {c['criterion'] for c in local}!=set(chosen):raise ValueError('Missing figure criteria')
        for case in local:
            row=chosen[case['criterion']]
            if case['id']!=row['id'] or case['fold']!=mapping[row['id']][0]:raise ValueError('Figure case selection mismatch')
            if any(case.get(k)!=v for k,v in row.items()):raise ValueError('Figure case metrics mismatch')
            if not (folder/case['file']).is_file():raise ValueError('Missing prediction figure')
        if not (folder/f'{architecture}-curves.png').is_file():raise ValueError('Missing training curves')
    if len(cases)!=6*len(summaries) or not (folder/'comparison.png').is_file():raise ValueError('Missing comparison figures')
    return {p.name:sha256(p) for p in sorted(folder.glob('*')) if p.is_file()}


def audit(records_root,manifest,splits,checkpoint_root=None,figures=None):
    validate_splits(splits,manifest);summaries=[];experiments=[]
    for architecture in ARCHITECTURES:
        root=Path(records_root)/architecture
        summary=aggregate(root,manifest,splits,verify_checkpoints=False)
        if summary!=read(root/'summary.json'):raise ValueError('Committed summary differs from recomputed results')
        folds=[]
        for fold in splits['folds']:
            checkpoint=Path(checkpoint_root)/architecture/f"fold-{fold['fold']}"/'best.weights.h5' if checkpoint_root else None
            folds.append(verify_saved_fold(root/f"fold-{fold['fold']}",fold,manifest,splits,checkpoint))
        summaries.append(summary)
        experiments.append({'architecture':architecture,'configuration':summary['configuration'],
                            'records':str(root),'checkpoints':str(Path(checkpoint_root)/architecture) if checkpoint_root else None,
                            'folds':folds,'fold_mean':summary['fold_mean'],'fold_sample_std':summary['fold_sample_std'],
                            'pooled_counts':summary['pooled_counts'],'pooled_micro':summary['pooled_micro'],
                            'total_training_seconds':summary['total_training_seconds']})
    compare_protocols(summaries)
    return {'status':'verified','scope':'Saved baseline records; no training or new outer-test model selection',
            'dataset_fingerprint':splits['dataset_fingerprint'],'n_folds_per_model':5,'architectures':ARCHITECTURES,
            'outer_ids_per_architecture':sum(len(f['test']) for f in splits['folds']),
            'checkpoint_verification':'all twenty SHA-256 verified' if checkpoint_root else 'not performed; checkpoint files required',
            'experiments':experiments,'figure_files_sha256':verify_figures(figures,summaries,records_root) if figures else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',default='reports/experiments/baseline10fp32_final')
    parser.add_argument('--manifest',default='reports/dataset/manifest.json');parser.add_argument('--splits',default='reports/splits.json')
    parser.add_argument('--checkpoints',help='Root containing architecture/fold-N/best.weights.h5; never fetched automatically')
    parser.add_argument('--figures',default='reports/figures/baseline10fp32_final');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.records,read(args.manifest),read(args.splits),args.checkpoints,args.figures)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'outer_ids_per_architecture':result['outer_ids_per_architecture'],
                      'checkpoint_verification':result['checkpoint_verification'],'figure_files':len(result['figure_files_sha256'] or {})}))


if __name__=='__main__':main()
