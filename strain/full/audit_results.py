"""Independent arithmetic, split, checkpoint and log checks; no model selection."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def audit(output):
    output=Path(output)
    read=lambda p:json.loads(Path(p).read_text())
    checks=[]
    def check(label,condition):
        if not condition:raise AssertionError(label)
        checks.append(label)
    split=read(output/'splits.json');ids=split['splits']
    check('Full partition counts',split['counts']==dict(train=453,validation=288,test=90))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        check(a+'/'+b+' IDs disjoint',not set(ids[a])&set(ids[b]))
        check(a+'/'+b+' content groups disjoint',not set(split['groups'][a])&set(split['groups'][b]))
    manifest=read('reports/dataset/manifest.json')
    check('All eligible samples retained',set(sum(ids.values(),[]))=={s['id'] for s in manifest['samples'] if s['eligible']})
    old=read('strain/results/experiment256_final_20261013/splits.json')
    check('Historical fitting IDs remain training only',set(old['splits']['train'])<=set(ids['train']))
    protection=read(output/'protected_files.json')
    for p,expected in protection.items():
        check('Unmodified '+p,hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected)
    lock=read(output/'selection.lock.json')
    check('Selections sealed before test',lock['validation_only'] and not lock['test_data_accessed'])
    comparison=read(output/'comparison.json')
    check('Actual overall time and deadline status',comparison['elapsed_seconds']>0 and comparison['under_60_minutes']==(comparison['elapsed_seconds']<3600))
    if 'first_finalization_attempt_seconds' in comparison:
        check('First supervisor runtime recorded',0<comparison['first_finalization_attempt_seconds']<3600)
    for r in comparison['models']:
        name=r['architecture'];folder=output/name
        check(name+' test result available',r.get('test_common') is not None)
        check(name+' checkpoint hash',hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest()==r['checkpoint_sha256'])
        check(name+' selection matches sealed checkpoint',r['checkpoint_sha256']==lock['selections'][name]['checkpoint_sha256'])
        for key,partition in [('validation_common','validation'),('validation_calibrated','validation'),('test_common','test'),('test_calibrated','test')]:
            metric=r[key];counts=metric['counts'];tp,fp,fn,tn=[counts[k] for k in ['tp','fp','fn','tn']]
            rows=metric['per_image']
            check(name+' '+key+' IDs',set(z['id'] for z in rows)==set(ids[partition]) and len(rows)==len(ids[partition]))
            check(name+' '+key+' pooled counts',all(sum(z[k] for z in rows)==counts[k] for k in counts))
            check(name+' '+key+' pixel total',tp+fp+fn+tn==len(ids[partition])*256*256)
            expected=dict(hard_mask_dice=2*tp/(2*tp+fp+fn),hard_mask_iou=tp/(tp+fp+fn),
                          hard_mask_precision=tp/(tp+fp) if tp+fp else 0,hard_mask_recall=tp/(tp+fn) if tp+fn else 0)
            check(name+' '+key+' independent arithmetic',all(math.isclose(metric['micro'][k],v,abs_tol=1e-10) for k,v in expected.items()))
            check(name+' '+key+' Dice/IoU consistency',math.isclose(metric['micro']['hard_mask_dice'],2*metric['micro']['hard_mask_iou']/(1+metric['micro']['hard_mask_iou']),abs_tol=1e-10))
        check(name+' common threshold',r['validation_common']['threshold']==r['test_common']['threshold']==.5)
        check(name+' calibration recall floor',r['validation_calibrated']['micro']['hard_mask_recall']>=r['calibrated_recall_floor'])
        history=read(folder/'history.json');length=len(history['loss'])
        check(name+' full finite epoch history',length==r['epochs_completed']+r['partial_epochs'] and all(len(v)==length and all(math.isfinite(x) for x in v) for v in history.values()))
        for k in ['loss','hard_mask_dice','hard_mask_iou','hard_mask_precision','hard_mask_recall']:
            check(name+' logged train/validation '+k,k in history and 'val_'+k in history)
        check(name+' CSV epoch history',len(list(csv.DictReader((folder/'history.csv').open())))==length)
        check(name+' curves exist',(folder/'training_curves.png').exists())
        check(name+' best epoch valid',0<=r['best_validation_epoch']<=length)
        trial=folder/'loss_comparison.json'
        if trial.exists():
            candidates=read(trial)
            winner=max(candidates['candidates'],key=lambda z:z['validation']['micro']['hard_mask_dice'])
            check(name+' validation-only loss choice',winner['loss']==candidates['selected_loss']==r['loss'])
    result=dict(passed=True,check_count=len(checks),checks=checks)
    (output/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,check_count=len(checks))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');audit(parser.parse_args().output)
