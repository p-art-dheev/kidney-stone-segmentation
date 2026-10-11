"""Saved outer test / inner checkpoint-selection splits over content groups."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_splits(manifest, n_folds=5, seed=42, selection_fraction=0.2):
    if not manifest['passed']:
        raise ValueError('Dataset manifest did not pass integrity checks')
    samples=[s for s in manifest['samples'] if s['eligible']]
    ids=np.asarray([s['id'] for s in samples]); groups=np.asarray([s['similarity_group'] for s in samples])
    folds=[]
    outer=GroupKFold(n_splits=n_folds,shuffle=True,random_state=seed)
    for i,(development,test) in enumerate(outer.split(ids,groups=groups),1):
        inner=GroupShuffleSplit(n_splits=1,test_size=selection_fraction,random_state=seed+i)
        fit,selection=next(inner.split(development,groups=groups[development]))
        folds.append({'fold':i,'seed':seed+i,'train':ids[development[fit]].tolist(),
                      'selection':ids[development[selection]].tolist(),'test':ids[test].tolist()})
    result={'schema_version':1,'dataset_fingerprint':fingerprint(manifest),'seed':seed,
            'n_folds':n_folds,'selection_fraction':selection_fraction,
            'policy':'GroupKFold outer + GroupShuffleSplit inner; groups are content similarity clusters, not patient IDs',
            'folds':folds}
    validate_splits(result,manifest)
    return result


def validate_splits(splits,manifest):
    if splits['dataset_fingerprint'] != fingerprint(manifest):
        raise ValueError('Dataset fingerprint differs from saved splits')
    samples={s['id']:s for s in manifest['samples'] if s['eligible']}
    seen=Counter()
    if len(splits['folds']) != splits['n_folds']:
        raise ValueError('Incorrect number of folds')
    for fold in splits['folds']:
        roles=[set(fold[role]) for role in ['train','selection','test']]
        if any(not role for role in roles):
            raise ValueError('Empty split role')
        if any(len(fold[role]) != len(set(fold[role])) for role in ['train','selection','test']):
            raise ValueError('Duplicate sample within a role')
        if set.union(*roles) != set(samples):
            raise ValueError('Split does not partition eligible samples')
        for i in range(3):
            for j in range(i):
                if roles[i]&roles[j]: raise ValueError('Sample leakage between roles')
                if {samples[n]['similarity_group'] for n in roles[i]} & {samples[n]['similarity_group'] for n in roles[j]}:
                    raise ValueError('Content group leakage between roles')
        seen.update(fold['test'])
    if set(seen) != set(samples) or any(count != 1 for count in seen.values()):
        raise ValueError('Every eligible sample must be in outer test exactly once')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='reports/dataset/manifest.json');parser.add_argument('--output',default='reports/splits.json')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--folds',type=int,default=5)
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    result=make_splits(manifest,args.folds,args.seed)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for fold in result['folds']:print(f"Fold {fold['fold']}: train={len(fold['train'])}, inner selection={len(fold['selection'])}, outer test={len(fold['test'])}")


if __name__=='__main__':main()
