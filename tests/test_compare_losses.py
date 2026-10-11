import copy
import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
from test_reporting import fake_fixture
from kssd.compare_losses import area_boundaries,area_group,compare
from kssd.metrics import metrics_from_counts


def paired_fixture(tmp_path):
    roots=[tmp_path/'baseline',tmp_path/'alternative']
    for root in roots:
        root.mkdir();manifest,splits,_=fake_fixture(root)
    for root,loss in zip(roots,['dice','focal_tversky']):
        for fold in splits['folds']:
            folder=root/f"fold-{fold['fold']}"
            evaluation=json.loads((folder/'evaluation.json').read_text())
            evaluation['configuration']['loss']=loss
            weights=folder/'best.weights.h5';weights.write_bytes(b'synthetic-checkpoint-fixture')
            evaluation['checkpoint_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
            (folder/'evaluation.json').write_text(json.dumps(evaluation))
            (folder/'history.json').write_text(json.dumps({'val_hard_mask_dice':[1]}))
            source_files=['data.py','train.py','models.py','tf_metrics.py','augmentation.py',
                          'config.py','splits.py','metrics.py','reproducibility.py','losses.py']
            record={'status':'complete','config':evaluation['configuration'],'fold':fold,
                    'dataset_fingerprint':splits['dataset_fingerprint'],
                    'initial_weights_sha256':'synthetic-initial-weight-fixture',
                    'environment':{'deterministic_ops':True,'pythonhashseed':'42',
                                   'source_sha256':{source:'synthetic-same-source' for source in source_files}}}
            (folder/'provenance.json').write_text(json.dumps(record))
    return roots,manifest,splits


def test_area_boundaries_ignore_outer_test_labels():
    manifest={'samples':[{'id':str(i),'stone_area':i+1} for i in range(6)]}
    fold={'train':['0','1','2'],'selection':['3'],'test':['4','5']}
    expected=area_boundaries(manifest,fold)
    altered=copy.deepcopy(manifest);altered['samples'][4]['stone_area']=100000
    assert area_boundaries(altered,fold)==expected
    assert area_group(expected[0],expected)=='small'
    assert area_group(expected[1],expected)=='medium'


def test_paired_bins_include_every_sample_and_differences(tmp_path):
    roots,manifest,splits=paired_fixture(tmp_path)
    result=compare(*roots,manifest,splits)
    assert sum(bin['dice']['n_images'] for bin in result['pooled_size_groups'].values())==831
    assert all(value==0 for value in result['paired_fold_mean_difference'].values())
    assert len(result['folds'])==5
    # Explicitly synthetic alternative misses one pixel; verify signed differences.
    path=roots[1]/'fold-1/evaluation.json';evaluation=json.loads(path.read_text())
    row=evaluation['per_image'][0];area=row['stone_area'];row['tp']-=1;row['fn']+=1
    row.update(metrics_from_counts(**{key:row[key] for key in ['tp','fp','fn','tn']}))
    evaluation['counts']={key:sum(r[key] for r in evaluation['per_image']) for key in ['tp','fp','fn','tn']}
    evaluation['micro']=metrics_from_counts(**evaluation['counts'])
    evaluation['macro']={key:float(np.mean([r[key] for r in evaluation['per_image']])) for key in evaluation['micro']}
    path.write_text(json.dumps(evaluation))
    changed=compare(*roots,manifest,splits)
    pair=next(r for r in changed['folds'][0]['paired_per_image'] if r['id']==row['id'])
    assert np.isclose(pair['dice_difference_focal_minus_baseline'],-1/(2*area-1))
    assert changed['paired_fold_mean_difference']['hard_mask_dice']<0


@pytest.mark.parametrize('field',['loss','threshold'])
def test_mismatched_configs_rejected(tmp_path,field):
    roots,manifest,splits=paired_fixture(tmp_path)
    for path in roots[1].glob('fold-*/evaluation.json'):
        record=json.loads(path.read_text());record['configuration'][field]='dice' if field=='loss' else .8
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Expected Dice|mismatched training'):compare(*roots,manifest,splits)


@pytest.mark.parametrize('changed',['initial_weights_sha256','source'])
def test_mismatched_initial_weights_or_sources_rejected(tmp_path,changed):
    roots,manifest,splits=paired_fixture(tmp_path)
    path=roots[1]/'fold-1/provenance.json';record=json.loads(path.read_text())
    if changed=='source':record['environment']['source_sha256']['models.py']='changed-source'
    else:record[changed]='changed-weights'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='initial weights|computational source'):compare(*roots,manifest,splits)
