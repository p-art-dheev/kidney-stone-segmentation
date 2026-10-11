import copy
import json
from pathlib import Path
import numpy as np
import pytest
from kssd.reporting import aggregate,compare_protocols,verify_evaluation
from kssd.metrics import metrics_from_counts
from kssd.splits import make_splits


def fake_fixture(tmp_path):
    manifest=json.loads(Path('reports/dataset/manifest.json').read_text());splits=make_splits(manifest)
    # Explicitly synthetic count fixtures test aggregation; never scientific output.
    for fold in splits['folds']:
        rows=[]
        for name in fold['test']:
            sample=next(s for s in manifest['samples'] if s['id']==name);tp=sample['stone_area']
            counts=dict(tp=tp,fp=0,fn=0,tn=512*512-tp)
            rows.append({'id':name,'stone_area':tp,**counts,**metrics_from_counts(**counts)})
        total={k:sum(r[k] for r in rows) for k in ['tp','fp','fn','tn']}
        result={'status':'complete','fold':fold['fold'],'seed':fold['seed'],'outer_test_count':len(rows),
                'per_image':rows,'counts':total,'micro':metrics_from_counts(**total),
                'macro':{k:float(np.mean([r[k] for r in rows])) for k in metrics_from_counts(**total)},
                'selected_epoch':1,'epochs_completed':1,'checkpoint_sha256':'synthetic-fixture',
                'configuration':{'architecture':'modified_unet','epochs':1,'threshold':.5,'mask_threshold':127},
                'dataset_fingerprint':splits['dataset_fingerprint'],'parameters':1,'training_seconds':0}
        folder=tmp_path/f"fold-{fold['fold']}";folder.mkdir();(folder/'evaluation.json').write_text(json.dumps(result))
    return manifest,splits,result


def test_recomputes_means_std_and_pooled_counts(tmp_path):
    manifest,splits,_=fake_fixture(tmp_path);summary=aggregate(tmp_path,manifest,splits,False)
    assert summary['n_folds']==5 and summary['fold_mean']['hard_mask_dice']==1
    assert summary['fold_sample_std']['hard_mask_dice']==0
    assert summary['pooled_counts']['tp']==sum(s['stone_area'] for s in manifest['samples'] if s['eligible'])


def test_incomplete_or_changed_metrics_rejected(tmp_path):
    manifest,splits,result=fake_fixture(tmp_path)
    result['micro']['hard_mask_dice']=.9
    with pytest.raises(ValueError,match='Metric'):verify_evaluation(result,splits['folds'][-1],manifest)
    (tmp_path/'fold-1/evaluation.json').unlink()
    with pytest.raises(ValueError,match='Missing completed fold'):aggregate(tmp_path,manifest,splits,False)


def test_changed_checkpoints_rejected(tmp_path):
    manifest,splits,_=fake_fixture(tmp_path)
    with pytest.raises(ValueError,match='checkpoint'):aggregate(tmp_path,manifest,splits)


def test_fair_protocol_validation(tmp_path):
    manifest,splits,_=fake_fixture(tmp_path);a=aggregate(tmp_path,manifest,splits,False);b=copy.deepcopy(a)
    b['configuration']['architecture']='unetpp';assert compare_protocols([a,b])
    b['configuration']['threshold']=.7
    with pytest.raises(ValueError,match='mismatched'):compare_protocols([a,b])
