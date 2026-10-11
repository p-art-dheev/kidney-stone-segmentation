import copy
import json
from pathlib import Path
import pytest
from kssd.splits import make_splits,validate_splits


@pytest.fixture
def manifest():
    return json.loads(Path('reports/dataset/manifest.json').read_text(encoding='utf-8'))


def test_actual_831_samples_grouped_reproducibly(manifest):
    a=make_splits(manifest);b=make_splits(manifest)
    assert a==b and validate_splits(a,manifest)
    assert sum(len(f['test']) for f in a['folds'])==831
    assert make_splits(manifest,seed=43)!=a


def test_sample_leakage_detected(manifest):
    splits=make_splits(manifest);fold=splits['folds'][0]
    fold['train'].append(fold['test'][0])
    with pytest.raises(ValueError,match='leakage'):validate_splits(splits,manifest)


def test_group_leakage_detected(manifest):
    splits=make_splits(manifest);fold=splits['folds'][0]
    # Swap two samples across roles without changing the partition; groups still cross roles.
    fold['train'][0],fold['test'][0]=fold['test'][0],fold['train'][0]
    with pytest.raises(ValueError,match='group leakage'):validate_splits(splits,manifest)


def test_changed_manifest_rejected(manifest):
    splits=make_splits(manifest);changed=copy.deepcopy(manifest);changed['mask_threshold']=0
    with pytest.raises(ValueError,match='fingerprint'):validate_splits(splits,changed)
