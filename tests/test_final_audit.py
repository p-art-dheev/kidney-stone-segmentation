import json,shutil
from pathlib import Path
import pytest
from kssd.final_audit import read,verify_saved_fold,verify_figures


def fixture(tmp_path):
    source=Path('reports/experiments/baseline10fp32_final/modified_unet/fold-1')
    folder=tmp_path/'fold';shutil.copytree(source,folder)
    manifest=read('reports/dataset/manifest.json');splits=read('reports/splits.json')
    return folder,manifest,splits


def test_saved_record_audit_without_checkpoints_is_explicit(tmp_path):
    folder,manifest,splits=fixture(tmp_path)
    result=verify_saved_fold(folder,splits['folds'][0],manifest,splits)
    assert result['checkpoint_verified'] is False and result['outer_count']==179


@pytest.mark.parametrize('field,value,message',[
    ('precision_policy','mixed_float16','precision'),('optimizer_class','LossScaleOptimizer','optimizer')])
def test_actual_runtime_mismatch_rejected(tmp_path,field,value,message):
    folder,manifest,splits=fixture(tmp_path);provenance=read(folder/'provenance.json')
    if field=='precision_policy':provenance['environment'][field]=value
    else:provenance[field]=value
    (folder/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match=message):verify_saved_fold(folder,splits['folds'][0],manifest,splits)


def test_missing_checkpoint_rejected(tmp_path):
    folder,manifest,splits=fixture(tmp_path)
    with pytest.raises(ValueError,match='checkpoint'):verify_saved_fold(folder,splits['folds'][0],manifest,splits,tmp_path/'missing.h5')


def test_nonfinite_json_rejected(tmp_path):
    path=tmp_path/'nan.json';path.write_text('{"value": NaN}')
    with pytest.raises(ValueError,match='Non-finite'):read(path)


@pytest.mark.parametrize('field,value,message',[
    ('selected_epoch',1,'Selected checkpoint'),('optimizer_updates',1,'Optimizer update')])
def test_selected_epoch_and_update_budget_rejected(tmp_path,field,value,message):
    folder,manifest,splits=fixture(tmp_path);result=read(folder/'evaluation.json')
    assert result[field]!=value
    result[field]=value;(folder/'evaluation.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match=message):verify_saved_fold(folder,splits['folds'][0],manifest,splits)
