from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pytest
from kssd.attention_audit import validate
from kssd.final_audit import read

RECORDS=Path('reports/experiments/attention10fp32_20261012/attention_unet')


def arguments():
    return [read('reports/dataset/manifest.json'),read('reports/splits.json'),read('reports/experiments/baseline10fp32_final/modified_unet/summary.json')]


def test_attention_audit_rejects_missing_folds(tmp_path):
    with pytest.raises(ValueError,match='Missing completed fold'):
        validate(tmp_path,*arguments())


@pytest.mark.skipif(not (RECORDS/'summary.json').exists(),reason='Real Attention experiment is not complete yet')
def test_attention_records_only_audit_is_explicit_about_missing_weights_and_data():
    result=validate(RECORDS,*arguments(),figures=RECORDS.parent/'figures')
    assert result['status']=='verified' and result['saved_binary_masks']==831
    assert result['checkpoint_verification'].startswith('not performed')
    assert result['ground_truth_mask_verification'].startswith('not performed')
    assert len(result['folds'])==5 and len(result['figure_files_sha256'])==9


@pytest.mark.skipif(not (RECORDS/'summary.json').exists(),reason='Real Attention experiment is not complete yet')
def test_attention_audit_rejects_tampered_prediction_mask(tmp_path):
    folder=tmp_path/'records';shutil.copytree(RECORDS,folder)
    path=next((folder/'fold-1/predictions').glob('*.png'))
    with Image.open(path) as image:values=np.asarray(image).copy()
    values[0,0]=255-values[0,0];Image.fromarray(values).save(path)
    with pytest.raises(ValueError,match='checksum mismatch'):
        validate(folder,*arguments())
