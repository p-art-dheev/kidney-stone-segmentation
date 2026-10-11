import numpy as np
import pytest
from PIL import Image
from kssd.data import paired_files, load_pair, read_grayscale
from kssd.validate_data import validate


def fixture_data(tmp_path):
    for folder in ['image','label']: (tmp_path/folder).mkdir()
    for i in range(3):
        image=np.full((16,16),i*50,dtype=np.uint8)
        mask=np.zeros((16,16),dtype=np.uint8); mask[2:4,3:5]=255
        Image.fromarray(image).save(tmp_path/'image'/f'{i}.tif')
        Image.fromarray(mask).save(tmp_path/'label'/f'{i}.tif')
    return tmp_path


def test_integrity_and_non_resizing(tmp_path):
    root=fixture_data(tmp_path); report=validate(root,3,16)
    assert report['passed'] and report['count']==3
    assert all(s['stone_area']==4 for s in report['samples'])
    x,y=load_pair(root/'image/1.tif',root/'label/1.tif')
    assert x.shape==y.shape==(16,16,1)
    assert x.max()==pytest.approx(50/255)
    assert set(np.unique(y))=={0,1}


def test_missing_pair_rejected(tmp_path):
    root=fixture_data(tmp_path); (root/'label/0.tif').unlink()
    with pytest.raises(ValueError,match='Pair mismatch'): paired_files(root)


def test_shape_and_count_errors(tmp_path):
    root=fixture_data(tmp_path)
    Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(root/'label/1.tif')
    report=validate(root,838,16)
    assert not report['passed'] and len(report['errors'])==2
    with pytest.raises(ValueError,match='dimensions'):load_pair(root/'image/1.tif',root/'label/1.tif')


def test_duplicate_images_conflicting_labels(tmp_path):
    root=fixture_data(tmp_path)
    Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(root/'image/1.tif')
    Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(root/'label/1.tif')
    report=validate(root,3,16)
    assert report['exact_duplicate_groups']==[['0.tif','1.tif']]
    assert not report['passed']


def test_mask_policy_explicit_and_no_uint16_clipping(tmp_path):
    root=fixture_data(tmp_path)
    mask=np.zeros((16,16),dtype=np.uint8);mask[0,:3]=[1,127,255]
    Image.fromarray(mask).save(root/'label/0.tif')
    assert load_pair(root/'image/0.tif',root/'label/0.tif',0)[1].sum()==3
    assert load_pair(root/'image/0.tif',root/'label/0.tif',127)[1].sum()==1
    Image.fromarray(np.zeros((16,16),dtype=np.uint16)).save(root/'image/0.tif')
    with pytest.raises(ValueError,match='uint8'):read_grayscale(root/'image/0.tif')
