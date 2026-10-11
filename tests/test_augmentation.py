import numpy as np
import pytest
from kssd.augmentation import augment_pair,AugmentationParameters
from kssd.metrics import binary_metrics


def phantom():
    mask=np.zeros((128,128,1),np.float32);mask[32:80,55:95]=1
    return mask.copy(),mask


def test_identity_and_exact_flip():
    x,y=phantom()
    identity=AugmentationParameters(0,0,0,0,0)
    a,b=augment_pair(x,y,42,identity)
    np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
    flip=AugmentationParameters(0,0,0,0,1)
    a,b=augment_pair(x,y,42,flip)
    np.testing.assert_array_equal(a,x[:,::-1]);np.testing.assert_array_equal(b,y[:,::-1])


def test_alignment_binary_labels_and_reproducibility():
    x,y=phantom();parameters=AugmentationParameters(noise_max_std=0)
    for seed in range(30):
        a,b=augment_pair(x,y,seed,parameters)
        assert binary_metrics(b,(a>=.5).astype(np.float32))['hard_mask_dice']>.99
        assert set(np.unique(b))<={0,1}
        a2,b2=augment_pair(x,y,seed,parameters)
        np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(b,b2)


def test_noise_is_image_only_and_bounded():
    x,y=phantom();a,b=augment_pair(x,y,1,AugmentationParameters(0,0,0,.0075,0))
    np.testing.assert_array_equal(b,y)
    assert not np.array_equal(a,x) and 0<=a.min()<=a.max()<=1


def test_bad_mask_rejected():
    x,y=phantom();y[0,0]=.5
    with pytest.raises(ValueError,match='binary'):augment_pair(x,y,1)


def test_validation_augmentation_is_disabled():
    from kssd.config import ExperimentConfig
    from kssd.train import dataset
    ids=['1.tif','10.tif']
    plain=list(dataset('data/kssd2025/data',ids,ExperimentConfig(augmentation=False)).as_numpy_iterator())
    configured=list(dataset('data/kssd2025/data',ids,ExperimentConfig(augmentation=True),training=False).as_numpy_iterator())
    for a,b in zip(plain,configured):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
