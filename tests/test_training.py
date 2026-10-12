import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from kssd.config import ExperimentConfig
from kssd.models import build_model,ARCHITECTURES
from kssd.reproducibility import set_seeds
from kssd.train import dataset


@pytest.mark.parametrize('architecture',ARCHITECTURES)
def test_architecture_weight_roundtrip_and_output(architecture,tmp_path):
    tf.keras.backend.clear_session();set_seeds(42)
    model=build_model(architecture,(32,32,1))
    x=np.ones((1,32,32,1),dtype=np.float32)
    before=model(x,training=False).numpy()
    assert before.shape==x.shape and np.isfinite(before).all()
    assert before.min()>=0 and before.max()<=1
    path=tmp_path/f'{architecture}.weights.h5';model.save_weights(path)
    model.set_weights([np.zeros_like(w) for w in model.get_weights()]);model.load_weights(path)
    np.testing.assert_allclose(model(x,training=False),before,atol=1e-7)


def test_fresh_seeded_model_and_optimizer():
    weights=[]
    for i in range(2):
        tf.keras.backend.clear_session();set_seeds(43)
        model=build_model('modified_unet',(32,32,1))
        model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')
        assert int(model.optimizer.iterations)==0
        weights.append(model.get_weights())
    for a,b in zip(*weights):np.testing.assert_array_equal(a,b)


def test_actual_final_batch_is_not_dropped():
    config=ExperimentConfig(batch_size=2)
    ids=['1.tif','10.tif','1000.tif']
    batches=list(dataset('data/kssd2025/data',ids,config).as_numpy_iterator())
    assert [x.shape[0] for x,y in batches]==[2,1]
    assert all(set(np.unique(y))<={0,1} for x,y in batches)


def test_invalid_configs():
    for kwargs in [{'epochs':0},{'batch_size':0},{'threshold':2},{'optimizer':'unknown'},{'precision':'float16'}]:
        with pytest.raises(ValueError):ExperimentConfig(**kwargs)


@pytest.mark.parametrize('architecture',ARCHITECTURES)
def test_mixed_precision_keeps_probabilities_and_variables_float32(architecture):
    try:
        tf.keras.backend.clear_session();tf.keras.mixed_precision.set_global_policy('mixed_float16');set_seeds(43)
        model=build_model(architecture,(32,32,1))
        output=model(np.ones((1,32,32,1),dtype=np.float32),training=False).numpy()
        assert output.dtype==np.float32 and np.isfinite(output).all()
        assert all(str(v.dtype)=='float32' for v in model.trainable_variables)
        assert output.min()>=0 and output.max()<=1
        model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')
        assert isinstance(model.optimizer,tf.keras.mixed_precision.LossScaleOptimizer)
    finally:
        tf.keras.mixed_precision.set_global_policy('float32');tf.keras.backend.clear_session()


def test_nested_runner_synthetic_end_to_end(tmp_path,monkeypatch):
    from PIL import Image
    from kssd.validate_data import validate
    from kssd.splits import make_splits
    from kssd.train import run_fold
    root=tmp_path/'synthetic';(root/'image').mkdir(parents=True);(root/'label').mkdir()
    for i in range(15):
        image=np.full((32,32),i*16,dtype=np.uint8);mask=np.zeros_like(image);mask[8:12,8:12]=255
        Image.fromarray(image).save(root/'image'/f'{i}.tif');Image.fromarray(mask).save(root/'label'/f'{i}.tif')
    manifest=validate(root,15,32);splits=make_splits(manifest,3)
    config=ExperimentConfig(epochs=1,n_folds=3,input_size=32,batch_size=2,augmentation=True)
    result=run_fold(root,manifest,splits,config,1,tmp_path/'run')
    output=tmp_path/'run/fold-1'
    assert result['status']=='complete' and result['selected_epoch']==1
    assert result['outer_test_count']==len(splits['folds'][0]['test'])
    assert len(result['per_image'])==result['outer_test_count']
    assert (output/'history.csv').exists() and (output/'best.weights.h5').exists()
    assert json.loads((output/'provenance.json').read_text())['status']=='complete'
    with pytest.raises(ValueError,match='already exists'):run_fold(root,manifest,splits,config,1,tmp_path/'run')
    def forbid_outer(*args,**kwargs):raise AssertionError('Runtime pilot must not look at outer test')
    monkeypatch.setattr('kssd.train.evaluate',forbid_outer)
    pilot=run_fold(root,manifest,splits,config,1,tmp_path/'pilot',evaluate_outer=False)
    assert pilot['status']=='selection_only'
    assert not (tmp_path/'pilot/fold-1/evaluation.json').exists()
