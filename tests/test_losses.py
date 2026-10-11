import numpy as np
import pytest
import tensorflow as tf
from kssd.losses import FocalTverskyLoss


def test_formula_and_per_sample_reduction():
    truth=np.array([[1,1,0,0],[1,0,0,0]],dtype=np.float32).reshape(2,2,2,1)
    pred=np.array([[.8,.2,.5,0],[.5,.2,.1,0]],dtype=np.float32).reshape(2,2,2,1)
    loss=FocalTverskyLoss(reduction="none")
    tp=(truth*pred).sum((1,2,3));fn=(truth*(1-pred)).sum((1,2,3));fp=((1-truth)*pred).sum((1,2,3))
    expected=(1-(tp+1e-6)/(tp+.7*fn+.3*fp+1e-6))**.75
    np.testing.assert_allclose(loss(truth,pred),expected,rtol=1e-6)
    assert np.isclose(FocalTverskyLoss()(truth,pred),expected.mean())


def test_false_negative_weighting():
    truth=tf.constant([[[[1.],[0.]]]])
    missed=tf.constant([[[[0.],[0.]]]])
    extra=tf.constant([[[[1.],[1.]]]])
    assert float(FocalTverskyLoss()(truth,missed))>float(FocalTverskyLoss()(truth,extra))


@pytest.mark.parametrize("truth",[np.zeros((1,2,2,1)),np.ones((1,2,2,1))])
def test_perfect_and_empty_have_zero_loss_and_finite_gradient(truth):
    pred=tf.Variable(truth,dtype=tf.float32)
    with tf.GradientTape() as tape:value=FocalTverskyLoss()(truth,pred)
    gradient=tape.gradient(value,pred)
    assert float(value)==0 and np.isfinite(gradient).all()


def test_nonperfect_gradient_and_serialization():
    truth=tf.constant([[[[1.],[0.]]]]);pred=tf.Variable([[[[.2],[.8]]]])
    loss=FocalTverskyLoss()
    with tf.GradientTape() as tape:value=loss(truth,pred)
    gradient=tape.gradient(value,pred).numpy()
    assert np.isfinite(gradient).all() and gradient[0,0,0,0]<0 and gradient[0,0,1,0]>0
    restored=tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))
    assert restored.get_config()==loss.get_config()
    assert float(restored(truth,pred))==float(value)


def test_invalid_loss_parameters():
    for args in [{"fn_weight":-.1},{"exponent":0},{"smooth":0},{"exponent":float('nan')}]:
        with pytest.raises(ValueError):FocalTverskyLoss(**args)


def test_compiled_optimizer_step_and_matched_configs():
    import json
    from pathlib import Path
    from kssd.config import ExperimentConfig
    baseline=json.loads(Path('configs/10epoch/modified_unet.json').read_text())
    alternative=json.loads(Path('configs/focal_tversky/modified_unet.json').read_text())
    assert {k for k in baseline if baseline[k]!=alternative[k]}=={'loss'}
    ExperimentConfig(**alternative)
    model=tf.keras.Sequential([tf.keras.layers.Input((4,4,1)),
                               tf.keras.layers.Conv2D(1,1,activation='sigmoid')])
    model.compile(optimizer='adam',loss=FocalTverskyLoss())
    image=np.ones((2,4,4,1),dtype=np.float32)
    truth=np.zeros_like(image);truth[:,1:3,1:3]=1
    result=model.train_on_batch(image,truth)
    assert np.isfinite(result) and int(model.optimizer.iterations)==1
