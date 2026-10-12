import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from kssd.models import build_model,attention_gate
from kssd.tf_metrics import dice_loss
from kssd.config import ExperimentConfig
from kssd.train import evaluate


def test_attention_shapes_and_finite_backward():
    tf.keras.backend.clear_session();tf.keras.mixed_precision.set_global_policy('float32')
    model=build_model('attention_unet',(32,32,1))
    x=tf.random.uniform((2,32,32,1));y=tf.cast(x>.8,tf.float32)
    with tf.GradientTape() as tape:
        output=model(x,training=True);loss=tf.reduce_mean(dice_loss(y,output))
    gradients=tape.gradient(loss,model.trainable_variables)
    assert output.shape==(2,32,32,1) and output.dtype==tf.float32
    assert np.isfinite(output.numpy()).all() and np.isfinite(loss.numpy())
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    assert any(np.any(g.numpy()!=0) for v,g in zip(model.trainable_variables,gradients) if 'attention_' in v.path)
    optimizer=tf.keras.optimizers.Adam(.001);optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    assert int(optimizer.iterations)==1
    for level in range(4):
        gate=model.get_layer(f'attention_{level}_coefficient')
        assert gate.output.shape[-1]==1
        assert gate.output.shape[1:3]==model.get_layer(f'attention_{level}_gated_skip').output.shape[1:3]


def test_attention_gate_zero_logits_halves_skip_and_broadcasts():
    skip=tf.keras.Input((8,8,16));decoder=tf.keras.Input((8,8,16))
    model=tf.keras.Model([skip,decoder],attention_gate(skip,decoder,8,'test_gate'))
    model.set_weights([np.zeros_like(w) for w in model.get_weights()])
    actual=model([np.ones((2,8,8,16)),np.ones((2,8,8,16))]).numpy()
    np.testing.assert_array_equal(actual,np.full((2,8,8,16),.5))


def test_gate_rejects_misaligned_spatial_shapes():
    with pytest.raises(ValueError,match='spatial'):
        attention_gate(tf.keras.Input((8,8,16)),tf.keras.Input((4,4,16)),8,'bad')


def test_saved_masks_use_single_evaluation_forward_pass(tmp_path):
    data=tmp_path/'data';(data/'image').mkdir(parents=True);(data/'label').mkdir()
    for name in ['a.tif','b.tif','c.tif']:
        Image.fromarray(np.full((32,32),255,np.uint8)).save(data/'image'/name)
        Image.fromarray(np.full((32,32),255,np.uint8)).save(data/'label'/name)
    calls=[]
    def model(images,training=False):
        calls.append(len(images));return tf.convert_to_tensor(images*.5)
    result=evaluate(model,data,['a.tif','b.tif','c.tif'],ExperimentConfig(architecture='attention_unet',input_size=32),tmp_path/'predictions')
    assert calls==[2,1] and result['counts']=={'tp':3072,'fp':0,'fn':0,'tn':0}
    assert len(list((tmp_path/'predictions').glob('*.png')))==3
    assert np.all(np.asarray(Image.open(tmp_path/'predictions/a.png'))==255)
    from kssd.attention_experiment import verify_masks
    assert len(verify_masks(tmp_path/'predictions',data,result,.5))==3
    Image.fromarray(np.zeros((32,32),np.uint8)).save(tmp_path/'predictions/a.png')
    with pytest.raises(ValueError,match='counts'):
        verify_masks(tmp_path/'predictions',data,result,.5)
