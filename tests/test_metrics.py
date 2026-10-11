import numpy as np
import pytest
from kssd.metrics import binary_metrics
from kssd.tf_metrics import HardMaskMetric, dice_loss
import tensorflow as tf


@pytest.mark.parametrize("truth,pred,expected", [
    ([1, 0], [1, 0], (1, 1, 1, 1)),
    ([1, 0], [0, 0], (0, 0, 0, 0)),
    ([1, 0], [0, 1], (0, 0, 0, 0)),
    ([1, 1, 0], [1, 0, 1], (0.5, 1/3, 0.5, 0.5)),
    ([0, 0], [1, 0], (0, 0, 0, 0)),
    ([0, 0], [0, 0], (1, 1, 0, 0)),
    ([1, 1, 1], [0.4999, 0.5, 0.5001], (0.8, 2/3, 1, 2/3)),
])
def test_cases(truth, pred, expected):
    values = binary_metrics(truth, pred)
    for kind, target in zip(["dice", "iou", "precision", "recall"], expected):
        assert values[f"hard_mask_{kind}"] == pytest.approx(target)
        metric = HardMaskMetric(kind)
        metric.update_state(truth, pred)
        assert float(metric.result()) == pytest.approx(target)


def test_identity_and_batch_accumulation():
    rng = np.random.default_rng(42)
    truth = rng.integers(0, 2, (7, 9, 9, 1))
    pred = rng.random(truth.shape)
    values = binary_metrics(truth, pred)
    assert values["hard_mask_dice"] == pytest.approx(2*values["hard_mask_iou"]/(1+values["hard_mask_iou"]))
    for kind in ["dice", "iou", "precision", "recall"]:
        metric = HardMaskMetric(kind)
        metric.update_state(truth[:6], pred[:6])
        metric.update_state(truth[6:], pred[6:])
        assert float(metric.result()) == pytest.approx(values[f"hard_mask_{kind}"])
        clone = HardMaskMetric.from_config(metric.get_config())
        assert clone.threshold == 0.5
        metric.reset_state()
        assert float(metric.tp) == 0


def test_threshold_and_input_validation():
    assert binary_metrics([1], [0.7], threshold=0.8)["fn"] == 1
    for truth, pred, threshold in [([1], [float('nan')], .5), ([2], [.5], .5),
                                   ([1], [1.1], .5), ([1], [.5], 2), ([1], [], .5)]:
        with pytest.raises(ValueError):
            binary_metrics(truth, pred, threshold)


def test_dice_loss_is_soft_per_sample_and_has_finite_gradient():
    truth = tf.ones((2, 3, 3, 1))
    prob = tf.Variable(tf.fill((2, 3, 3, 1), 0.6))
    with tf.GradientTape() as tape:
        loss = dice_loss(truth, prob)
    assert loss.shape == (2,)
    assert np.isfinite(tape.gradient(loss, prob)).all()
    assert np.allclose(loss, 0.25)


def test_compiled_model_roundtrip(tmp_path):
    from kssd.tf_metrics import make_metrics
    model = tf.keras.Sequential([tf.keras.Input((4, 4, 1)), tf.keras.layers.Conv2D(1, 1, activation="sigmoid")])
    model.compile(optimizer="adam", loss=dice_loss, metrics=make_metrics())
    x = np.ones((3, 4, 4, 1), dtype=np.float32)
    result = model.train_on_batch(x, x, return_dict=True)
    assert all(np.isfinite(v) for v in result.values())
    before = model(x).numpy()
    model.save(tmp_path / "metric_roundtrip.keras")
    restored = tf.keras.models.load_model(tmp_path / "metric_roundtrip.keras")
    np.testing.assert_allclose(restored(x), before)
    assert restored.test_on_batch(x, x, return_dict=True)["hard_mask_dice"] >= 0


def test_all_notebooks_use_shared_binary_metrics():
    import ast
    import json
    from pathlib import Path
    for path in Path('.').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        assert 'from kssd.train import run_fold' in code
        assert 'from kssd.tf_metrics import dice_loss, make_metrics' in code
        assert 'def dice_coef' not in code
        assert 'max(h[' not in code
        ast.parse('\n'.join(line for line in code.splitlines() if not line.startswith('!')))
