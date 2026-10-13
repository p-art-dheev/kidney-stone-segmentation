# Check pairing, split coverage, loss gradients and deadline handling.
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from strain.full.train import fp_loss, dataset, Timing


def test_fp_sensitive_loss_has_correct_finite_gradients():
    truth = tf.constant([[[[1.0], [0.0]]]])
    p = tf.Variable([[[[0.6], [0.8]]]])
    with tf.GradientTape() as tape:
        value = tf.reduce_mean(fp_loss(truth, p))
    gradient = tape.gradient(value, p).numpy().ravel()
    assert np.isfinite(value.numpy()) and np.isfinite(gradient).all()
    assert gradient[0] < 0 and gradient[1] > 0
    less_fp = tf.constant([[[[0.6], [0.1]]]])
    assert fp_loss(truth, less_fp).numpy()[0] < value.numpy()


def test_full_partition_covers_every_eligible_image_without_group_leakage():
    original = json.loads(
        Path("strain/results/experiment256_final_20261013/splits.json").read_text()
    )
    manifest = json.loads(Path("reports/dataset/manifest.json").read_text())
    partitions = {
        p: {
            s["id"]
            for s in manifest["samples"]
            if s["eligible"] and s["similarity_group"] in original["groups"][p]
        }
        for p in ["train", "validation", "test"]
    }
    assert {p: len(v) for p, v in partitions.items()} == dict(
        train=453, validation=288, test=90
    )
    assert set.union(*partitions.values()) == {
        s["id"] for s in manifest["samples"] if s["eligible"]
    }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not partitions[a] & partitions[b]
        assert not set(original["groups"][a]) & set(original["groups"][b])
    assert set(original["splits"]["train"]) <= partitions["train"]


def test_paired_augmentation_and_last_batch():
    tf.keras.utils.set_random_seed(20261013)
    x = np.zeros((5, 256, 256, 1), np.float32)
    x[:, 5:9, 70:80] = 1
    batches = list(dataset((x, x.copy()), 4, True))
    assert [len(a) for a, b in batches] == [4, 1]
    for a, b in batches:
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    plain = list(dataset((x, x.copy()), 4, False))
    np.testing.assert_array_equal(np.concatenate([a.numpy() for a, b in plain]), x)


def test_deadline_marks_partial_epoch_incomplete():
    timer = Timing(0)
    timer.set_model(SimpleNamespace(stop_training=False))
    timer.set_params(dict(steps=3))
    timer.on_epoch_begin(0)
    timer.on_train_batch_end(0)
    timer.on_epoch_end(0)
    assert timer.hit and timer.model.stop_training and timer.full == [False]
