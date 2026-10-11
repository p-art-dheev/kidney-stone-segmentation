"""Differentiable training loss and stateful epoch-wide binary metrics."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kssd")
def dice_loss(y_true, y_pred):
    truth, prob = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    axes = tf.range(1, tf.rank(truth))
    intersection = tf.reduce_sum(truth * prob, axis=axes)
    denominator = tf.reduce_sum(truth + prob, axis=axes)
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


@tf.keras.utils.register_keras_serializable(package="kssd")
class HardMaskMetric(tf.keras.metrics.Metric):
    """Accumulate confusion counts; unequal batch sizes do not bias the metric."""

    def __init__(self, kind="dice", threshold=0.5, name=None, **kwargs):
        if kind not in {"dice", "iou", "precision", "recall"}:
            raise ValueError("Unknown metric kind")
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must lie in [0, 1]")
        super().__init__(name=name or f"hard_mask_{kind}", **kwargs)
        self.kind, self.threshold = kind, threshold
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype="float64")
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype="float64")
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype="float64")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("Pixel metric does not support sample weights")
        truth = tf.cast(y_true, tf.bool)
        pred = tf.cast(y_pred, tf.float32) >= self.threshold
        for variable, mask in [(self.tp, truth & pred), (self.fp, ~truth & pred),
                               (self.fn, truth & ~pred)]:
            variable.assign_add(tf.reduce_sum(tf.cast(mask, tf.float64)))

    def result(self):
        if self.kind == "dice":
            numerator, denominator = 2 * self.tp, 2 * self.tp + self.fp + self.fn
        elif self.kind == "iou":
            numerator, denominator = self.tp, self.tp + self.fp + self.fn
        elif self.kind == "precision":
            numerator, denominator = self.tp, self.tp + self.fp
        else:
            numerator, denominator = self.tp, self.tp + self.fn
        empty_value = 1.0 if self.kind in {"dice", "iou"} else 0.0
        return tf.where(denominator > 0, tf.math.divide_no_nan(numerator, denominator), empty_value)

    def reset_state(self):
        for variable in [self.tp, self.fp, self.fn]:
            variable.assign(0)

    def get_config(self):
        return {**super().get_config(), "kind": self.kind, "threshold": self.threshold}


def make_metrics(threshold=0.5):
    return [HardMaskMetric(kind, threshold) for kind in ["dice", "iou", "precision", "recall"]]
