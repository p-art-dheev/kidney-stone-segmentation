"""Research-inspired foreground-only Focal Tversky loss; no performance claim."""
import math
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="kssd")
class FocalTverskyLoss(tf.keras.losses.Loss):
    """Per-image soft TP/(TP + .7 FN + .3 FP), exponent .75.

    Reduces spatial/channel axes, then Keras averages samples. No batch pooling,
    class weighting, thresholding, or background-class loss is used.
    """

    def __init__(self, fn_weight=0.7, fp_weight=0.3, exponent=0.75,
                 smooth=1e-6, name="focal_tversky_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        values=(fn_weight,fp_weight,exponent,smooth)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Loss parameters must be finite")
        if min(fn_weight,fp_weight)<0 or not math.isclose(fn_weight+fp_weight,1.0):
            raise ValueError("FN/FP weights must be nonnegative and sum to one")
        if exponent<=0 or smooth<=0:
            raise ValueError("Exponent and smoothing must be positive")
        self.fn_weight,self.fp_weight=fn_weight,fp_weight
        self.exponent,self.smooth=exponent,smooth

    def call(self, y_true, y_pred):
        truth=tf.cast(y_true,tf.float32)
        prob=tf.cast(y_pred,tf.float32)
        axes=tf.range(1,tf.rank(truth))
        tp=tf.reduce_sum(truth*prob,axis=axes)
        fn=tf.reduce_sum(truth*(1-prob),axis=axes)
        fp=tf.reduce_sum((1-truth)*prob,axis=axes)
        index=(tp+self.smooth)/(tp+self.fn_weight*fn+self.fp_weight*fp+self.smooth)
        error=tf.maximum(1-index,0.0)
        # The fractional power has an infinite derivative at zero. Evaluate its
        # inactive branch at one and explicitly choose zero loss/gradient there.
        positive=error>0
        base=tf.where(positive,error,tf.ones_like(error))
        return tf.where(positive,tf.pow(base,self.exponent),tf.zeros_like(error))

    def get_config(self):
        return {**super().get_config(),"fn_weight":self.fn_weight,
                "fp_weight":self.fp_weight,"exponent":self.exponent,"smooth":self.smooth}
