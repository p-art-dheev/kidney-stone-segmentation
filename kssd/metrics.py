"""Hard binary metrics. Threshold boundary is inclusive (probability >= threshold)."""
import numpy as np


def confusion_counts(y_true, probabilities, threshold=0.5):
    truth = np.asarray(y_true)
    prob = np.asarray(probabilities)
    if truth.shape != prob.shape:
        raise ValueError("Truth and predictions must have identical shapes")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1]")
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Ground truth must be binary")
    truth, pred = truth.astype(bool), prob >= threshold
    return {"tp": int(np.count_nonzero(truth & pred)),
            "fp": int(np.count_nonzero(~truth & pred)),
            "fn": int(np.count_nonzero(truth & ~pred)),
            "tn": int(np.count_nonzero(~truth & ~pred))}


def metrics_from_counts(tp, fp, fn, tn=0):
    if min(tp, fp, fn, tn) < 0:
        raise ValueError("Counts cannot be negative")
    union = tp + fp + fn
    return {"hard_mask_dice": 2 * tp / (2 * tp + fp + fn) if union else 1.0,
            "hard_mask_iou": tp / union if union else 1.0,
            "hard_mask_precision": tp / (tp + fp) if tp + fp else 0.0,
            "hard_mask_recall": tp / (tp + fn) if tp + fn else 0.0}


def binary_metrics(y_true, probabilities, threshold=0.5):
    counts = confusion_counts(y_true, probabilities, threshold)
    return {**counts, **metrics_from_counts(**counts)}


def hard_mask_dice(y_true, probabilities, threshold=0.5):
    return binary_metrics(y_true, probabilities, threshold)["hard_mask_dice"]


def hard_mask_iou(y_true, probabilities, threshold=0.5):
    return binary_metrics(y_true, probabilities, threshold)["hard_mask_iou"]


def hard_mask_precision(y_true, probabilities, threshold=0.5):
    return binary_metrics(y_true, probabilities, threshold)["hard_mask_precision"]


def hard_mask_recall(y_true, probabilities, threshold=0.5):
    return binary_metrics(y_true, probabilities, threshold)["hard_mask_recall"]
