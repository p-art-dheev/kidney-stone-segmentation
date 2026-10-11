# Binary evaluation policy

`kssd.metrics` is the NumPy reference. `kssd.tf_metrics` supplies registered,
stateful Keras metrics and a separate differentiable, per-image soft Dice loss.
Predictions equal to the threshold are positive; the default threshold is 0.5.
Ground truth must be binary. All four evaluation metrics use exactly the same
binary prediction, with integer TP/FP/FN/TN counts and no smoothing.

Dice = 2TP/(2TP+FP+FN); IoU = TP/(TP+FP+FN); precision = TP/(TP+FP);
recall = TP/(TP+FN). Both-empty Dice/IoU are 1. Undefined precision/recall
are 0, so empty predictions do not receive perfect precision. These are explicit
reporting conventions, not clinical judgments. Counts must accompany scores.

Primary fold metrics pool pixels (micro averaging); Keras accumulates epoch-wide
counts, including the last incomplete batch. Per-image metrics and their mean
(macro averaging) may also be reported, separately labelled. Dice/IoU identity
holds for each common count set, including the both-empty convention; it need
not hold between separately macro-averaged values.

Legacy Results assets and notebook outputs used incompatible metrics, independent
epoch maxima, and final-epoch checkpoints. They are preliminary historical
evidence, superseded as a reporting protocol, and cannot be retrospectively
validated without matching data, split/checkpoint provenance and reevaluation.
Updating code does not validate those old outputs. Until inner checkpoint
selection is integrated, notebook final-epoch scores are exploratory only.
