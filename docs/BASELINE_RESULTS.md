# Measured five-fold baseline results

All twenty final float32 outer-fold evaluations completed on the RTX 4070 Laptop
GPU using the frozen grouped splits, ten epochs, batch two, Adam LR 0.001,
Dice loss, training-only paired augmentation and threshold 0.5. Each checkpoint
was selected by inner-selection hard-mask Dice, with earliest-epoch tie breaking.
All outer images were evaluated once per architecture. These are short-budget
results, not reproduction of the paper's 150-epoch/batch-16 protocol.

Mean ± sample standard deviation of each fold's pooled pixel metrics:

| Architecture | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| U-Net++ | 0.3678 ± 0.2567 | 0.2490 ± 0.1894 | 0.3763 ± 0.2169 | 0.4317 ± 0.3622 |
| U-Net3+ | 0.3230 ± 0.4535 | 0.2805 ± 0.4098 | 0.3561 ± 0.4892 | 0.3014 ± 0.4364 |
| TransUNet | 0.4842 ± 0.3380 | 0.3749 ± 0.3166 | 0.5623 ± 0.3990 | 0.5063 ± 0.3042 |

These uncertainties describe fold variability, not confidence intervals. Modified
U-Net has the highest measured mean Dice in this fixed experiment; no statistical
significance or external generalization claim follows. Content groups are heuristic
duplicate/near-duplicate groups, not verified patient identities; folds are unequal.
U-Net3+ had zero outer Dice in three folds and recovered in two. U-Net++ and
TransUNet also varied substantially. Poor outcomes are retained, not discarded
or replaced by independent metric maxima. Ten epochs and tiny foreground regions
limit interpretation; higher-budget optimization remains unrun.

The strict aggregator rechecked expected outer IDs, per-image confusion counts,
mask area and total pixels, all derived metrics, finite fixed-budget histories,
selected epoch, and checkpoint SHA-256. The protocol comparator confirmed all
four architecture configs match except architecture. Compact complete records:
`reports/experiments/baseline10fp32_final/<architecture>/`; native checkpoints:
`/var/tmp/kssd-workspace-20261012/runs/baseline_fp32_final/<architecture>/fold-N/`.
Completed checkpoints also have checksum-verified local Windows backup copies
under ignored `runs/persistent-backup-20261012/`. Cloud sync is unverified.

Earlier partial float32 and failed mixed-precision runs remain preserved separately.
`reports/experiments/baseline10amp_verified/` contains exploratory mixed-precision
results for three architectures; TransUNet failed before a complete first fold.
Those runs are excluded from this table. The matched Focal Tversky comparison,
full figure review and final audit are still pending.

Baseline comparison, all fold training/inner-selection curves, and deterministic
highest/lowest-Dice, largest false-negative/false-positive, smallest-area and
middle-Dice prediction panels are in `reports/figures/baseline10fp32_final/`.
Generation reloaded each selected checkpoint using the original evaluation batch
shape/order and verified displayed confusion counts against saved evaluations.
The complete index records selection rules, source cases, metric values and file
hashes. Comparison, U-Net3+ curves and a large-false-negative Modified U-Net case
were visually inspected. The latter has TP/FP/FN 41/20/622 and Dice 0.1133:
the model largely misses the foreground despite its higher aggregate mean.
Displayed sample-SD error bars can extend below zero and are not confidence
intervals. Full baseline count verification and runtime provenance checks passed;
all twenty records show actual float32/Adam and complete status.
