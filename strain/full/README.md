# Full training-partition experiment

Run from the repository root in the existing WSL CUDA runtime:

```bash
/var/tmp/kssd-gpu312-20261012/bin/python -m strain.full.run --data /var/tmp/kssd-workspace-20261012/data/kssd2025/data --output strain/results/full256_rerun
```

One supervisor starts the 3600-second clock before imports, source/cache checks,
diagnostics, batch pilots, controlled loss trials, training, testing and reporting.
It reserves time for evaluation/reporting and terminates only its own worker if
the deadline is reached. Incomplete results are unavailable rather than zero.

Expand the prior frozen content-group assignments to all 831 eligible unique
images: 453 training, 288 validation, 90 testing. Use every training image for
every model. Patient IDs are unavailable: heuristic content groups cannot prove
patient independence. Exact duplicate exclusions follow the existing manifest;
no difficult sample is removed. Recheck every original decoded hash and verify
all cached bilinear images and nearest-neighbor masks against the originals.
Use float32 uint8/255 normalization and the existing >127 mask encoding. No
source image, mask, main training module or historical checkpoint is modified.

Warm-start each architecture from its own preserved 256-pixel checkpoint.
Verify that all original fitting IDs are inside the expanded training partition.
No pretrained backbone is added and no architecture is replaced. The target
is 50 additional epochs, not 50 epochs from fresh initialization. Baseline
weights remain eligible at epoch zero if fine-tuning never improves validation.

Measure stable batch sizes 4/8/16 with real backward passes and full validation;
discard pilot updates. Choose the fastest measured stable batch. Previous AMP
failures justify retaining float32. All training is sequential. Cached tensors,
paired train-only flips, deterministic validation, bounded workers/prefetch and
validation once per epoch avoid repeated decoding and preprocessing.

For U-Net++ and TransUNet, compare two five-epoch warm-start trials at Adam
LR 0.0003: current per-image Dice + 0.25 BCE versus per-image Tversky
(FP weight 0.7, FN weight 0.3) + the same 0.25 BCE. Only loss changes in that
comparison. Rank by pooled validation Dice at threshold 0.5; current loss wins
ties. Continue the winning trajectory up to the shared measured epoch budget.
For Modified U-Net and U-Net3+, keep the existing loss and use gentle LR 0.0001.
Plateau scheduling monitors validation Dice, with patience eight and LR floor
0.00005; early stopping uses patience twelve after ten epochs. The target model
order gives U-Net++/TransUNet priority if time becomes tight.

Keep one best validation checkpoint, plus the necessary latest stage weights
for continuation. Selected-trajectory training/validation loss and all four
hard-mask metrics are logged to CSV/JSON and plotted without dataset-size labels.
Losing five-epoch trials remain separately labelled, not silently merged into
the selected trajectory's epoch count. CSV and plot epoch numbers start at one.

Seal all checkpoints and calibration choices before loading any test image or
mask. Primary comparison is pooled pixel Dice/IoU/precision/recall at threshold
0.5. Optional calibration maximizes validation Dice while retaining at least
85% of that selected checkpoint's common-threshold validation recall. Both
test results use the same single prediction pass and remain separately labelled.
Both-empty Dice/IoU are one; zero-denominator precision/recall are zero.

The expanded test set differs from the earlier 16-image test. Comparing raw
numbers across those protocols is not a matched estimate of improvement. The
historical split has previously reported outcomes, so this is an engineering
experiment, not an independent external validation study. All tuning uses only
training/validation data. Raw CT diagnostic panels stay local; compact numeric
evidence and curves can be committed. Historical file hashes are verified at
completion. Actual elapsed time, partial epochs, errors, validation and test
results are recorded explicitly.
