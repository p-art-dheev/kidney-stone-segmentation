# Attention U-Net — full training partition, 50 epochs

Run from the repository root using the existing WSL GPU runtime:

```bash
/var/tmp/kssd-gpu312-20261012/bin/python -m strain.attention.run --data /var/tmp/kssd-workspace-20261012/data/kssd2025/data --output strain/results/attention256_new_run
```

Use all 453 frozen training images, 288 validation images and 90 held-out test
images. These cover all 831 eligible unique samples. Training on validation or
test images would compromise evaluation, so the complete training partition
is used rather than merging the partitions. The original content-group split
is retained; patient independence cannot be guaranteed without patient IDs.

Reuse the four-gate Attention U-Net builder in `kssd/models.py`. Start from
fresh initialization, not old cross-validation or baseline weights. Recheck
all original/cached pairs through `strain.full.data.prepare`, then reuse the
in-memory paired augmentation and batching in `strain.full.train.dataset`.
Inputs remain 256×256; CT normalization is uint8 /255, masks stay binary after
nearest-neighbor resizing. Test arrays are loaded only after selection is sealed.

The initial mixed-precision pilot passed batches 8 and 16, but batch 32 caused
an out-of-memory error followed by a cuDNN allocator failure. That attempt is
preserved without claiming any training epoch. The corrected fresh process
measures float32 batches 4/8 and mixed float16 batches 8/16. Choose the
fastest finite configuration by backward-pass and full-validation timings.
Validate metrics, weights, raw gradients and accepted optimizer updates.
Mixed precision uses Keras automatic loss scaling and a float32 sigmoid head.
Discard all pilot updates before the real training run. Resource or numerical
failures are retained in pilot.json and trigger a stable fallback.

The restart uses output `attention256_50epochs_20261014_v2` and
`--started 1791919273.408109` to include the original failed pilot and restart
delay in the reported total wall-clock time.

Run exactly 50 epochs with no early stopping, use Adam at 0.001, per-image
Dice + 0.25 BCE and validation-Dice plateau scheduling (patience 8, floor 1e-5).
Verify every epoch consumes every training sample, including the final short
batch. Select the best checkpoint using validation Dice at threshold 0.5.
Save a latest full model with optimizer every 10 epochs and retain the best
weights locally under `strain/checkpoints/<run-name>/`.

Save per-epoch training/validation loss, Dice, IoU, precision and recall to
CSV/JSON and architecture-only curves. Optional threshold calibration uses
validation only and retains at least 85% of the common validation recall.
Report common-threshold test metrics as primary and calibrated metrics as
secondary; both share one held-out prediction pass after the selection lock.

`strain.attention.audit` independently checks epoch/sample coverage, split
separation, checkpoint hashes, pooled metric arithmetic, threshold selection
and preservation of historical files. The new report records actual elapsed
time and results. No 50-epoch or performance claim is made if a run fails.
