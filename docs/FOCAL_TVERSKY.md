# Controlled loss experiment (not yet measured)

Question: does foreground Focal Tversky loss improve segmentation, particularly
for small annotated stone regions, relative to Dice under matched conditions?
This is a known loss, not a novel method. No improvement is claimed.

Real-data functionality smoke passed for Modified U-Net at batch 2, 512x512:
finite optimizer step and predictions, then zeroing/restoring weights reproduced
predictions. Reproduce in a fresh output directory:

```sh
python -m kssd.smoke --data data/kssd2025/data --output runs/new-focal-smoke --architectures modified_unet --loss focal_tversky
```

Add `--require-gpu` when testing GPU execution. The smoke supports either loss and
records the requested architectures explicitly; one requested model passing does
not imply all four were tested. This is not performance evidence.

Definition verified against Abraham and Khan, [arXiv:1810.07842](https://arxiv.org/pdf/1810.07842),
page 2, and the [authors' code](https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py).
The printed equation 4 uses exponent `1/gamma`, with gamma `4/3`; the code names
the exponent itself gamma `0.75`. The paper and released code weight FN by 0.7
and FP by 0.3. The implementation names the FN and FP weights explicitly.

For each image, using continuous sigmoid probabilities and binary ground truth:

`TI = (soft_TP + epsilon) / (soft_TP + 0.7 soft_FN + 0.3 soft_FP + epsilon)`

`L = (1 - TI)^0.75`

Foreground only; sum spatial/channel axes separately for each image, then average
images through Keras's standard loss reduction. No class/background sum or batch
flattening. Epsilon is **1e-6**, matching the baseline numerical-stability scale;
this differs from the authors' code's smoothing of 1 and batch flattening. These
choices avoid a one-pixel pseudocount materially influencing our smallest masks.
At exact TI=1 we define loss and gradient as zero to avoid the fractional power's
infinite derivative at zero. Elsewhere the stated power is evaluated directly.
This is a controlled application of the loss, not a full reproduction of their
attention/deep-supervision architecture or training protocol.

Priority: Modified U-Net, five identical saved outer/inner folds, seeds, initial
weights, augmentation, optimizer, learning rate, 10 epochs/batch 2, checkpoint
policy and threshold 0.5. Change only `loss` from `dice` to `focal_tversky`.
Use separate output roots and compare initial-weight hashes before conclusions.
Do not run the loss comparison until baseline execution prerequisites pass.

Size analysis will use annotated foreground pixel area, with per-fold tercile
boundaries fitted only on outer-training images (train + inner selection), never
test images. Small/medium/large denote dataset-relative area, not clinical stone
size; pixel spacing is unavailable. Apply the same boundaries to both losses.
Report per-bin counts, pooled binary metrics, per-image means, paired differences
and fold variability, including precision/recall tradeoffs and inconclusive results.

Comparison implementation checks complete checkpoint/count records, configs differing
only in loss, identical initial weights, computational source hashes and numerical
environment. It reports paired fold differences with sample SD and training-derived
size groups, and rejects missing/mismatched experiments before writing a report:

```sh
python -m kssd.compare_losses --baseline runs/baseline/modified_unet --alternative runs/focal/modified_unet --output runs/loss-comparison.json
```

Status: loss and comparison implementations only. Both real five-fold runs, measured
size analysis, failure figures and performance conclusions remain unverified.
