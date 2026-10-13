# Findings before the isolated experiment

The existing `kssd/data.py`, `models.py`, `train.py`, `metrics.py`,
`tf_metrics.py`, dataset manifest, grouped split policy and GPU diagnostic
records were inspected before implementing this experiment. Existing source
and results are preserved.

838 matched 512-square uint8 TIFF pairs are available. Masks are indexed with
grayscale palettes, with decoded intensities in 0–9 and 247–255. CT image
intensities are exported display intensities, not Hounsfield units. The main
loader's palette decoding, >127 binary policy and division by 255 are correct.
The older notebooks' >0 mask rule incorrectly includes faint background pixels.
Seven exact duplicate pairs have one canonical eligible member: 831 eligible
unique images. Similarity groups, rather than fabricated patient IDs, are reused
for separation. No supplied mapping establishes patient-independent splits.

The screenshot's historical table is not the existing project's corrected
five-fold result table. `docs/METRICS.md` documents incompatible legacy metrics,
independent epoch maxima and final-epoch checkpoint inconsistencies. Precision
or recall alone cannot establish good segmentation: false positives lower
precision and missed stones lower recall. All four final experiment metrics
must instead use one common prediction, threshold and confusion count set.

Current model builders have float32 sigmoid outputs, accept 256-square inputs,
and keep their original components. Binary metrics accumulate pooled counts
correctly, including short final batches. Dice loss is per image, rather than
an accidentally batch-flattened loss. Main pipeline augmentation pairs images
and masks; this experiment also transforms a concatenated pair together.

Foreground is extremely sparse: minimum three original pixels and 50 masks
with at most ten pixels. Nearest-neighbor resizing can erase individual
components even when the full mask survives. This is documented for all 838
images and no difficult sample is removed. The round-trip resize overlap is
an annotation distortion measurement, not a prediction accuracy score.

CUDA is available through the existing WSL TensorFlow 2.20 / Keras 3.13.2
runtime on RTX 4070 Laptop. Native Windows TensorFlow is not used for this run.
Previous AMP diagnostics include unstable/failed architectures. Float32,
cached 256-pixel arrays, batch four, bounded prefetch and sequential models are
therefore used. Real 256/512 backward-pass timing records provide the measured
speed improvement; the 512 pilot is a compute-only upsample of cached pairs.

A preliminary validation-only review showed that 18-epoch patience can stop
learning prematurely with only 13 batches per epoch. A 60-epoch stopping warm-up
was then fixed for all models without inspecting preliminary test scores.
The original deadline covers both attempts. Preliminary evidence is retained;
only the explicitly named final run is the final comparison. This amendment
does not guarantee that low validation scores are caused by BatchNorm: tiny
training size, group distribution differences, foreground imbalance and
optimization behavior also limit performance.
