# Isolated fast experiment

The [five clearly named model files](models/README.md) provide `build()` and
`load_best()` entry points for Modified U-Net, U-Net3+, U-Net++, TransUNet and
Attention U-Net, with an explanation of every output file.

Attention U-Net has completed [50 epochs](results/attention256_50epochs_20261014_v2/report.md)
on the complete frozen training partition. Its [runner and protocol](attention/README.md)
reuse the verified preprocessing and held-out evaluation. The file index below
includes Attention U-Net's checkpoint, loss/metric logs and curves.

The latest completed four-model run is [full256_20261013_v2](results/full256_20261013_v2/report.md).
Open the [completed model file index](COMPLETED_MODELS.md) to find each model's
architecture entry point, results, loss/metric logs, curves and checkpoint.
Training and evaluation code for this run is in [full/](full/README.md).

The earlier subset comparison is `results/experiment256_final_20261013/`.
The earlier `experiment256_20261013/` is retained preliminary evidence and was
superseded after a validation-only stopping-protocol amendment. The shared clock
included both attempts: 1748.75 seconds (29 minutes 9 seconds). All four final
models completed, with 28 relevant checks and the independent result audit passing.

Run from the repository root in the existing CUDA TensorFlow environment:

```bash
python -m strain.run_experiment --data /var/tmp/kssd-workspace-20261012/data/kssd2025/data --output strain/results/experiment256_rerun --reuse-verified-preview strain/results/experiment256_final_20261013
```

The supervisor includes preprocessing, runtime imports, all four timing pilots,
training, validation, testing and report generation in its 3600-second budget.
It terminates its worker early enough to write partial results before the deadline.
Each model gets an equal share of remaining time and at most 150 epochs, with
18-epoch validation Dice patience after a 60-epoch warm-up (only 13 batches per
epoch; allow BatchNorm running statistics to settle). Inspect `resize_preview.png` before training,
then write `preview_verified.json` in the result directory with inspection notes.
No training starts before this marker. This inspection time counts toward the budget.
For a rerun on this workstation, the command above reuses the existing inspection
only if the new preview PNG is byte-identical. If downloading a fresh checkout
without the local preview, omit that flag and inspect the generated preview.

All code, caches, evidence and weights stay under strain. Original files are
verified against decoded hashes in the frozen main-project manifest and never
modified. Cached uint8 images use Pillow bilinear resize to 256 square, then
float32 / 255, appropriate for these exported grayscale images rather than HU.
Palette-decoded masks use >127 and nearest-neighbor binary resize. The full
dataset resize audit lists lost connected components and entirely vanished
stones; none are removed from the experiment. Round-trip mask Dice quantifies
resampling distortion separately from predicted metrics.

Exactly floor(10% of 831 eligible unique images) are selected, uniformly within
seeded disjoint existing content-group partitions: 51 train, 16 validation,
16 test. Patient/scan metadata is absent: these groups do not prove patient
independence. Exact identifiers are frozen before training, identical for all
models. Empty resized masks are retained. Train-only paired flips use a shared
concatenated image/mask tensor. Validation/test preprocessing is deterministic.

Reuse `kssd.models.build_model` without removing model components. All four
accept 256 square grayscale inputs, including the unchanged TransUNet patch
and attention structure. Pilot actual backward passes at both 256 and 512
with equal batch size; the 512 pilot upsamples cached train pairs solely to
measure compute, not original-resolution accuracy. Discard pilot weights.
Float32 is chosen for numerical stability given the existing AMP diagnostics;
the smaller resident arrays, cached decoding, prefetching and batch size four
provide acceleration without precision-dependent instability.

Loss is per-image Dice + 0.25 pixel-mean BCE, Adam 0.001 and a validation-loss
plateau scheduler. This is a fixed fair protocol, not a hyperparameter sweep.
Select the highest pooled validation Dice checkpoint (earliest ties), reload
and check reproducible inference, then evaluate test once at threshold 0.5.
Save epoch training/validation loss and all four metrics in CSV and JSON.
Keras CSV epoch indices start at zero; plots label epochs starting at one.
Each model's `training_curves.png` contains loss, Dice, IoU, precision and recall,
with training/validation lines and the selected epoch, without dataset-size labels.
Final metrics pool TP/FP/FN across all held-out test pixels; per-image metrics
are also saved. Empty truth and prediction yield Dice/IoU 1; precision/recall
with zero denominators are 0. Macro and micro aggregation stay separate.

Tiny subset scores have substantial uncertainty. Resized scores and the old
512-resolution cross-validation table are different protocols; improvement
cannot be inferred from their numerical difference alone. Missing model/test
results remain unavailable. Best weights remain local and ignored by Git.
