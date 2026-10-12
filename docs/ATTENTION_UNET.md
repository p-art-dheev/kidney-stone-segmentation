# Attention U-Net: matched short-budget experiment

User-authorized extension after the verified four-model baseline release.
Additive spatial skip gates are inspired by [Oktay et al., Attention U-Net](https://arxiv.org/abs/1804.03999).
This is a local adaptation of Modified U-Net, not an exact reproduction of the
paper's pancreas implementation or the KSSD paper's 150-epoch protocol.

Encoder filters 16/32/64/128/256, convolution blocks, BatchNorm, dropout and
transposed-convolution decoder remain identical in design to Modified U-Net.
After each decoder upsampling, gate the matching skip: two learned 1x1
projections to half the skip channels, addition, ReLU, scalar 1x1 projection,
sigmoid, and multiplication with the skip. The gate uses the upsampled decoder
as context at the same resolution. Four gates precede skip concatenation.

The config changes only architecture: ten epochs, batch two, float32, Adam
0.001, Dice loss, threshold 0.5, train-only paired augmentation and frozen
grouped folds/seeds. Earliest maximum inner hard Dice selects the checkpoint.
Fold 1 is the pilot and is retained as a CV fold; a stable finite complete
history, verified checkpoint, memory feasibility and valid saved masks permit
automatic continuation to folds 2–5. No outer metric controls continuation.
Previous architectures and their results are not retrained or edited.

The runner saves binary prediction PNGs during the single outer evaluation.
It verifies every saved mask against original counts and uses these masks for
success/failure panels without a second outer inference pass. H5 checkpoints
remain local/ignored; compact records, masks and selected panels can be pushed.

```sh
export PYTHONHASHSEED=42
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
python -m kssd.attention_experiment --data data/kssd2025/data --baseline runs/baseline_fp32_final/modified_unet --output runs/attention10fp32_20261012 --export reports/experiments/attention10fp32_20261012
```

Requires the verified WSL GPU runtime, official dataset and existing baseline
checkpoint directory. The output must be new. Run status/results will be
recorded from actual artifacts; no missing fold is imputed. The historical
baseline source hashes differ after adding this architecture and optional mask
persistence; actual new hashes and original provenance are retained.

Implementation regression gate: `CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=42
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 python -m pytest -q tests
--basetemp tmp/attention-tests-verified`: **77 passed**, 1166 upstream warnings,
61.50 seconds. [Full log](../reports/resources/attention_tests.txt).
An initial collection command also picked up a preserved scratch test copy;
its duplicate-module error is retained in the collection-failure log. Restricting
collection to `tests/` resolved it without discarding the scratch files.

Full-size GPU smoke passed: 1,967,357 parameters (22,124 more than Modified
U-Net), a 512x512 batch of two, float32/Adam/Dice, finite forward and optimizer
update, and exact checkpoint prediction round trip. First update: 11.62 seconds;
peak TensorFlow allocation: 1,277,059,840 bytes. Evidence:
[GPU smoke record](../reports/resources/attention_gpu_smoke.json).
Training implementation commit: `0de39f5`; native workspace was clean at launch.
GPU smoke results are functionality evidence, not a CV performance score.

Fold-1 pilot passed: all ten epochs finite, selected epoch 9, 179 outer masks
verified against saved counts, checkpoint SHA-256 verified and locally backed up.
Training time 378.20 seconds; Dice/IoU/precision/recall
0.474540/0.311080/0.344465/0.762456. Automatic fold-2 training started.
Largest-FP and lowest-Dice panels were inspected from saved masks: the model
detects some annotated foreground but also predicts bright background structures;
1226.tif misses all 57 foreground pixels (TP/FP/FN 0/511/57). These failures
are retained. No training setting or epoch is changed using these outer results.
The pilot-stage snapshot above is preserved; the experiment subsequently
completed all five folds. [Final measured results](ATTENTION_RESULTS.md) replace
the incomplete status. The full suite passed **80 tests**, 1166 upstream warnings,
in **77.50 seconds**; [final test log](../reports/resources/attention_final_tests.txt).
All 831 saved masks, five checkpoint hashes, histories, selection epochs and
matched configurations passed independent checks. No real-data run failed or
was retrained. The primary run's source commit remains `0de39f5`; the standalone
records-only auditor was added afterward and is not an optimizer change.

After all five folds complete, the committed records and binary masks can be
audited without TensorFlow, raw images or checkpoint downloads:

```sh
python -m kssd.attention_audit --output runs/attention-records-audit.json
```

For the full local integrity check, also supply recovered weights and the
official dataset. This checks all saved masks against original ground truth,
without another model forward pass:

```sh
python -m kssd.attention_audit --checkpoints runs/persistent-backup-20261012/attention10fp32_20261012/attention_unet --data data/kssd2025/data --output runs/attention-full-audit.json
```

These commands require a complete five-fold export; they reject missing folds.
Without weights/data they explicitly state which checks were not performed.
Record aggregation alone:

```sh
python -m kssd.reporting --run runs/persistent-backup-20261012/attention10fp32_20261012/attention_unet --output runs/attention-summary-check.json
```
