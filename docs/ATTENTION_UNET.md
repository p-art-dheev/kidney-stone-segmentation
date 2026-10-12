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
