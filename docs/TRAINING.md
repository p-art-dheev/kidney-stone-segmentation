# Reproducible nested group training

Use Python >=3.11 and pinned package extras: `python -m pip install -e ".[train,test]"`.
Native Windows TensorFlow 2.20 is CPU-only; use a CUDA-capable Linux/WSL GPU environment
for full-resolution training. An isolated WSL environment is being validated; its status
and measured runtime are recorded in VALIDATION_LOG.md, not assumed from installation.
Use `--require-gpu` on training commands when GPU execution is intended; the runner
then fails before creating a run if TensorFlow cannot see one.

## Splits

```sh
python -m kssd.splits --manifest reports/dataset/manifest.json --output reports/splits.json
```

Seed 42, GroupKFold(5, shuffle=True) over fixed content clusters, followed by an inner
GroupShuffleSplit(test_size=0.2, seed=42+fold). Fractions apply to **groups**, not images.
Saved identifiers partition 831 eligible images in every fold; outer test contains every
image exactly once. Training / inner selection / outer test counts:

| Fold | Training | Inner selection | Outer test |
|---|---:|---:|---:|
| 1 | 446 | 206 | 179 |
| 2 | 455 | 97 | 279 |
| 3 | 665 | 48 | 118 |
| 4 | 589 | 70 | 172 |
| 5 | 606 | 142 | 83 |

These unequal sizes reflect conservative cluster preservation, not missed samples.
Content clusters are not known patients; see DATASET.md. No test pixels/labels determine
checkpoint selection. The inner selection set influences checkpoint choice and is not a
final unbiased test. Each outer fold is evaluated once after loading the selected weights.

## Lifecycle

`kssd.train.run_fold` validates the dataset fingerprint, split exclusivity and live source
hashes; clears the Keras session; seeds Python/NumPy/TensorFlow to 42+fold; enables
deterministic TensorFlow ops; constructs a fresh model and optimizer with zero iterations.
Set `PYTHONHASHSEED=42` **before** starting Python (setting it after launch has no effect).
Models are float32. Train-only shuffling has an explicit seed; tested spatial transforms
are enabled in the committed experiment configs (see AUGMENTATION.md). Streaming batches include the
last incomplete batch. There is no training-state transfer across folds or architectures.

Checkpoint policy: maximum inner-selection hard-mask Dice; earliest epoch wins ties.
Optional early stopping uses the same prespecified metric. CSVLogger writes every epoch;
JSON history, parameter count, initial-weight hash, dependency versions, hardware,
training duration, selected epoch and checkpoint hash are saved. Weights-only HDF5 is used
for all models, avoiding unregistered library-layer full-model deserialization. Rebuild
from saved configuration before loading weights. Existing output directories are rejected.
Interrupted directories without evaluation.json are incomplete, never accepted results.

TransU-Net preserves the library's no-backbone, patch-size-1 topology and legacy attention
settings; local Keras Reshape layers replace the unsupported symbolic `tf.reshape`.
No installed source is modified and no source-code `exec` patch is used. This library
variant is not claimed to be the authors' exact TransUNet implementation.

Primary scores pool pixel confusion counts within each outer fold. Per-image means are
stored separately as macro scores. The first inference batch is a warm-up; latency excludes
that batch, disk I/O and preprocessing and includes `.numpy()` device synchronization.
No confidence intervals or clinical generalization claims are implied by five folds.

The three notebooks are thin entry points into this same implementation. Their names
retain the historical 10-epoch budget; old assets remain labelled preliminary.
