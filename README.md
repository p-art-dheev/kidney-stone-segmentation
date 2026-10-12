# Kidney Stone Segmentation — KSSD2025

Research implementation of Modified U-Net, U-Net++, U-Net3+ and TransU-Net for
binary segmentation in exported axial CT images. Reference: Bouzon et al.,
[*KSSD2025* (IEEE Access, 2025)](https://doi.org/10.1109/ACCESS.2025.3610027).
The corrected protocol differs from the paper and historical notebooks.

**Validation is in progress.** Shared metrics, preprocessing, grouped splits,
checkpoint selection, augmentation and model weight round trips have been tested.
All four models passed actual 512×512 CPU and GPU optimizer-step smoke checks.
The full test suite passed 59 tests in both environments. A full Modified U-Net
training/inner-selection epoch completed in 363.70 s on CPU and 110.47 s on GPU.
The [verified WSL GPU environment](docs/GPU_ENVIRONMENT.md) includes exact dependencies.
Corrected five-fold performance results, the matched Focal Tversky comparison and
prediction/error figures are **not yet available**. No best model is established.
See the [validation log](docs/VALIDATION_LOG.md) for measured checks and open gates.

## Dataset and protocol

[KSSD2025 on Kaggle](https://www.kaggle.com/datasets/murillobouzon/kssd2025-kidney-stone-segmentation-dataset)
contains 838 matched 512×512 TIFF image/mask pairs. The validator checked every pair.
Images are uint8 grayscale, normalized by 255; no resizing is performed. Decode
mask palettes, then use **mask >127**. The legacy >0 rule included faint background
artifacts. Exported intensities are not calibrated Hounsfield units.

Seven redundant decoded-image copies are excluded through the manifest, leaving
831 eligible images; original files remain intact. A prespecified thumbnail
similarity heuristic forms 39 content clusters for leakage reduction. Patient/scan
IDs are unavailable: these clusters are not patients and cannot establish patient
independence. Details, hashes, overlays and limitations: [dataset audit](docs/DATASET.md).

Five seeded outer group folds each hold out different images. An inner group split
within each outer-training partition selects the earliest checkpoint attaining
maximum inner hard-mask Dice. The selected weights are evaluated once on the outer
test fold. Every eligible image appears in outer test exactly once. All batches,
including incomplete final batches, are used. See [training methodology](docs/TRAINING.md).

Evaluation uses probabilities **>=0.5** for Dice, IoU, precision and recall from the
same confusion counts. Fold scores pool pixels; per-image means are saved separately.
Across-fold error bars use sample standard deviation, not confidence intervals.
Both-empty Dice/IoU are 1; undefined precision/recall are 0. Training Dice loss uses
continuous probabilities and is separate from binary metrics: [metric policy](docs/METRICS.md).

## Architectures and budgets

| Architecture | Parameters at 512×512 | Implementation |
|---|---:|---|
| Modified U-Net | 1,945,233 | Filters 16/32/64/128, bottleneck 256, BatchNorm/dropout |
| U-Net++ | 1,967,713 | keras-unet-collection nested skips |
| U-Net3+ | 2,374,583 | keras-unet-collection full-scale skips |
| TransU-Net | 4,747,553 | Local library-based hybrid, embed 192, 4 heads, 3 transformer blocks |

The 10-epoch configs use mixed_float16 with float32 probabilities, losses and
trainable variables, automatic loss scaling, and fresh seeded weights/optimizers
per fold. All four passed full-resolution GPU checks at batch 2. TransU-Net uses Keras Reshape layers for compatibility;
installed library source is never modified. This library variant is not claimed
to exactly reproduce the original TransUNet authors' implementation.

The shared preliminary budget is **10 epochs, batch 2, Adam, LR 0.001, Dice loss**.
Configs use identical train-only paired augmentation: small rotation/shift/zoom,
horizontal flips and image-only noise. Masks use nearest-neighbor interpolation.
Exact distributions and deviations from unspecified paper details are documented
in [augmentation](docs/AUGMENTATION.md). Paper-budget **150 epochs/batch 16** configs
are provided but have not run; memory feasibility must be verified first.

## Setup and reproduction

Run commands from the repository root with Python **>=3.11**. The verified native
CPU environment uses Python 3.13.5, TensorFlow 2.20.0 and Keras 3.13.2. Native Windows
TensorFlow 2.20 is CPU-only; full experiments need a verified Linux/WSL CUDA environment.
The isolated WSL Python 3.12 runtime has passed actual GPU training checks;
see [GPU evidence](docs/GPU_ENVIRONMENT.md).

```sh
python -m venv .venv
```

Activate it (PowerShell: `.venv\Scripts\Activate.ps1`; Linux: `source .venv/bin/activate`),
then install:

```sh
python -m pip install -e ".[train,test]"
```

On Linux/WSL additionally install `python -m pip install "tensorflow[and-cuda]==2.20.0"`.
Verify TensorFlow detects a GPU before long runs. Dependencies are pinned in
`pyproject.toml`; each run also records all installed-package versions and source hashes.
If Linux venv creation fails because ensurepip is unavailable, install the appropriate
Python venv package or bootstrap pip into an isolated `--without-pip` venv; the local
WSL setup used an official downloaded pip wheel without modifying system packages.

Download/unzip the official Kaggle archive into `data/kssd2025/`; it contains a
nested `data/` with `image/` and `label/`. See [dataset instructions](docs/DATASET.md)
for the verified archive hash. Create `tmp/` first (PowerShell:
`New-Item -ItemType Directory -Force tmp`; Linux: `mkdir -p tmp`). Then:

```sh
python -m kssd.validate_data --data data/kssd2025/data --output runs/dataset-audit
python -m kssd.splits --manifest reports/dataset/manifest.json --output runs/splits-check.json
python -m pytest -q --basetemp tmp/tests
python -m kssd.smoke --data data/kssd2025/data --output runs/new-smoke
```

The dataset/split outputs should match the committed protocol; do not replace it
silently with another dataset version. For GPU smoke add `--require-gpu`. Set
`PYTHONHASHSEED=42` before Python (PowerShell: `$env:PYTHONHASHSEED='42'`; Linux:
`export PYTHONHASHSEED=42`). First run a selection-only resource pilot, which never
evaluates outer test:

```sh
python -m kssd.train --data data/kssd2025/data --config configs/representative.json --output runs/new-pilot --fold 1 --selection-only
```

After verifying resources, run all five folds per model. For example:

```sh
python -m kssd.train --data data/kssd2025/data --config configs/10epoch/modified_unet.json --output runs/baseline/modified_unet --require-gpu
python -m kssd.reporting --run runs/baseline/modified_unet --output runs/baseline/modified_unet-summary.json
```

Repeat with `unetpp`, `unet3plus` and `transunet` configs/output roots. Do not compare
different budgets or protocols. The aggregator rejects missing folds, changed
checkpoint hashes, mismatched counts/metrics and incompatible configurations.
It also rejects short fixed-budget runs and inconsistent or nonfinite histories.
`--require-gpu` fails before creating a run if TensorFlow cannot see a GPU; omit it
only when deliberately choosing the much slower CPU execution path.
Existing fold directories are rejected; choose new roots for reruns. Interrupted
runs are incomplete until verified evaluation outputs exist.

The notebooks are thin wrappers loading these same 10-epoch configs. Run from
this checkout; `KSSD_DATA` and `KSSD_RUN` override data/output locations. They
contain no precomputed corrected results or hidden Kaggle training logic.

## Research-inspired experiment

Optional foreground Focal Tversky loss is numerically tested: FN weight 0.7, FP
weight 0.3, exponent 0.75, smoothing 1e-6. Its Modified U-Net config changes only
the loss relative to the Dice baseline:

```sh
python -m kssd.train --data data/kssd2025/data --config configs/focal_tversky/modified_unet.json --output runs/focal/modified_unet --require-gpu
```

This experiment has **not** produced measured CV results. Its definition, source,
reduction choices and planned training-derived size bins are documented in
[Focal Tversky](docs/FOCAL_TVERSKY.md). No novelty or improvement is claimed.

## Files and artifacts

| Location | Purpose |
|---|---|
| `kssd/`, `tests/` | Shared implementation and automated checks |
| `configs/` | Preliminary, paper-budget, pilot and controlled loss configs |
| `reports/dataset/`, `reports/splits.json` | Audited manifest, overlays and frozen folds |
| `reports/resources/`, `reports/legacy_archive_audit.json` | Measured functionality/resource and compatibility checks |
| `docs/` | Audit, methodology, acceptance status and validation log |
| `runs/` (ignored) | New weights, histories, provenance, evaluations and reports |
| `Results/` | Preserved historical archives/figures, superseded protocol |
| `paper/` | Supplied reference paper |

Keep new large weights and downloaded data outside Git. Retain complete fold
directories; provenance records initial-weight and selected-checkpoint hashes,
config, split IDs, hardware, source and dependencies. Share checkpoints through an
artifact store if needed; publish compact validated tables/plots only after strict
aggregation passes. No existing user assets or remote history have been removed.

## Legacy preliminary results — superseded and unvalidated

| Model | Soft Dice | Soft IoU | Binary precision | Binary recall |
|---|---|---|---|---|
| Modified U-Net | 0.5799 ± 0.0126 | 0.4256 ± 0.0130 | 0.9610 ± 0.0781 | 0.7371 ± 0.1241 |
| U-Net++ | 0.5361 ± 0.0187 | 0.3843 ± 0.0184 | 0.7727 ± 0.1200 | 0.9964 ± 0.0027 |
| U-Net3+ | 0.5891 ± 0.0150 | 0.4337 ± 0.0143 | 1.0000 ± 0.0000 | 0.6623 ± 0.0417 |
| TransU-Net | 0.2850 ± 0.1551 | 0.1914 ± 0.1156 | 0.9916 ± 0.0169 | 0.8743 ± 0.2435 |

These historical values mix probability and binary metrics, independently select
metric maxima and use flawed mask preprocessing. They do not describe one selected
checkpoint and cannot support a model ranking. Old plots/archives are retained for
provenance. All 20 archives passed finite-weight restoration checks; that establishes
compatibility, not performance or defensible checkpoint selection.

Remaining gates: corrected five-fold comparisons; matched
loss experiment and size/error analysis; result-derived plots; final audit. No
clinical readiness or external-hospital generalization is established.
