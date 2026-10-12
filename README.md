# Kidney Stone Segmentation — KSSD2025

Research implementation of Modified U-Net, U-Net++, U-Net3+ and TransU-Net for
binary segmentation in exported axial CT images. Reference: Bouzon et al.,
[*KSSD2025* (IEEE Access, 2025)](https://doi.org/10.1109/ACCESS.2025.3610027).
The corrected protocol differs from the paper and historical notebooks.

**The final non-training baseline audit is complete.** All twenty saved float32
folds, checkpoint hashes, counts, histories and matched configurations passed
independent verification. All 29 result plots were reviewed. The matched Focal
Tversky experiment is **pending**: its interrupted run has zero completed epochs
and no selected fold checkpoint. No further training was started in this pass.
The [central audit](reports/FINAL_PROJECT_AUDIT.md) records every completed,
exploratory, failed and pending run. See the [experiment manifest](reports/experiment_manifest.json),
[reproduction instructions](docs/EVALUATION_REPRODUCTION.md) and
[validation log](docs/VALIDATION_LOG.md).
The post-push regression suite passed **71 tests**; actual commands and outcomes
are in [release verification](reports/final_release_verification.json).

Verified **10-epoch, batch-two** baseline: mean ± sample SD across five
outer-fold pooled-pixel scores. These are not the paper's 150-epoch results.

| Architecture | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| U-Net++ | 0.3678 ± 0.2567 | 0.2490 ± 0.1894 | 0.3763 ± 0.2169 | 0.4317 ± 0.3622 |
| U-Net3+ | 0.3230 ± 0.4535 | 0.2805 ± 0.4098 | 0.3561 ± 0.4892 | 0.3014 ± 0.4364 |
| TransUNet | 0.4842 ± 0.3380 | 0.3749 ± 0.3166 | 0.5623 ± 0.3990 | 0.5063 ± 0.3042 |
| Attention U-Net | 0.7585 ± 0.2205 | 0.6515 ± 0.2864 | 0.7890 ± 0.2621 | 0.7803 ± 0.1996 |

The subsequently authorized Attention extension also completed five folds; its
mean Dice is 0.0325 below Modified U-Net under this budget. All 831 saved binary
masks and five local checkpoint identities passed independent checks; 80 tests
passed. [Attention results and failure analysis](docs/ATTENTION_RESULTS.md).

[Per-fold results](docs/BASELINE_RESULTS.md) and [comparison figure](reports/figures/baseline10fp32_final/comparison.png)
retain zero/poor folds. Short training, tiny foreground, unequal heuristic content
groups and unknown patient identities limit interpretation; no clinical readiness,
statistical significance or external generalization is established.

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
| Attention U-Net | 1,967,357 | Modified U-Net encoder/decoder with four additive spatial skip gates |

The final 10-epoch configs use float32 and fresh seeded weights/optimizers per
fold. Short mixed-precision GPU checks passed, but TransUNet developed NaN losses
during longer training; the matched comparison was restarted in float32.
See [measured acceleration limitations](docs/GPU_ACCELERATION.md).
TransU-Net uses Keras Reshape layers for compatibility;
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

### Attention U-Net extension — complete

Four additive spatial skip gates preserve Modified U-Net’s encoder/decoder and training protocol. All five folds completed; the pilot is retained. All 831 binary masks and checkpoint identities passed independent checks.

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| Attention U-Net | 0.7585 ± 0.2205 | 0.6515 ± 0.2864 | 0.7890 ± 0.2621 | 0.7803 ± 0.1996 |

See [full Attention results](docs/ATTENTION_RESULTS.md), [protocol](docs/ATTENTION_UNET.md) and [comparison figure](reports/experiments/attention10fp32_20261012/figures/comparison.png). This is a short-budget comparison, not a paper-budget reproduction or significance claim.

### Focal Tversky loss

Optional foreground Focal Tversky loss is numerically tested: FN weight 0.7, FP
weight 0.3, exponent 0.75, smoothing 1e-6. Its Modified U-Net config changes only
the loss relative to the Dice baseline:

```sh
python -m kssd.train --data data/kssd2025/data --config configs/focal_tversky/modified_unet.json --output runs/focal/modified_unet --require-gpu
```

This command requires a new training run; it was not executed in the final audit.
The interrupted prior run produced **no** completed CV fold or selected checkpoint.
Its definition, source,
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

Canonical compact records are in `reports/experiments/baseline10fp32_final/`;
figures are in `reports/figures/baseline10fp32_final/`. Current selected weights
are **local only**, under WSL `/var/tmp/kssd-workspace-20261012/runs/baseline_fp32_final`
and checksum-verified Windows backup `runs/persistent-backup-20261012/baseline_fp32_final`.
OneDrive/off-device synchronization is unverified. A fresh clone can validate
committed records without weights:

```sh
python -m kssd.final_audit --output runs/records-only-audit.json
```

To additionally verify recovered weights:

```sh
python -m kssd.final_audit --checkpoints runs/persistent-backup-20261012/baseline_fp32_final --output runs/checkpoint-audit.json
```

The [manifest](reports/experiment_manifest.json) lists all twenty checkpoint hashes
and requirements. Historical `Results/` archives are different checkpoints.

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

Remaining research work: new matched Focal training, loss comparison and its
training-derived size/error analysis; paper-budget experiments remain unrun.
The baseline comparison, figures and final non-training audit are complete.
