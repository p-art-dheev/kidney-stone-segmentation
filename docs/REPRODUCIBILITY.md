# Reproducibility and deliverable build

The completed workload is **250 training epochs across 25 fold runs** (five models,
five folds each, ten epochs per fold). This is aggregate compute work, not 250
epochs for any individual checkpoint. Every reported metric is from the actual
selected inner-validation checkpoint and untouched outer-fold evaluation.

## Recorded experiment environment

Linux/WSL Ubuntu, Python 3.12.3, TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.2.6,
SciPy 1.15.3, Pillow 11.1.0, scikit-learn 1.7.2, keras-unet-collection 0.1.13,
RTX 4070 Laptop GPU. Exact lock: `reports/resources/wsl_gpu.lock.txt`.

```sh
python3.12 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install -r reports/resources/wsl_gpu.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
export PYTHONHASHSEED=42
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
python -m pytest -q tests --basetemp tmp/reproduce-tests
```

The latest fresh full suite passed 80 tests, 1166 upstream warnings, 406.96 seconds
on the original Windows-mounted workspace. This duration differs from native
WSL earlier tests; it is not a training-time measurement.

## Saved-record reproduction (no training)

```sh
python -m kssd.final_audit --output runs/new-baseline-record-audit.json
python -m kssd.attention_audit --output runs/new-attention-record-audit.json
python -m kssd.reporting \
  --run reports/experiments/baseline10fp32_final/modified_unet \
  --output runs/new-summary.json
```

These auditors recompute metrics from integer counts, check every expected outer
identifier, split fingerprint, configuration and history selection. Without
`--checkpoints`, they explicitly do not claim weight verification. The committed
fresh full audit additionally verified all 25 local weight hashes and all 831
Attention saved masks against decoded ground truth.

## Data and selected weight recovery

Obtain the official dataset using `docs/DATASET.md`; do not substitute files with
matching names. All 838 decoded source pairs must match the frozen manifest.

```sh
python -m kssd.validate_data --data data/kssd2025/data \
  --output runs/new-dataset-audit --mask-threshold 127
python -m kssd.final_audit \
  --checkpoints runs/persistent-backup-20261012/baseline_fp32_final \
  --output runs/new-full-baseline-audit.json
python -m kssd.attention_audit \
  --checkpoints runs/persistent-backup-20261012/attention10fp32_20261012/attention_unet \
  --data data/kssd2025/data --output runs/new-full-attention-audit.json
```

Selected H5 weights remain ignored, locally checksum-backed-up, and **not published
on GitHub**. See `reports/experiment_manifest.json` for full paths and identities.
Remote OneDrive recovery is unverified. A fresh clone can verify compact records
and saved masks but needs recovered selected weights or retraining for new
model predictions. The old `Results/` .keras archives use the historical protocol
and cannot replace the selected corrected-run weights.

## New experiments

Choose a fresh output directory; the runner rejects overwriting completed runs.

```sh
python -m kssd.smoke --data data/kssd2025/data \
  --output runs/new-smoke --require-gpu
python -m kssd.train --data data/kssd2025/data \
  --config configs/10epoch/modified_unet.json \
  --output runs/new-modified-unet --require-gpu --save-predictions
```

Repeat with `unetpp.json`, `unet3plus.json`, `transunet.json`, and
`attention_unet.json` for matched new runs. Existing source hashes remain in
historical provenance. The architecture extension changes random-number
consumption, so identical seed values do not guarantee identical matching-layer
initial weights. Near-threshold binary masks may also vary across environments.

Focal Tversky's interrupted experiment has zero completed epochs/folds. It is an
unexecuted alternative, not a reported improvement. Extended configurations are
also unrun. A linear estimate for 150 epochs per fold for the four original models
at the measured batch-two speed is 72.95 GPU-hours; it is not a timing measurement
for a new run and excludes changed overhead or convergence behavior.

## Build the final article and Beamer presentation

Official Elsevier CAS 2.4 class, common style and BibTeX style are vendored under
`report/` without edits, retaining copyright and LPPL notices. The report uses
`cas-sc`; slides use `beamer`, not a PowerPoint conversion.

```sh
PYTHONPATH=. python scripts/build_project_deliverables.py
cd report
tectonic --keep-logs main.tex
cd ../presentation
tectonic --keep-logs main.tex
```

Tectonic 0.17.0 was downloaded into ignored scratch storage; a normal TeX Live
installation can instead use `latexmk -pdf main.tex` in each directory.
The first Tectonic run downloads TeX dependencies and therefore needs internet.
Both output PDFs are actual compilations of their delivered LaTeX sources.
Slides share `../report/figures/`, so distribute both directories together.

The data builder never trains or performs model inference. It uses canonical
JSON records, verified saved masks and source figure bytes. It requires the local
official data for the additional three-example montage. Its source map and figure
hashes are in `reports/compliance-20261013/deliverable_data.json`.

Replace title-author and contribution placeholders with verified team facts,
recompile and check layout before submission. Do not edit table numbers manually;
regenerate from canonical results. The deadline and separate lecture writing
guidance require team verification.
