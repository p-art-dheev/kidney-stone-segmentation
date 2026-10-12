# Reproduce the saved baseline evaluation audit

Run from the checkout root, with Python >=3.11. The [manifest](../reports/experiment_manifest.json)
contains configurations, seeds, checkpoint hashes, runtime versions and storage
availability. The [central audit](../reports/FINAL_PROJECT_AUDIT.md) explains the
completed baseline and pending Focal experiment. No command below trains a model.

## From a fresh clone: committed records

```sh
python -m pip install -e .
python -m kssd.final_audit --output runs/records-only-audit.json
```

This independently checks frozen grouped splits, all outer IDs, per-image counts,
mask areas, metrics, full finite histories, inner-only selected epochs, actual
precision/optimizer, configuration matching and deterministic figure selection.
It explicitly reports checkpoint verification as not performed. No TensorFlow,
raw dataset or new checkpoint download is needed for this records-only check.
Byte hashes of text records can differ with Git line-ending conversion; dataset
and split hashes in the manifest use canonical JSON (sorted keys, compact
separators, UTF-8). H5/PNG identities are byte hashes.

## With recovered selected weights

The new weights are not published on GitHub. On the training machine they exist
at `/var/tmp/kssd-workspace-20261012/runs/baseline_fp32_final` and in the ignored
Windows checkout folder `runs/persistent-backup-20261012/baseline_fp32_final`.
These are local requirements; cloud sync was not verified. Recover the complete
architecture/fold-N directories, preserving the compact records and
`best.weights.h5`. The original tracked legacy `.keras` files are not substitutes.

```sh
python -m kssd.final_audit --checkpoints runs/persistent-backup-20261012/baseline_fp32_final --output runs/checkpoint-audit.json
python -m kssd.reporting --run runs/persistent-backup-20261012/baseline_fp32_final/modified_unet --output runs/modified-unet-summary-check.json
```

The first checks all twenty selected H5 hashes. The second recomputes one model's
five-fold summary from saved evaluations and verifies its weights; repeat for
`unetpp`, `unet3plus`, `transunet`. This does not rerun outer inference or choose
new checkpoints. Saved per-image confusion counts can be recomputed algebraically
without images; validating their relationship to pixels requires raw data and
restored inference. The existing qualitative regeneration did this for 24
selected cases, not a second full twenty-fold evaluation.

## Regenerate verified qualitative figures by inference

Recover the selected weights and complete records above, install
`python -m pip install -e ".[train,test]"`, and obtain the exact official data
under `data/kssd2025/data` following [dataset instructions](DATASET.md).
Use the [verified WSL runtime](GPU_ENVIRONMENT.md) for numerical reproducibility.
Linux/WSL:

```sh
export PYTHONHASHSEED=42
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
python -m kssd.figures --root runs/persistent-backup-20261012/baseline_fp32_final --data data/kssd2025/data --output runs/recreated-baseline-figures
```

PowerShell environment equivalents are `$env:PYTHONHASHSEED='42'`,
`$env:TF_NUM_INTRAOP_THREADS='2'`, `$env:TF_NUM_INTEROP_THREADS='2'`.
Native Windows TensorFlow 2.20 is CPU-only. Different devices/libraries can change
near-threshold masks; generation checks restored case counts and refuses a
mismatch. It uses original evaluation batch shape/order and does not retrain.
See [figure methodology](FIGURES.md). Choose a new output directory.

## Tests and pending experiment

```sh
python -m pytest -q --basetemp tmp/tests
```

The suite includes tiny synthetic optimizer fixtures, not another real-data CV
run. Post-push outcomes are in the final release verification report. Focal has
zero completed epochs/folds and no resumable epoch checkpoint. Its smoke weights
cannot supply a matched loss comparison; new training is required and was not
started in the final pass.
