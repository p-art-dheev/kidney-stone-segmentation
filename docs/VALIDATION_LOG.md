# Measured validation log

## Phase 0 - audit

`530f9b7`: audit document committed and push confirmed by `git ls-remote origin refs/heads/main`.
Inspected notebook code, archive metadata, reference-paper text, legacy prediction figure,
Git status/remotes/history, installed dependencies, free disk and NVIDIA/TF/WSL GPU visibility.
No training or dataset metrics measured.

## Phase 1 - evaluation metrics

Shared NumPy integer-count reference and registered TensorFlow loss/stateful metrics;
all four notebook architectures use them. Independent epoch maxima removed; final-epoch
scores are explicitly exploratory until inner checkpoint selection is implemented.
Stale notebook outputs cleared; original figures/models preserved and old README table labelled.

`python -m pytest tests/test_metrics.py -q --basetemp=tmp/pytest3`: **12 passed** in 8.16 s;
17 upstream Keras/NumPy deprecation warnings. Tested perfect/missed/disjoint/partial/empty
cases, boundary >=0.5, invalid inputs, Dice/IoU identity, last-batch accumulation,
finite loss gradients, a real optimizer step and compiled model save/load/evaluation,
plus notebook syntax/integration. `git diff --check` passed.

Windows sandbox runs could not create the default pytest temp directory and later stalled
in model I/O. The final passing run used repository-local temp files outside the sandbox,
with TF intra/inter-op threads limited to 2. No dataset evaluation has been measured.

Metric commit `53db6fa` pushed and remote hash confirmed.

## Phase 2 - dataset integrity

Official archive downloaded (78.5 MiB), resumed after a 180-second transfer timeout;
archive SHA256 and CRC recorded in DATASET.md. Both `>0` and `>127` audits executed
over all 838 pairs. Initially raw-mask hash differences falsely looked like conflicting
labels for duplicate images; comparison of the actual binary masks found zero
disagreements. Validator now tests conflicts after the declared binarization.

Final `python -m kssd.validate_data --data data/kssd2025/data --output runs/dataset-audit --mask-threshold 127`:
**passed**, 838 valid pairs, 831 eligible unique images, 39 content similarity groups,
zero empty masks. Synthetic integrity suite: **5 passed** in 0.47 s. Full mask
histogram and duplicate checks measured; representative overlay visually inspected.
Notebooks use strict pairing, explicit normalization/binarization and eligibility.
Patient/scan grouping remains unavailable; heuristic group limits documented.

Dataset commit `db8b8bf` pushed and remote hash confirmed.

## Phase 3 - splitting, model lifecycle and checkpoint selection

Saved deterministic nested content-group splits over all 831 eligible images. Four split
tests detect altered manifests, sample overlap and group leakage and confirm exact
outer-test coverage. Strict shared configuration, streaming non-dropping batches, fresh
seeded models/optimizers, inner hard-Dice checkpoint selection and one outer evaluation
implemented for all architectures. Notebooks now delegate to the shared implementation.
TransU-Net uses explicit Keras Reshape without editing installed source. Model weights
round trips and finite probability outputs tested for all four builders at 32x32;
this reduced synthetic shape is a functionality test, not resized dataset training.

Full suite before final interface cleanup: **29 passed** in 28.29 s on native CPU,
including a real one-epoch synthetic nested training/checkpoint/outer-evaluation run.
Upstream Keras/NumPy/gast deprecation warnings were observed; library input structure
was normalized to a single input/output afterward. `compileall` and diff checks pass.
Full-resolution GPU smoke testing is a separate Phase 4 gate, not claimed here.

WSL Ubuntu-22.04 Python 3.10 could not resolve Keras 3.13.2 (requires Python >=3.11).
An isolated Python 3.12 environment in the other Ubuntu distribution is downloading
the pinned GPU dependencies; no WSL system packages or existing environments modified.
Python project requirement corrected to >=3.11.

Final cleanup recheck: `tests/test_training.py` **8 passed** in 26.79 s, no library
input-structure warning remains. Upstream Keras/gast deprecations persist.
