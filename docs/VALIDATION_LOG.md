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

Training commit `f49cf3d` pushed and remote hash confirmed.

## Phase 4 - augmentation, configuration and real-data smoke

Reference page 8/Table 2 visually verified. Small paired affine augmentation implemented
with nearest-neighbor masks, image-only Gaussian noise and explicit sampling conventions.
Five augmentation tests passed; all tests after integration: **34 passed** in 64.78 s.
Upstream deprecation warnings remain. A reproducible paired CT/mask overlay was inspected.

`python -m kssd.smoke --data data/kssd2025/data --output runs/cpu-smoke`:
all four architectures passed actual **2x512x512x1** optimizer-step, finite-output and
weights save/load checks on native CPU. Initial compiled-batch times: Modified U-Net
8.17 s; U-Net++ 9.00 s; U-Net3+ 28.13 s; TransU-Net 16.25 s. These include compilation
and are not steady-state speed comparisons or segmentation performance evidence.

`python -m kssd.train --data data/kssd2025/data --config configs/representative.json --output runs/cpu-representative --fold 1 --selection-only`:
one full representative epoch **completed**: 446 training images, 206 inner-selection
images; all 223 training batches used. **363.70 s** training/selection duration on CPU,
batch size 2, augmentation enabled. Inner-selection binary Dice was 0 at this early
checkpoint; this is a measured training diagnostic, not an outer-test result. Checkpoint
and history were saved. No outer-test prediction was generated. Tests explicitly enforce
the no-outer-evaluation pilot path. Compact measured outputs are in reports/resources;
weights remain in ignored runs/cpu-representative/fold-1/best.weights.h5.

Ten-epoch/batch-2 and 150-epoch/batch-16 configs prepared for all architectures. Neither
full CV budget is completed at this gate. Native CPU full training would be costly
(a single Modified U-Net epoch already takes six minutes); GPU dependency installation
is still underway. GPU usability and full-run duration must be verified in Phase 5.
