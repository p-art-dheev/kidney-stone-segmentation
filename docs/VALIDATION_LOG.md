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

Augmentation commit `5d71929` pushed and remote hash confirmed.

## Phase 5a - verified evaluation aggregation (full experiment gate pending)

Strict aggregator requires all five outer evaluations, exact test IDs, matching ground
truth/pixel counts, identical configuration across folds and checkpoint hashes. It
recomputes per-image/fold metrics, macro means, pooled counts and across-fold sample
standard deviations; earliest maximum inner-Dice checkpoint epoch is checked against
the saved history. Cross-architecture comparisons reject protocol differences.
Four synthetic aggregation/error-detection tests passed in 2.85 s; synthetic fixtures
are tests only and are never written as research outputs. Provenance now also captures
all installed dependency versions, actual source file hashes and Git dirty state.

**Phase 5 is not complete:** no corrected five-fold model comparison exists yet. The
isolated WSL GPU environment is still downloading CUDA dependencies. No metric values,
best-model ranking or improvement claims have been fabricated while waiting.

## Phase 5b - preserved legacy archive compatibility

`python -m kssd.audit_legacy --data data/kssd2025/data` completed successfully:
all 20 original `.keras` archives restore their extracted weights into the corresponding
current architecture, with matching weight shapes and finite values. SHA-256 hashes,
archive metadata and measured parameter counts are recorded in
`reports/legacy_archive_audit.json`. Original archives are unchanged.

This run checks weights only. An earlier optional full-resolution CPU prediction
passed for the first Modified U-Net and U-Net++ archives but became unusually slow
for U-Net3+ and was stopped. No full inference pass is claimed for all legacy models.
Inspection of U-Net3+ fold-1 weights found no nonfinite or subnormal weights; this
does not rule out numerical issues in intermediate activations. The separately
verified newly initialized model smoke checks remain valid. Legacy checkpoints
lack defensible nested selection provenance and cannot replace corrected training.

**Phase 5 remains incomplete:** CUDA dependencies are still downloading, corrected
five-fold experiments have not run, and no new performance ranking is available.

Full native-CPU suite recheck: `python -m pytest -q --basetemp tmp/pytest-phase5`
**38 passed** in 214.06 s, including synthetic training and checkpoint restoration.
934 upstream deprecation warnings remain. Archive CLI exits 0; `compileall` and
`git diff --check` pass.

Archive compatibility commit `83b2907` pushed and remote hash confirmed.

## Independent Phase 6a - mathematical loss implementation only

While Phase 5 GPU dependency downloads block experiments, independently verified
the Focal Tversky definition using the research paper and authors' released code.
Implemented foreground, per-image loss with FN weight 0.7, FP weight 0.3, exponent
0.75, smoothing 1e-6, and an explicit finite-gradient convention at perfect overlap.
Sampling/reduction differences from the original implementation are documented.
The Modified U-Net alternative config differs from its baseline only in loss.

`python -m pytest tests/test_losses.py -q --basetemp tmp/pytest-loss-final`:
**7 passed** in 42.43 s. Formula, reduction, FN weighting, perfect/empty and imperfect
gradients, serialization, config matching and a compiled Adam step are verified.
Two upstream Keras deprecation warnings. No real-data performance experiment or
size-stratified comparison is claimed; Phase 6 acceptance remains incomplete.
The dependent matched experiment waits for baseline prerequisites and measured runs.

Optional loss implementation commit `12fd797` pushed and remote hash confirmed.

## Independent documentation/configuration alignment

During the GPU download limitation, corrected stale README instructions and claims
independently of future experiment results: actual parameters, batch budget,
content-group/inner-selection protocol, setup commands, artifact policy, and explicit
unmeasured result gates. Removed unsupported general mixed-precision claims and
obsolete Kaggle runtime estimates. Legacy numbers remain clearly superseded.

All three notebooks now load committed 10-epoch CLI configs, including tested
augmentation; previously their default configuration unintentionally disabled it.
Notebook code cells parse successfully, outputs are empty, all referenced configs
exist with epochs 10/batch 2/augmentation enabled, and README local links resolve.
`git diff --check` passes. Documentation of actual final results and final-review
readiness still depends on incomplete experiments; Phase 8 is not fully complete.

Documentation/config alignment commit `e89b4f2` pushed and remote hash confirmed.

## Independent Phase 6b - matched comparison and size-analysis validator

Implemented comparison that requires complete, checkpoint-verified five-fold runs
whose configs differ only in loss. It verifies exact folds, dataset, initial weight
hashes, deterministic execution, computational source hashes and environment parity.
Per-fold area terciles use outer-training labels only; results include bin counts,
micro/macro scores, paired per-image Dice differences and paired fold differences
with sample standard deviation. Corrupted reported mask areas are now rejected by
the general aggregator too.

`python -m pytest tests/test_reporting.py tests/test_compare_losses.py -q --basetemp tmp/pytest-comparison-final`:
**11 passed** in 30.19 s. Tests verify no outer-label influence on bins, complete
sample accounting, known negative paired differences, rejection of changed configs,
initial weights, computational sources and stone areas. Fixtures are explicitly
synthetic and never emitted as scientific outputs. `git diff --check` passes.

CUDA installation is still downloading large dependencies. Full baseline and loss
experiments, measured size analysis and failure figures remain unrun. Neither
Phase 5 nor Phase 6 is claimed complete.

Matched-comparison implementation commit `e9766d2` pushed and remote hash confirmed.

## Independent Phase 7a - verified figure generation implementation

Implemented deterministic case selection, four-model mean/sample-SD comparison,
all-fold training/inner-selection curves and CT/GT/prediction/error/crop panels.
The generator requires complete matched aggregate results and original checkpoints.
Source pixel hashes are checked, original evaluation batch shape/order preserved,
and restored predictions must reproduce saved confusion counts before plotting.
Figure index records checkpoints, plotted values and selected cases; status stays
running if generation fails. No successful-case or clinical ambiguity label is
invented when measured case rankings do not support it.

`python -m pytest tests/test_figures.py -q --basetemp tmp/pytest-figures-final`:
**4 passed** in 24.19 s. Synthetic prediction, comparison and curve outputs were
visually inspected; labels, error colors and error bars are readable. Synthetic
images remain ignored temporary tests, not research outputs. `compileall` and
`git diff --check` pass. Measured baseline/loss figures and error interpretation
remain incomplete until actual five-fold runs finish.
