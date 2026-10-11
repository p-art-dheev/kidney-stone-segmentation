# Repository audit and execution plan

Audit date: 2026-10-11 (Asia/Kolkata). Starting commit: `4f3e387`.

## Verified starting state

- Clean `main`, tracking `origin/main`; HTTPS origin matches the requested repository. Push dry run succeeded.
- Three notebooks implement four architectures. No Python package, tests, dependency manifest, dataset, saved split manifests, or machine-readable histories exist.
- Twenty final-epoch `.keras` archives and six distinct figures exist (the U-Net++/3+ figures are duplicated). Archives contain Keras 3.13.2 configuration, weights, and unregistered custom loss names. Existence is not evidence that they load or match reported best metrics.
- Every notebook uses soft Dice/IoU and hard precision/recall; reports independent maxima across epochs; saves the final epoch. Consequently the published table does not describe one binary prediction or one checkpoint.
- All pipelines silently intersect image/mask filenames, convert files to 8-bit grayscale, threshold masks at `>0`, load all images in RAM, use image-level KFold(seed=42), and omit the incomplete training batch. None seeds all model/shuffle randomness or augments data.
- Outer held-out folds are used for epoch selection/reporting. There is no inner selection split, dataset validation, duplicate grouping, patient grouping, or fold provenance.
- TransU-Net patches installed source with `exec` and extracts weights from archives. README incorrectly describes weights-only saving; actual notebooks save complete archives. A duplicate visualization cell remains.
- Example legacy figures have overlapping titles and visible faint mask speckles. Mask encoding requires inspection before selecting a binarization policy.
- Python 3.13.5; TensorFlow 2.20.0; Keras 3.13.2; NumPy 2.2.6; sklearn 1.7.2; Pillow 11.1.0. `keras-unet-collection` is missing. Native TensorFlow detects no GPUs.
- RTX 4070 Laptop GPU, 8188 MiB VRAM, NVIDIA driver 580.88. WSL Ubuntu-22.04 detects it, but Python 3.10.12 has no TensorFlow/NumPy. About 20 GiB disk free at audit time. Full-resolution training cost is not yet measured.
- No uncommitted user work or repository AGENTS.md was found. Existing assets will be preserved; no history rewriting.

## Reference-paper comparison

The supplied IEEE Access paper (DOI [10.1109/ACCESS.2025.3610027](https://doi.org/10.1109/ACCESS.2025.3610027)) was inspected. Pages 3-7 describe 838 axial slices, annotation, and the lightweight encoder `[16,32,64,128]` with a 256-filter bottleneck, BN, dropout, transposed-convolution decoder and sigmoid output. This broadly matches the custom notebook. Exact implementation equivalence of the library models is unverified.

The source dataset's identifying metadata was removed before image export (page 3); patient-level grouping must not be inferred from sequential filenames. Paper results are external evidence, not results of this repository. Longer training does not guarantee improvement. The training/augmentation table must be visually verified because PDF text extraction omits table cells.

Native Windows TensorFlow >2.10 does not provide CUDA GPU support; see [official installation documentation](https://www.tensorflow.org/install/pip). WSL is a possible GPU environment, pending isolated installation and a real device operation.

## Sequential plan and acceptance gates

| Phase | Prerequisites / acceptance | Compute and expected outputs |
|---|---|---|
| 0 audit | Code, artifacts, paper, Git and environment inspected | This permanent report; commit/push |
| 1 metrics | Exact binary confusion-count tests including empty masks, boundary and Dice/IoU identity; all notebooks use shared semantics; legacy evidence labelled | CPU seconds; shared package, tests, metric policy |
| 2 dataset | Obtain official files; validate all expected pairs, dimensions/values/hashes/areas; duplicate review and overlays; document grouping limits | CPU minutes, download size unknown; manifest/report/overlays |
| 3 training | Phase 2 verified; deterministic grouped outer/inner folds; fresh model/optimizer; no dropped samples; best inner checkpoint evaluated once on outer fold | CPU tests + model checks; runner, split manifests, histories |
| 4 augmentation/config | Paper table verified; paired transforms tested; short smoke then representative fold | GPU duration to measure; configs and resource log |
| 5 fair evaluation | Phases 1-4 accepted; same fixed protocol for all four architectures | 20 full-resolution runs, 150 epochs where feasible; actual histories/counts/fold results/checkpoints |
| 6 loss experiment | Reliable Focal Tversky definition; matched Modified U-Net baseline and alternative; size bins learned from outer training data | 5 additional matched runs; controlled comparison, no unsupported claims |
| 7 error analysis | Verified phase 5/6 outputs | CPU/inference; deterministic successes/failures, curves/tables |
| 8 documentation | Actual code and measured status reflected accurately | Reproduction commands, dependencies, artifact policy |
| 9 final audit | Every mandatory gate checked; every intended commit confirmed remote | Final report with explicit incomplete gates and commit ledger |

Commit/push after each independently completed task. Stop on push rejection/conflict/permission failure. Expensive runs follow measured smoke/representative runtime. A blocked phase stays incomplete; only independent tooling/documentation may proceed. Synthetic tests will never be reported as dataset results.
