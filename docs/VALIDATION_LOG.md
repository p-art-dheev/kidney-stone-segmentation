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
