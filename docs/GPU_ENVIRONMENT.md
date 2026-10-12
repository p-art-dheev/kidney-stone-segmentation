# Verified GPU environment

TensorFlow 2.20.0 executed an actual matrix multiplication on GPU:0 with CPU
fallback disabled. All four architectures subsequently passed one augmented
512x512 training batch and weight-restoration checks, using batch size 2.
These checks establish functionality, not segmentation performance.

Measured environment: WSL distribution `Ubuntu`, Python 3.12.3, NVIDIA GeForce
RTX 4070 Laptop GPU (8188 MiB physical memory), driver 580.88, Keras 3.13.2.
TensorFlow reports a CUDA 12.5.1 build with cuDNN 9; the installed CUDA library
versions are recorded in the dependency lock. Actual cuDNN loaded was 9.27.0.
The TensorFlow device log reported 5518 MiB available to its allocator.

Evidence: `reports/resources/wsl_gpu_environment.json`, `gpu_smoke.json` and
`wsl_gpu.lock.txt`. The runtime's `pip check` found no broken requirements.
The complete suite passed **59 tests** in 145.75 s, with 812 upstream deprecation
warnings. TensorFlow logged a nonfatal layout-optimizer warning for Modified
U-Net; finite training/prediction and weight restoration passed despite it.

## Installation and reproduction

Use a CUDA-capable Linux/WSL environment with Python 3.12. Start from a checkout
of this repository and obtain the dataset as described in README.md. In Bash:

```sh
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r reports/resources/wsl_gpu.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
export PYTHONHASHSEED=42
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
python -m pytest -q --basetemp tmp/pytest-reproduce
python -m kssd.smoke --data data/kssd2025/data --output runs/smoke-reproduce --require-gpu
python -m kssd.train --data data/kssd2025/data --config configs/representative.json --output runs/pilot-reproduce --fold 1 --selection-only --require-gpu
```

The workstation used an isolated persistent runtime at
`/var/tmp/kssd-gpu312-20261012` and a clean native WSL sparse checkout at
`/var/tmp/kssd-workspace-20261012`, initially commit `7201932`. The checkout
shares Git objects with the original Windows repository, which must remain
available; original assets were preserved. It includes code/configs/reports/docs/
tests and root files. Dataset files were copied into native storage; training
verifies all decoded image/mask hashes against the frozen manifest.

Earlier `/tmp` runtime/cache files disappeared between WSL invocations. The
replacement `/var/tmp` runtime survived repeated invocations and the subsequent
app restart. The durable ignored Windows wheel cache is `tmp/gpu-wheels/`.
66 official PyPI wheels passed SHA-256 checks before offline installation;
MarkupSafe was added when dependency resolution identified it as missing.
There were no system-wide Python or driver changes.

Training outputs and selected weights remain in the native checkout's ignored
`runs/` directory. Compact measured records are copied into the original
repository for commits and review. Keep those weights to reproduce predictions;
they are not included in Git. Existing run folders are rejected rather than
overwritten. GPU visibility alone does not establish completed CV results.
