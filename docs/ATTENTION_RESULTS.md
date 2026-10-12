# Attention U-Net: experiment in progress

Completed outer folds: **1/5**. Fold 1 is the retained pilot; no fold is retrained. Missing folds have no results and no five-fold mean is claimed.

| Fold | Selected epoch | Dice | IoU | Precision | Recall | Training seconds | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 9 | 0.474540 | 0.311080 | 0.344465 | 0.762456 | 378.20 | `25830fd50270bd4bd7df11db75464eec05f68602dbebf17f36daa046bed2b624` |

Each completed fold passed finite history, earliest inner maximum, count/mask and checkpoint identity checks. All derived prediction PNGs are saved in its records folder; new H5 weights are local/ignored with verified Windows copies.

[Protocol](ATTENTION_UNET.md) · [Saved records](../reports/experiments/attention10fp32_20261012/) · [Manifest](../reports/experiment_manifest.json)

Matched Modified U-Net five-fold Dice is 0.7910 ± 0.0724. A single Attention fold is not a fair estimate of its five-fold mean. All failed/poor folds will remain reported; no outer result chooses a new epoch or training setting.
