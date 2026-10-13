# Attention U-Net — completed 50-epoch run

Actual wall-clock time: 16.49 minutes. Fit and epoch validation: 438.66 seconds.

All 50 complete epochs used every training image. Selected validation epoch: 43. Fresh random initialization; precision: mixed_float16; batch: 16.

Primary held-out test metrics at threshold 0.5, pooled pixel counts:

| Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|
| 0.8477 | 0.7356 | 0.8992 | 0.8018 |

Validation and optional validation-calibrated test results are saved separately in attention_unet/result.json. Both test thresholds share one prediction pass. All choices were sealed before test loading.

Frozen split: 453 training / 288 validation / 90 test images, all 831 eligible unique samples. Seven existing duplicate exclusions are retained; 100% of the training partition is fitted. Content-group separation does not prove patient independence without patient IDs.

Preprocessing reuses verified 256-pixel bilinear CT images, uint8 /255 normalization and nearest-neighbor binary masks. No entire mask vanished, but small components disappeared in 50 original images; IDs remain in preprocessing.json.

Best weights: `strain/checkpoints/attention256_50epochs_20261014_v2/attention_unet.best.weights.h5`. Latest full model with optimizer: `strain/checkpoints/attention256_50epochs_20261014_v2/attention_unet.latest.keras`. Training/validation loss and all four metrics are saved to CSV/JSON with curves. Pilot updates were discarded.

Historical files and checkpoints passed their hash checks. Independent verification is recorded in independent_verification.json. No competing project experiment was running before launch; no unrelated process was stopped.

Rerun from repository root:

```bash
python -m strain.attention.run --data /var/tmp/kssd-workspace-20261012/data/kssd2025/data --output strain/results/attention256_new_run
```
