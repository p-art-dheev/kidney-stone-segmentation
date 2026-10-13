# Interim selected validation results

Threshold 0.5; pooled pixel metrics; 20 additional epochs per completed trajectory. Held-out test results are pending.

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Modified U-Net | 0.8138 | 0.6860 | 0.8256 | 0.8023 |
| U-Net++ | 0.3823 | 0.2363 | 0.3690 | 0.3966 |
| TransUNet | 0.6365 | 0.4668 | 0.6527 | 0.6211 |

Best validation epochs: Modified U-Net 14, U-Net++ 14, TransUNet 20. U-Net3+ is still running. These figures are validation results, not final test results.
