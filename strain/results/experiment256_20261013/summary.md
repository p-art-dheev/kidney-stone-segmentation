# Isolated 256 × 256 experiment

Elapsed: 466.46 seconds. Status: worker_exit_-15.

Held-out test; threshold 0.5; all metrics from the same validation-selected checkpoint.

| Model | Epochs | Dice | IoU | Precision | Recall | Status |
|---|---:|---:|---:|---:|---:|---|
| modified_unet | 21 | 0.0004 | 0.0002 | 0.0002 | 1.0000 | complete |
| unet3plus | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | complete |
| unetpp | 19 | 0.0005 | 0.0002 | 0.0002 | 1.0000 | complete |
| transunet | — | unavailable | unavailable | unavailable | unavailable | unavailable |

Tiny regions erased by nearest-neighbor resizing remain in all partitions. See preprocessing.json.
Content groups are not patient IDs. The small test sample cannot establish clinical generalization.
Dice/IoU are 1 for empty truth and prediction; precision/recall are 0 for zero denominators.
The existing image illustrates legacy scores; these resized test scores use a different split and resolution.
