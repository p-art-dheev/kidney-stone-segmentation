# Isolated 256 × 256 experiment

Elapsed: 1748.75 seconds. Status: finished.

Held-out test; threshold 0.5; all metrics from the same validation-selected checkpoint.

| Model | Epochs | Dice | IoU | Precision | Recall | Status |
|---|---:|---:|---:|---:|---:|---|
| modified_unet | 79 | 0.8364 | 0.7189 | 0.8565 | 0.8174 | complete |
| unet3plus | 117 | 0.7871 | 0.6490 | 0.8595 | 0.7260 | complete |
| unetpp | 80 | 0.2632 | 0.1516 | 0.1597 | 0.7489 | complete |
| transunet | 96 | 0.3236 | 0.1930 | 0.2078 | 0.7306 | complete |

Tiny regions erased by nearest-neighbor resizing remain in all partitions. See preprocessing.json.
Content groups are not patient IDs. The small test sample cannot establish clinical generalization.
Dice/IoU are 1 for empty truth and prediction; precision/recall are 0 for zero denominators.
The existing image illustrates legacy scores; these resized test scores use a different split and resolution.
