# Verified experimental results

Generated from canonical JSON; source map: report tables and reports/experiments.

## A. Dataset statistics

| Item | Value |
|---|---|
| Source image-mask pairs | 838 |
| Eligible unique images | 831 |
| Excluded exact duplicates | 7 |
| Image / mask shape | 512 x 512 |
| Formats | TIFF / uint8 grayscale |
| Labels | 0 background; 1 stone |
| Empty masks | 0 |
| Masks with area 1-10 pixels | 50 |
| Area min / Q1 / median / Q3 / max | 3 / 38.25 / 101.5 / 231 / 755 |
| Content groups / largest original cluster | 39 / 173 |
| Normalization / mask rule | uint8 / 255; decoded mask >127 |


### Frozen split sizes

| Fold | Seed | Train | Inner | Outer |
|---|---|---|---|---|
| 1 | 43 | 446 | 206 | 179 |
| 2 | 44 | 455 | 97 | 279 |
| 3 | 45 | 665 | 48 | 118 |
| 4 | 46 | 589 | 70 | 172 |
| 5 | 47 | 606 | 142 | 83 |


## B. Training configurations

| Architecture | Epochs | Batch | Precision | Optimizer | LR | Loss |
|---|---|---|---|---|---|---|
| Modified U-Net | 10 | 2 | float32 | Adam | 0.001 | Dice |
| U-Net++ | 10 | 2 | float32 | Adam | 0.001 | Dice |
| U-Net3+ | 10 | 2 | float32 | Adam | 0.001 | Dice |
| TransUNet | 10 | 2 | float32 | Adam | 0.001 | Dice |
| Attention U-Net | 10 | 2 | float32 | Adam | 0.001 | Dice |


Total workload: 250 training epochs across 25 fold runs. All: base seed 42, fold seeds 43-47, train-only paired augmentation, hard threshold >=0.5, earliest maximum inner hard Dice checkpoint. RTX 4070 Laptop; TF 2.20 / Keras 3.13.2; full lock in reports/resources/wsl_gpu.lock.txt. Training protocols differ from the source paper; see report settings section.

## C. Base paper versus local replication attempt

| Architecture | Paper Dice | Local Dice | Paper IoU | Local IoU |
|---|---|---|---|---|
| Modified U-Net | 0.9706 ± 0.0039 | 0.7910 ± 0.0724 | 0.9465 ± 0.0062 | 0.6589 ± 0.0973 |
| U-Net++ | 0.9663 ± 0.0063 | 0.3678 ± 0.2567 | 0.9397 ± 0.0083 | 0.2490 ± 0.1894 |
| U-Net3+ | 0.9668 ± 0.0061 | 0.3230 ± 0.4535 | 0.9402 ± 0.0086 | 0.2805 ± 0.4098 |
| TransUNet | 0.9553 ± 0.0089 | 0.4842 ± 0.3380 | 0.9221 ± 0.0127 | 0.3749 ± 0.3166 |


Results-only comparison: paper Table 3 (PDF p.9) versus canonical local records. Literature and local protocols differ; training settings are documented separately.

## D. Model comparison

| Architecture | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| U-Net++ | 0.3678 ± 0.2567 | 0.2490 ± 0.1894 | 0.3763 ± 0.2169 | 0.4317 ± 0.3622 |
| U-Net3+ | 0.3230 ± 0.4535 | 0.2805 ± 0.4098 | 0.3561 ± 0.4892 | 0.3014 ± 0.4364 |
| TransUNet | 0.4842 ± 0.3380 | 0.3749 ± 0.3166 | 0.5623 ± 0.3990 | 0.5063 ± 0.3042 |
| Attention U-Net | 0.7585 ± 0.2205 | 0.6515 ± 0.2864 | 0.7890 ± 0.2621 | 0.7803 ± 0.1996 |


Mean ± sample SD of five fold-pooled metrics, not confidence intervals or per-image averages.

| Architecture | Fold | Selected epoch | Dice | IoU | Precision | Recall | Training s |
|---|---|---|---|---|---|---|---|
| Modified U-Net | 1 | 9 | 0.768234 | 0.623684 | 0.843667 | 0.705182 | 380.16 |
| Modified U-Net | 2 | 10 | 0.681916 | 0.517355 | 0.562056 | 0.866755 | 373.30 |
| Modified U-Net | 3 | 10 | 0.809669 | 0.680205 | 0.833461 | 0.787198 | 530.30 |
| Modified U-Net | 4 | 8 | 0.817730 | 0.691661 | 0.985547 | 0.698749 | 476.24 |
| Modified U-Net | 5 | 6 | 0.877459 | 0.781673 | 0.924307 | 0.835131 | 498.65 |
| U-Net++ | 1 | 8 | 0.490774 | 0.325183 | 0.434388 | 0.563983 | 429.14 |
| U-Net++ | 2 | 8 | 0.227154 | 0.128129 | 0.527317 | 0.144755 | 434.62 |
| U-Net++ | 3 | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 588.05 |
| U-Net++ | 4 | 10 | 0.462555 | 0.300859 | 0.403234 | 0.542340 | 514.81 |
| U-Net++ | 5 | 10 | 0.658314 | 0.490662 | 0.516556 | 0.907304 | 559.20 |
| U-Net3+ | 1 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1634.67 |
| U-Net3+ | 2 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1584.34 |
| U-Net3+ | 3 | 10 | 0.665653 | 0.498861 | 0.836283 | 0.552853 | 2193.52 |
| U-Net3+ | 4 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2033.89 |
| U-Net3+ | 5 | 10 | 0.949264 | 0.903428 | 0.944400 | 0.954179 | 2163.79 |
| TransUNet | 1 | 9 | 0.439656 | 0.281768 | 0.327510 | 0.668594 | 478.95 |
| TransUNet | 2 | 8 | 0.021893 | 0.011068 | 0.011746 | 0.160822 | 500.88 |
| TransUNet | 3 | 10 | 0.355818 | 0.216410 | 0.604973 | 0.252023 | 735.46 |
| TransUNet | 4 | 10 | 0.694491 | 0.531970 | 0.953531 | 0.546128 | 718.81 |
| TransUNet | 5 | 9 | 0.908983 | 0.833152 | 0.913940 | 0.904080 | 678.45 |
| Attention U-Net | 1 | 9 | 0.474540 | 0.311080 | 0.344465 | 0.762456 | 378.20 |
| Attention U-Net | 2 | 6 | 0.796644 | 0.662019 | 0.812737 | 0.781177 | 411.91 |
| Attention U-Net | 3 | 7 | 0.942443 | 0.891150 | 0.988570 | 0.900428 | 586.19 |
| Attention U-Net | 4 | 5 | 0.593772 | 0.422244 | 0.821501 | 0.464897 | 527.52 |
| Attention U-Net | 5 | 10 | 0.985292 | 0.971011 | 0.977950 | 0.992746 | 554.67 |


### Parameters and runtime

| Architecture | Parameters |
|---|---|
| Modified U-Net | 1945233 |
| U-Net++ | 1967713 |
| U-Net3+ | 2374583 |
| TransUNet | 4747553 |
| Attention U-Net | 1967357 |

| Architecture | Training s | Inference ms/image range |
|---|---|---|
| Modified U-Net | 2258.64 | 31.042--42.107 |
| U-Net++ | 2525.83 | 37.658--45.097 |
| U-Net3+ | 9610.21 | 132.012--159.717 |
| TransUNet | 3112.55 | 61.897--72.460 |
| Attention U-Net | 2458.49 | 41.230--48.245 |


## E. Baseline versus modification

| Comparison | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| Attention U-Net | 0.7585 ± 0.2205 | 0.6515 ± 0.2864 | 0.7890 ± 0.2621 | 0.7803 ± 0.1996 |
| Attention minus baseline | -0.032463 | -0.007415 | -0.040763 | +0.001737 |


Modification: four additive spatial skip gates, 22,124 additional parameters. Config differs only in architecture; same frozen splits/seeds/budget/selection. Negative mean result; no improvement claim, no conditional improvement ablation. Focal Tversky and 150-epoch scores: NOT YET MEASURED.
