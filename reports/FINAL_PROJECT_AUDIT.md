# Final project audit — Kidney Stone Segmentation

## Release state

This final non-training completion pass verifies the saved 20-fold float32 baseline, publishes compact exploratory/partial evidence, completes qualitative review and reproducibility documentation, and adds an independent saved-record auditor. No new large training was started. The matched Focal Tversky CV remains pending because its existing run stopped before its first completed epoch and has no selected checkpoint. The baseline release is reviewable; the loss experiment is **not complete**.

The final-pass starting branch was `main` at `64227ac`, equal to `origin/main` after fetch; there were no stashes. Existing untracked exploratory reports and a Focal smoke report were preserved and included intentionally. Earlier completed source/audit tasks are retained in Git history. No user work was reset or removed. Git transport push authentication works; GitHub CLI API authentication is not configured. No existing LFS tracking or GitHub release workflow was found, so a new binary-checkpoint release was not invented.

## Objective and frozen protocol

Segment binary kidney-stone foreground in 838 exported 512×512 CT TIFF/mask pairs. Decode palette masks to grayscale and threshold >127; images are uint8 /255, with no resizing. Seven exact duplicate copies are excluded, leaving 831 eligible images; nonempty foreground is often tiny (50 masks have ≤10 foreground pixels). Content similarity clustering produces 39 groups, not verified patient identities. GroupKFold supplies five unequal outer folds; GroupShuffleSplit chooses the inner selection subset within each outer training partition. Seed 42 and per-fold seeds 43–47 are recorded.

Modified U-Net uses encoder filters 16/32/64/128/256 with skip decoding; U-Net++ uses nested skips; U-Net3+ uses full-scale skips (22 skip channels, aggregate 110); TransUNet combines the encoder/decoder with embedding 192, four heads and three transformer blocks. Parameters: 1,945,233 / 1,967,713 / 2,374,583 / 4,747,553 respectively. The local TransUNet Keras Reshape compatibility adaptation is documented and is not claimed to reproduce every original-author implementation detail.

Final common budget: ten epochs, batch two including remainder, float32, Adam LR 0.001, per-image differentiable Dice loss with smoothing 1e-6, hard threshold ≥0.5. Variables, output probabilities and losses are float32. Each fold starts with fresh seeded weights and optimizer; the maximum inner-selection hard Dice chooses the checkpoint with earliest ties. Outer outcomes never choose epochs, thresholds or hyperparameters. No outer-test augmentation or test-time augmentation is used.

Augmentation is train-only: paired affine rotation uniform ±2.5°, independent translations ±0.75% of image dimensions, zoom [0.9925,1.0075], horizontal flip probability 0.5; image-only Gaussian noise with SD uniform [0,0.0075], then clipping. Bilinear images and nearest-neighbor masks use the same transform. Exact distributions are our documented interpretation of paper details, not an exact paper reproduction. See [augmentation](../docs/AUGMENTATION.md).

Dataset fingerprint: `5fec0d38d1fd0386277a056cb1579235121f65a5da4fc3ac71abda8218db621f`. See [dataset manifest](dataset/manifest.json), [splits](splits.json), [experiment manifest](experiment_manifest.json) and [machine-readable verification](final_validation.json).

## Hardware and software

Verified runtime: Python 3.12.3, TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, keras-unet-collection 0.1.13; WSL Ubuntu with RTX 4070 Laptop GPU (8 GB physical, TensorFlow exposed approximately 5.5 GiB). CUDA/cuDNN details and dependency lock are in [GPU environment](../docs/GPU_ENVIRONMENT.md). Complete package versions and computational source hashes are saved per fold. The launcher sets PYTHONHASHSEED=42 and TensorFlow intra/inter-op threads to 2; deterministic ops are enabled. Native Windows TF 2.20 was used for CPU functionality gates, not CUDA training.

## Experiment status

The inventory below derives from saved files and process checks. A stale provenance status of `running` without a live process is classified as interrupted/failed, never active. Fold counts mean completed **outer evaluations**; selection-only pilots and optimizer smoke checks are not CV results.

| Run family / architecture | Loss | Precision evidence | Complete outer folds | Epoch evidence | Checkpoints / metrics | Included final baseline? |
|---|---|---|---:|---|---|---|
| baseline / modified_unet | dice | float32 configured; older runtime field absent | 2/5 | F1:10, F2:10, F3:7 (budget 10) | partial/interrupted/failed; 3 checkpoint files; 2 outer metric records | No |
| baseline_amp / modified_unet | dice | float32 | 1/5 | F1:10, F2:3 (budget 10) | partial/interrupted/failed; 2 checkpoint files; 1 outer metric records | No |
| baseline_amp_verified / modified_unet | dice | mixed_float16 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | No |
| baseline_amp_verified / transunet | dice | mixed_float16 | 0/5 | F1:4 (budget 10) | partial/interrupted/failed; 1 checkpoint files; 0 outer metric records | No |
| baseline_amp_verified / unet3plus | dice | mixed_float16 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | No |
| baseline_amp_verified / unetpp | dice | mixed_float16 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | No |
| baseline_fp32_final / modified_unet | dice | float32 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | Yes; independently verified |
| baseline_fp32_final / transunet | dice | float32 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | Yes; independently verified |
| baseline_fp32_final / unet3plus | dice | float32 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | Yes; independently verified |
| baseline_fp32_final / unetpp | dice | float32 | 5/5 | F1:10, F2:10, F3:10, F4:10, F5:10 (budget 10) | complete; 5 checkpoint files; 5 outer metric records | Yes; independently verified |
| focal_fp32_final / modified_unet | focal_tversky | float32 | 0/5 | F1:0 (budget 10) | partial/interrupted/failed; 0 checkpoint files; 0 outer metric records | No |
| gpu-amp2 optimizer checks | dice | modified_unet: B2, 5 updates, precision mixed_float16; unetpp: B2, 5 updates, precision mixed_float16; unet3plus: B2, 5 updates, precision mixed_float16; transunet: B2, 5 updates, precision mixed_float16 | N/A | No epochs/CV | Complete requested smoke set | No |
| gpu-amp4 optimizer checks | dice | modified_unet: B4, 5 updates, precision mixed_float16; unetpp: B4, 5 updates, precision mixed_float16 | N/A | No epochs/CV | Partial smoke set; memory failure where recorded | No |
| gpu-benchmark2 optimizer checks | dice | modified_unet: B2, 5 updates, precision older field absent; unetpp: B2, 5 updates, precision older field absent; unet3plus: B2, 5 updates, precision older field absent; transunet: B2, 5 updates, precision older field absent | N/A | No epochs/CV | Complete requested smoke set | No |
| gpu-benchmark8 optimizer checks | dice | modified_unet: B8, 5 updates, precision older field absent; unetpp: B8, 5 updates, precision older field absent | N/A | No epochs/CV | Partial smoke set; memory failure where recorded | No |
| gpu-focal-fp32-final-smoke optimizer checks | focal_tversky | modified_unet: B2, 1 updates, precision float32 | N/A | No epochs/CV | Complete requested smoke set | No |
| gpu-representative / modified_unet | dice | float32 configured; older runtime field absent | 0/5 | F1:1 (budget 1) | selection-only; 1 checkpoint files; 0 outer metric records | No |
| gpu-smoke optimizer checks | dice | modified_unet: B2, 1 updates, precision older field absent; unetpp: B2, 1 updates, precision older field absent; unet3plus: B2, 1 updates, precision older field absent; transunet: B2, 1 updates, precision older field absent | N/A | No epochs/CV | Complete requested smoke set | No |
| transunet_fp32_diagnostic / transunet | dice | float32 | 0/5 | F1:4 (budget 4) | selection-only; 1 checkpoint files; 0 outer metric records | No |
| Paper-budget configs / all four | Dice | float32 configured | 0/5 each | 150 planned, none recorded; B16 | Unrun; no selected checkpoints | No |
| CPU/GPU representative pilots and CPU Focal checks | Dice / Focal | float32 checks | N/A | One selection-only epoch or optimizer step | Functionality evidence in reports/resources; no outer CV | No |
| Legacy notebook archives / all four | Historical Dice | Historical | Not validated under corrected protocol | Original notebook protocol | 20 finite-weight restoration checks; historical metrics unverified | No |

All 15 native output families, individual saved states and locations are preserved in [experiment inventory](experiment_inventory.json). Compact interrupted snapshots are under [local run snapshots](experiments/local_run_snapshots); complete mixed results are under [exploratory mixed results](experiments/baseline10amp_verified). GPU batch-eight float32 and batch-four mixed checks ran out of memory in U-Net3+. The first purported mixed baseline reset the policy to float32 during Keras session clearing; it was stopped and preserved. The corrected mixed baseline completed three models, but TransUNet validation loss became NaN at epoch 3 and training loss at epoch 4. The strict serializer rejected NaN rather than publishing misleading metrics. Float32 four-epoch diagnostic passed; all final folds were restarted in float32. See [failure trace](resources/transunet_amp_failure.txt) and [acceleration evidence](../docs/GPU_ACCELERATION.md).

### Exploratory mixed-precision results — excluded from final comparison

All fifteen completed mixed-precision folds were independently rechecked against
counts, histories, actual policy/optimizer and the locally backed-up checkpoint
hashes. Their configurations match each other except architecture. TransUNet has
no completed fold, so this is not a complete four-model comparison. No result is
substituted into the canonical float32 baseline.

| Architecture | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Modified U-Net | 0.8482 ± 0.0931 | 0.7452 ± 0.1371 | 0.9071 ± 0.0752 | 0.8083 ± 0.1506 |
| U-Net++ | 0.5307 ± 0.3409 | 0.4174 ± 0.3060 | 0.4741 ± 0.3283 | 0.6604 ± 0.3629 |
| U-Net3+ | 0.0677 ± 0.1513 | 0.0407 ± 0.0910 | 0.1788 ± 0.3999 | 0.0417 ± 0.0933 |
| TransUNet | Unavailable: NaN failure | Unavailable | Unavailable | Unavailable |

Mean ± sample SD across five outer folds. Exact values/checksums are in
[release artifact checks](release_artifact_checks.json); full per-fold records,
epochs and provenance are in [exploratory records](experiments/baseline10amp_verified).
Partial earlier runs retain available evaluation/history/provenance snapshots
without a five-fold aggregate; missing folds are not filled or averaged away.

## Verified canonical baseline results

Mean ± sample standard deviation across five fold-level pooled-pixel scores; the sample SD uses ddof=1 and is not a confidence interval.

| Architecture | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| modified_unet | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| unetpp | 0.3678 ± 0.2567 | 0.2490 ± 0.1894 | 0.3763 ± 0.2169 | 0.4317 ± 0.3622 |
| unet3plus | 0.3230 ± 0.4535 | 0.2805 ± 0.4098 | 0.3561 ± 0.4892 | 0.3014 ± 0.4364 |
| transunet | 0.4842 ± 0.3380 | 0.3749 ± 0.3166 | 0.5623 ± 0.3990 | 0.5063 ± 0.3042 |

Modified U-Net has the highest measured mean Dice in this fixed short-budget experiment. This does not establish statistical significance, clinical readiness, patient-independent generalization or superiority under a longer/tuned budget. Zero/poor folds remain included.

### Per-fold selected-checkpoint metrics

| Architecture | Fold | Train / inner / outer | Selected epoch | Dice | IoU | Precision | Recall | Training seconds |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| modified_unet | 1 | 446 / 206 / 179 | 9 | 0.768234 | 0.623684 | 0.843667 | 0.705182 | 380.16 |
| modified_unet | 2 | 455 / 97 / 279 | 10 | 0.681916 | 0.517355 | 0.562056 | 0.866755 | 373.30 |
| modified_unet | 3 | 665 / 48 / 118 | 10 | 0.809669 | 0.680205 | 0.833461 | 0.787198 | 530.30 |
| modified_unet | 4 | 589 / 70 / 172 | 8 | 0.817730 | 0.691661 | 0.985547 | 0.698749 | 476.24 |
| modified_unet | 5 | 606 / 142 / 83 | 6 | 0.877459 | 0.781673 | 0.924307 | 0.835131 | 498.65 |
| unetpp | 1 | 446 / 206 / 179 | 8 | 0.490774 | 0.325183 | 0.434388 | 0.563983 | 429.14 |
| unetpp | 2 | 455 / 97 / 279 | 8 | 0.227154 | 0.128129 | 0.527317 | 0.144755 | 434.62 |
| unetpp | 3 | 665 / 48 / 118 | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 588.05 |
| unetpp | 4 | 589 / 70 / 172 | 10 | 0.462555 | 0.300859 | 0.403234 | 0.542340 | 514.81 |
| unetpp | 5 | 606 / 142 / 83 | 10 | 0.658314 | 0.490662 | 0.516556 | 0.907304 | 559.20 |
| unet3plus | 1 | 446 / 206 / 179 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1634.67 |
| unet3plus | 2 | 455 / 97 / 279 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1584.34 |
| unet3plus | 3 | 665 / 48 / 118 | 10 | 0.665653 | 0.498861 | 0.836283 | 0.552853 | 2193.52 |
| unet3plus | 4 | 589 / 70 / 172 | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2033.89 |
| unet3plus | 5 | 606 / 142 / 83 | 10 | 0.949264 | 0.903428 | 0.944400 | 0.954179 | 2163.79 |
| transunet | 1 | 446 / 206 / 179 | 9 | 0.439656 | 0.281768 | 0.327510 | 0.668594 | 478.95 |
| transunet | 2 | 455 / 97 / 279 | 8 | 0.021893 | 0.011068 | 0.011746 | 0.160822 | 500.88 |
| transunet | 3 | 665 / 48 / 118 | 10 | 0.355818 | 0.216410 | 0.604973 | 0.252023 | 735.46 |
| transunet | 4 | 589 / 70 / 172 | 10 | 0.694491 | 0.531970 | 0.953531 | 0.546128 | 718.81 |
| transunet | 5 | 606 / 142 / 83 | 9 | 0.908983 | 0.833152 | 0.913940 | 0.904080 | 678.45 |

### Checkpoint identities and confusion counts

| Architecture / fold | TP / FP / FN / TN | Selected weights SHA-256 |
|---|---|---|
| modified_unet / 1 | 19023 / 3525 / 7953 / 46893275 | `ba71a4bda76bb74fc1eb55266d5cc17760f4c2d60ac3e216b1e2b2a357a0588b` |
| modified_unet / 2 | 40513 / 31567 / 6228 / 73059868 | `8ea7f38f299ea66ac8f0d0cbe414b0520a6913a9f7248a9bef88043551370c21` |
| modified_unet / 3 | 15274 / 3052 / 4129 / 30910537 | `7a2ed08a2ee645a14f4db9e913365a8251dd93f29d805dc624744754e3086ccd` |
| modified_unet / 4 | 20662 / 303 / 8908 / 45058895 | `2f2255dd6e30148bbbc7304499999cb85f1c3680f0aa202dc5555cbe06838161` |
| modified_unet / 5 | 13469 / 1103 / 2659 / 21740721 | `4ace29fbf198b4ebde7ca17aa267eb756ad9f133933cc5c3d87b458f1f685e75` |
| unetpp / 1 | 15214 / 19810 / 11762 / 46876990 | `46ae90a3ada5d07ae323310a3682ce4982dd1d75f8e8badb16ee4bc1fc93cb98` |
| unetpp / 2 | 6766 / 6065 / 39975 / 73085370 | `50fecd0f8786a40f2300a1833ac98330b3b8ac2b624f03a6167f25eb65d14523` |
| unetpp / 3 | 0 / 38097 / 19403 / 30875492 | `e74bec22afdda2411b7a226430cae16a961abf8920e8a1294490150754a9f1e5` |
| unetpp / 4 | 16037 / 23734 / 13533 / 45035464 | `7b85f885680181cdf9b846870ed4a6dbdd8134ffd6a1bd5c337f0cbc440b7aac` |
| unetpp / 5 | 14633 / 13695 / 1495 / 21728129 | `387d8cec9fbf5253ad35ef327bf5b3b234e317852eb6bd8468e7f13a837d12a9` |
| unet3plus / 1 | 0 / 0 / 26976 / 46896800 | `de6e6ca1d3c0e7cfb8365fea9a291796718442d29a40db2e5a635669bc5a08ad` |
| unet3plus / 2 | 0 / 0 / 46741 / 73091435 | `4b78b9f9177b83977cdacbc60947b9e539646e07544788ad8a9ee1cc5c96b62d` |
| unet3plus / 3 | 10727 / 2100 / 8676 / 30911489 | `dc367106c650c53a117d9dc99c4f6b3853dfe54f61b0012c340bfb5d738e6dd8` |
| unet3plus / 4 | 0 / 0 / 29570 / 45059198 | `7a86fd992ecfc1d8f1008418ccdf564cd87298ccc80edc67d622b1974736d506` |
| unet3plus / 5 | 15389 / 906 / 739 / 21740918 | `3cf64aa39d5530a6ee1668d6abfb23a0f87e63aab554a467f52fe70d7209f4a0` |
| transunet / 1 | 18036 / 37034 / 8940 / 46859766 | `764aae423b55bacd6740cb0f3e8efd18994aa9cb985f10587fc1d6ef8091517c` |
| transunet / 2 | 7517 / 632447 / 39224 / 72458988 | `4c4b4efd19b3c5e090145017a011ddc54b87eb43139849a5ba93c216795f9a92` |
| transunet / 3 | 4890 / 3193 / 14513 / 30910396 | `3227adddd2ca75c4c3deb2b25363a1df325dd5aa57a1647f797511c8d14b2a84` |
| transunet / 4 | 16149 / 787 / 13421 / 45058411 | `d23804d7745c37a261298994df16c4295a275a0b7f4347b7781749e3bf20fb19` |
| transunet / 5 | 14581 / 1373 / 1547 / 21740451 | `fe1cff0b14eb1e2d6c646a5323885ce7e6f2a5eac512239198cfdbf90e5e1369` |

Full per-image rows, macro scores, parameter counts, selected epoch, latency convention and initialization/source hashes remain in the canonical [baseline records](experiments/baseline10fp32_final). The independent verifier recomputes saved aggregates and checks every expected ID, mask area, integer TP/FP/FN/TN, total pixels, history length/finiteness, earliest inner maximum, actual float32/Adam, deterministic initialization evidence and optimizer update budget. All 20 local selected checkpoint hashes match. All four configurations differ only in architecture. Every architecture covers all 831 eligible outer IDs exactly once.

Metric definitions: Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN); precision=TP/(TP+FP); recall=TP/(TP+FN). Both-empty Dice/IoU are 1; undefined precision/recall are 0. Training loss is a soft per-image Dice surrogate, not the reported binary metric. Macro per-image and pooled-pixel summaries are saved separately. Pixel totals per architecture are 831×512×512=217,841,664; test fold sizes are 179/279/118/172/83. Checkpoint-selection data is disjoint from outer test by recorded content group.

## Qualitative review and unresolved learning failures

All 29 baseline plots were reviewed: comparison, four all-fold curves and 24 deterministic prediction panels. The original checkpoint regeneration verified case counts using original evaluation batch shape/order; the final auditor verifies case selection, values and all 30 file hashes including index.json. See [figure index](figures/baseline10fp32_final/index.json) and [comparison](figures/baseline10fp32_final/comparison.png). Colors: yellow TP, red missed foreground, cyan FP; cropped panels focus on the largest ground-truth component.

- Modified U-Net detects the 3-pixel minimum-area case with TP/FP/FN 3/1/0, Dice 0.8571, but misses much of 302.tif: 41/20/622, Dice 0.1133. Its best mean does not eliminate failures.
- U-Net++ misses 307.tif entirely (TP/FP/FN 0/0/755), and its largest-FP panel includes spurious foreground far from the stone. Its zero-Dice fold is retained.
- U-Net3+ has three zero-Dice outer folds, while folds 3 and 5 recover. The missing 324.tif foreground has TP/FP/FN 0/0/741. A selected success reaches Dice 1.0 on six foreground pixels; this is not representative of all cases.
- TransUNet 688.tif has TP/FP/FN 14/10052/0: recall is 1 but Dice is 0.0028 because false positives dominate. The panel shows widespread predicted regions; high recall alone is misleading.

Observed facts are complete histories, zero-Dice folds, pixel counts and displayed masks. Tiny foreground, short optimization budget, initialization sensitivity and activation/gradient behavior are plausible contributors, not established causes. The counts and restored figures agree, so these failures cannot be dismissed as reporting errors. Their precise training mechanism remains unresolved and would require new controlled experiments, which this pass does not start.

## Matched Focal Tversky status

FN weight 0.7, FP weight 0.3, exponent 0.75, smoothing 1e-6; FP32 loss, per-image reduction then batch mean, perfect-mask finite-gradient guard. The same Modified U-Net config changes only loss; formula/source distinctions are in [loss documentation](../docs/FOCAL_TVERSKY.md). CPU numerical and full-size smoke checks passed, and the actual GPU Focal smoke/checkpoint round trip passed in 9.98 seconds.

The real five-fold run was launched before these final instructions, but is no longer active. Its fold-1 provenance exists; history.csv is empty, no history.json/evaluation.json/best.weights.h5 exists, and its log ends during the first epoch. There are **0 completed epochs and 0 complete outer folds**. The cause of interruption is not established by the available log. A smoke checkpoint after one update is not a valid selected fold checkpoint and cannot substitute. Consequently comparison, training-derived stone-area bins and Focal qualitative outcomes cannot be produced from saved artifacts. A new five-fold training run is required; it is not authorized by this final-pass instruction and was not started.

## Artifact publication and reproducibility

Code, configs, dataset/split manifests, complete baseline metrics/provenance/histories, exploratory/partial compact records, verification reports, representative figures and compact logs are appropriate Git artifacts. Raw TIFF data, caches, environments, temporary helpers and large new H5 checkpoint collections are ignored. The original 20 legacy .keras assets remain untouched. They are not replacements for the corrected selected checkpoints.

The new baseline checkpoints total approximately 639 MiB locally. No existing LFS tracking or suitable release workflow is configured. These new weights are **not published on GitHub**. Local WSL and Windows backup locations and each SHA-256 are in [experiment manifest](experiment_manifest.json). The Windows backup is local; OneDrive synchronization and off-device recovery are unverified. A fresh clone can reproduce record aggregation and integrity checks, but cannot regenerate predictions without recovering the new checkpoint collection or retraining. No false remote-backup claim is made.

Install `pip install -e .` for records-only auditing; use `pip install -e ".[train,test]"` plus the recorded Linux CUDA environment for model inference/tests. Run from repo root:

```sh
python -m kssd.final_audit --output runs/records-only-audit.json
python -m kssd.final_audit --checkpoints runs/persistent-backup-20261012/baseline_fp32_final --output runs/checkpoint-audit.json
python -m kssd.reporting --run runs/persistent-backup-20261012/baseline_fp32_final/modified_unet --output runs/modified-unet-summary-check.json
```

The first command explicitly reports that weights were not checked if none are supplied. The other commands require recovering checkpoints into the documented ignored folder. To regenerate verified prediction figures, also recover the complete run records and official dataset, verify its audited pixel hashes, set PYTHONHASHSEED=42 and TF_NUM_INTRAOP_THREADS=2 / TF_NUM_INTEROP_THREADS=2, then:

```sh
python -m kssd.figures --root runs/persistent-backup-20261012/baseline_fp32_final --data data/kssd2025/data --output runs/recreated-baseline-figures
```

Use the recorded runtime/hardware for numerical reproducibility; different devices/libraries can alter near-threshold binary masks. Figure regeneration refuses mismatched restored counts. This is inference only, not retraining. [Dataset instructions](../docs/DATASET.md) include source/hash and expected extraction layout. Existing test fixtures include a small synthetic optimizer test; the final test pass did not start another real-data CV experiment.

## Checks and remaining limits

- Initial full regression pass: 69 tests passed in 59.35 seconds; seven focused auditor tests passed after adding two final guards. The full post-push suite passed **71 tests**, 1042 upstream warnings, in **254.25 seconds**. Commands, remote file verification, summary regeneration and final backup evidence are in [release verification](final_release_verification.json); [complete test log](resources/final_post_commit_tests.txt).
- Independent saved-record auditor: verified all 831 IDs per architecture, all twenty local checkpoint hashes, optimizer budget/precision/configuration/selection histories and all 30 figure/index file identities. No new outer-test optimization was performed.
- No raw data, environment directory, credential or temporary helper is intentionally staged; final hygiene and link checks are recorded in the release verification report.
- Patient IDs are unavailable; content grouping is heuristic. Near-similar patients/images could still cross groups. Unequal folds and pixel pooling influence aggregate interpretation. Small masks make single pixels important; mask area is not physical stone size. Short budget, no external cohort, no statistical significance claim and incomplete Focal CV limit conclusions.
- Remaining experiment blocker: five-fold Focal Tversky requires new training because no resumable selected fold checkpoint exists. Remaining artifact limitation: new selected weights are local, not remotely accessible. Poor-fold causality is unresolved; no headline-improving rerun was performed.

The audit/release completion claim applies to the requested non-training final pass, not to the unfinished Focal research experiment or the unrun paper-budget reproduction.

The final backup refresh verified 280 files totaling 2,102,753,557 bytes locally,
and skipped three unfinished historical checkpoint copies. Final process/GPU
inspection found no real-data training job: GPU utilization 0%, 9/8188 MiB used.
The training monitor is paused after this final pass because no job is running;
this does not change Focal's pending status. The evidence-only closure commit
adds post-push logs and documentation without changing the tested implementation.
