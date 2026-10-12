# Attention U-Net: completed matched 10-epoch results

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 | 0.8298 ± 0.1621 | 0.7786 ± 0.0755 |
| Attention U-Net | 0.7585 ± 0.2205 | 0.6515 ± 0.2864 | 0.7890 ± 0.2621 | 0.7803 ± 0.1996 |

Mean ± sample SD over five outer grouped folds. Descriptive comparison only; no significance, clinical or external-generalization claim. All folds and zero-Dice cases are retained. Fold 1 was the retained pilot, not retrained.

Mean Dice difference, Attention minus Modified: -0.032463.

| Fold | Selected epoch | Dice | IoU | Precision | Recall | Training seconds | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 9 | 0.474540 | 0.311080 | 0.344465 | 0.762456 | 378.20 | `25830fd50270bd4bd7df11db75464eec05f68602dbebf17f36daa046bed2b624` |
| 2 | 6 | 0.796644 | 0.662019 | 0.812737 | 0.781177 | 411.91 | `843f66fa44fc426bae1a9da08e1f74618b20eaaa4abf01c4ab765f343943969c` |
| 3 | 7 | 0.942443 | 0.891150 | 0.988570 | 0.900428 | 586.19 | `44b782da1a44782dc3447096895c93a32b8f990f09b8ae96ad7388e06c4ec114` |
| 4 | 5 | 0.593772 | 0.422244 | 0.821501 | 0.464897 | 527.52 | `8bbeef5cf53c20484550d72112a5ddd4a728e7b1c3ae6c91539298d42998dbab` |
| 5 | 10 | 0.985292 | 0.971011 | 0.977950 | 0.992746 | 554.67 | `71b76bae7315075d3f6b27dbdc12c1b6bc5c045d424f9559ea3c50234ff01a39` |

Total measured training time: 2458.49 seconds; runner wall time (including validation, masks and figures): 2606.04 seconds.

All 831 saved prediction masks passed independent count/hash checks against original decoded ground truth. All five checkpoint hashes, finite complete histories, optimizer update budgets, earliest inner maxima, group split coverage and configurations passed. No new outer-test forward pass was used for this audit or figures.

[Complete records and figures](../reports/experiments/attention10fp32_20261012/) · [Independent validation](../reports/experiments/attention10fp32_20261012/independent_validation.json) · [Protocol](ATTENTION_UNET.md)

Checkpoint collection is local/ignored in the native run and checksum-verified Windows backup; it is not published remotely. Binary masks, compact records, hashes, logs and selected figures are Git artifacts. See the experiment manifest for paths and reproduction commands.

All eight plots were visually reviewed: comparison, five-fold curves and six
deterministically selected success/failure panels. 128.tif has TP/FP/FN 21/0/0
(Dice 1); 126.tif has only three foreground pixels and TP/FP/FN 2/0/1 (Dice 0.8).
1068.tif misses all 629 foreground pixels (0/4/629); 1012.tif misses all eight
(0/0/8). 1143.tif detects annotated foreground but also predicts bright
background structures (443/1344/44; Dice 0.3896). Median-Dice 493.tif has
20/4/1 and Dice 0.8889. Displayed masks and counts agree. The exact mechanism
behind fold variability is unresolved; the short budget, small foreground and
initialization/normalization dynamics are plausible contributors, not established
causes. No setting was tuned or fold discarded to raise the headline score.

Final regression suite: **80 passed**, 1166 upstream warnings, **77.50 seconds**.
Full-size GPU smoke, independent ground-truth intersections for all saved masks,
all selected checkpoint hashes, protocol comparison and deterministic figure
selection passed. No Attention experiment blocker remains. The checkpoint
publication limitation and existing Focal/150-epoch pending work remain separate.
