# PPT preparation brief: required LaTeX Beamer presentation

This filename is for teammates; the submitted slide artifact must be Beamer, not PowerPoint. The prepared deck is `presentation/main.tex` and its compiled PDF. Fifteen slides; rehearse about 10 minutes total (40 seconds average), plus five minutes Q&A. Slides share `report/figures/`, avoiding redundant asset copies.

## Slide 1: Problem, motivation and team

**Exact factual content and numbers:** Pixel segmentation localizes annotated kidney stones on 2D exported CT. Manual outlining motivates automation (Bouzon et al.). No clinical deployment claim. Put group ID, every name and roll number here.

**Equation:** None required.

**Recommended figure/table:** baseline-architecture.png; full asset path `report/figures/`.

**Source:** report/main.tex Introduction; references.bib bouzon2025.

**Presenter understanding:** Why masks, rather than image-level labels? Masks describe pixel location and extent.

**Team input still required:** Names, roll numbers, group ID.

## Slide 2: Dataset, objectives and metrics

**Exact factual content and numbers:** 838 matched pairs; seven redundant copies excluded -> 831 eligible. 512x512 grayscale; binary stone/background; no external project test cohort. Fold outer sizes 179/279/118/172/83. Dice and IoU primary, precision/recall secondary.

**Equation:** Dice=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN).

**Recommended figure/table:** required-three-examples.png; full asset path `report/figures/`.

**Source:** Tables A and split sizes; report Dataset and Evaluation.

**Presenter understanding:** Inner group selection chooses checkpoint; outer folds estimate held-out performance. Groups are not patient IDs.

**Team input still required:** None beyond roster.

## Slide 3: Base paper and literature

**Exact factual content and numbers:** Use full eight-author citation, complete title, IEEE Access 13 (2025), 171790-171806; DOI 10.1109/ACCESS.2025.3610027. Dataset plus four lightweight architectures.

**Equation:** None required.

**Recommended figure/table:** baseline-architecture.png; full asset path `report/figures/`.

**Source:** report/main.tex Base paper; paper/ primary PDF p.1.

**Presenter understanding:** The paper authors created KSSD2025 and their lightweight Modified U-Net; do not claim those as our inventions.

**Team input still required:** None.

## Slide 4: Baseline architecture

**Exact factual content and numbers:** Encoder channels 16/32/64/128; bottleneck 256; two 3x3 conv + BN + ReLU per block. Encoder dropout .1/.2/.3/.3, bottleneck .3. Max pooling and equal-resolution skips.

**Equation:** None required.

**Recommended figure/table:** baseline-architecture.png; full asset path `report/figures/`.

**Source:** kssd/models.py; primary paper Fig.5 PDF p.7.

**Presenter understanding:** Explain spatial compression and skip recovery; diagram is attributed redrawing with code-specific details.

**Team input still required:** None.

## Slide 5: Decoder and implemented alternatives

**Exact factual content and numbers:** Transposed-conv baseline decoder and 1x1 sigmoid. U-Net++ nested skips; U-Net3+ full-scale fusion (22 projection, 110 aggregate); TransUNet 1024 tokens, 192 embedding, 3 blocks/4 heads. No deep supervision or pretrained backbone.

**Equation:** None required.

**Recommended figure/table:** variant-architectures.png; full asset path `report/figures/`.

**Source:** report Methodology; tables/parameters.tex.

**Presenter understanding:** Describe only components in local code, not all enhancements in source papers.

**Team input still required:** None.

## Slide 6: Published versus local results

**Exact factual content and numbers:** Use results-only table C. Paper U-Net Dice .9706/IoU .9465; local baseline .7910/.6589. Preserve all four rows and standard deviations. Protocols differ; settings are documented separately.

**Equation:** None required.

**Recommended figure/table:** comparison.png; full asset path `report/figures/`.

**Source:** Table C; primary paper Table3 PDF p.9; canonical summary.json.

**Presenter understanding:** Several protocol changes coexist; the result table does not establish an exact replication.

**Team input still required:** None.

## Slide 7: Three required segmentation examples

**Exact factual content and numbers:** Display 128.tif, 493.tif, 1068.tif as full Input / Ground Truth / Predicted Mask plus crops. Attention TP/FP/FN 21/0/0, 20/4/1, 0/4/629; Dice 1/.8889/0; IoU 1/.8/0.

**Equation:** None required.

**Recommended figure/table:** required-three-examples.png; full asset path `report/figures/`.

**Source:** report Qualitative; tables/examples.tex; figure index.

**Presenter understanding:** Highest/middle/largest-FN selection is deliberate descriptive error analysis, not an unbiased sample.

**Team input still required:** None.

## Slide 8: Implementation and reproducibility

**Exact factual content and numbers:** data.py -> augmentation.py -> models.py -> train.py -> reporting.py/auditors. Frozen manifest/splits; deterministic new optimizer each fold; preserve remainder batches; earliest inner maximum.

**Equation:** None required.

**Recommended figure/table:** training-log.png; full asset path `report/figures/`.

**Source:** report Implementation; docs/REPRODUCIBILITY.md; kssd/.

**Presenter understanding:** Demo a records-only auditor and explain what it verifies; inference requires data and local weights.

**Team input still required:** Plan who runs the live demonstration.

## Slide 9: Training settings, logs and curves

**Exact factual content and numbers:** Table B: TF2.20/Keras3.13.2; Adam .001; batch2; 10 epochs; float32 Dice; RTX4070 Laptop. Show original CSV log and training/inner curves. Training-only affine/noise, no validation augmentation.

**Equation:** None required.

**Recommended figure/table:** modified_unet-curves.png + training-log.png; full asset path `report/figures/`.

**Source:** Table B; per-fold provenance/history; docs/AUGMENTATION.md.

**Presenter understanding:** Training loss is soft Dice; reported metrics use binary masks. Curves are inner, not repeated outer selection.

**Team input still required:** None.

## Slide 10: Our actual modification

**Exact factual content and numbers:** Baseline is paper-inspired Modified U-Net. Insert four additive attention gates before skip concatenation, inspired by Oktay et al. The config changes architecture only.

**Equation:** None required.

**Recommended figure/table:** attention-architecture.png; full asset path `report/figures/`.

**Source:** report Our modification; docs/ATTENTION_UNET.md.

**Presenter understanding:** Motivation: suppress irrelevant bright tissue. An additional existing architecture alone would not establish our modification.

**Team input still required:** None.

## Slide 11: Gate equation and controlled settings

**Exact factual content and numbers:** a=sigmoid(psi(ReLU(Ws*s+Wg*g))); gated skip=a*s. Intermediate width half skip channels. 1,967,357 parameters vs 1,945,233 (+22,124). Same split, seeds, augmentation, optimizer, loss, threshold, epochs and checkpoint policy.

**Equation:** Gate equation as above.

**Recommended figure/table:** attention-architecture.png; full asset path `report/figures/`.

**Source:** kssd/models.py attention_gate; tables/parameters.tex; comparison.json.

**Presenter understanding:** Matching seed values does not create identical initial weights for different parameter shapes.

**Team input still required:** None.

## Slide 12: Baseline versus Attention: negative result

**Exact factual content and numbers:** Table E: baseline Dice .7910±.0724 vs Attention .7585±.2205; IoU .6589±.0973 vs .6515±.2864. Delta Dice -.032463. Keep all folds; descriptive comparison only.

**Equation:** None required.

**Recommended figure/table:** attention-comparison.png; full asset path `report/figures/`.

**Source:** Table E and fold table D; report modification.

**Presenter understanding:** No mean improvement is established, so a superiority claim and improvement ablation are inappropriate.

**Team input still required:** None.

## Slide 13: Failure analysis and next experiments

**Exact factual content and numbers:** Attention 1143.tif: TP443 FP1344 FN44 Dice.3896; 1068.tif complete miss. Baseline also misses 302.tif. Plausible short budget/tiny foreground/gating dynamics; no proven cause. Focal CV has zero completed folds.

**Equation:** None required.

**Recommended figure/table:** attention_unet-largest_fp-1143.png; full asset path `report/figures/`.

**Source:** report Discussion and Focal subsection; ATTENTION_RESULTS.md.

**Presenter understanding:** Negative-result analysis is explicitly accepted. Next: longer prespecified budget, multiple seeds, gate placement; no post-hoc outer tuning.

**Team input still required:** None.

## Slide 14: Matched small-stone qualitative comparison

**Exact factual content and numbers:** 126.tif: three GT pixels. Baseline 3/1/0 Dice.8571 IoU.75; Attention 2/0/1 Dice.8 IoU.6667. This case illustrates sensitivity to one pixel; pixel area is not millimetres.

**Equation:** None required.

**Recommended figure/table:** paired-small-stone.png; full asset path `report/figures/`.

**Source:** paired-small-stone.png; baseline and Attention figure indexes.

**Presenter understanding:** Do not generalize from a hand-selected tiny case or invent clinical size bins.

**Team input still required:** None.

## Slide 15: Conclusion, contributions and references

**Exact factual content and numbers:** Best canonical mean: Modified U-Net; skip attention lowers mean under ten epochs. Patient IDs/external cohort absent; selected weights local. Cite Bouzon 2025 and Oktay2018 plus architecture papers. State actual member contributions and future work.

**Equation:** None required.

**Recommended figure/table:** Table E or compact conclusion text; full asset path `report/figures/`.

**Source:** report Conclusion, limitations, author contributions; references.bib.

**Presenter understanding:** Assign every member speaking time and prepare shared answers; do not invent ownership from code history.

**Team input still required:** Every member actual contribution, speaking part, submission deadline.

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

## Examiner questions and evidence-based answers

- **Did you reproduce 97% Dice?** No. That is the paper result from 150 epochs. Our verified baseline mean is .7910 under ten epochs and a different corrected/grouped protocol.
- **Where is the test set?** Each outer fold is untouched during training and checkpoint selection. An inner group split selects the checkpoint. There is no extra external project cohort.
- **Are these patient-independent folds?** No patient IDs exist in the export. Content similarity clusters reduce obvious leakage but do not establish patient independence.
- **What did you modify?** Four same-resolution additive spatial gates on the baseline skip connections, adapted from Oktay et al.; not the invention of Attention U-Net.
- **Did the modification improve?** No mean improvement: Dice delta -.032463. Individual folds vary. We retain failures and make no significance claim.
- **Why no component ablation?** The guideline makes improvement ablation conditional. No improvement is claimed. Baseline/no gates versus all gates is the completed controlled comparison, not a gate-level causal study.
- **Why threshold masks at 127?** Decoded intensities occur in 0-9 and 247-255. >0 adds faint background artifacts. Any integer threshold 9-246 gives the same corrected binary mask.
- **How were metrics/checkpoints reconciled?** All hard scores use probability>=.5; earliest inner hard-Dice maximum selects one checkpoint. Integer TP/FP/FN counts and selected-weight hashes were independently checked.
- **Can you reproduce from GitHub?** Saved-record audits and Attention masks can be checked. New inference requires official data and recovered local H5 weights or retraining; selected weights are not published.
- **Does high recall prove good segmentation?** No: TransUNet 688.tif has recall1 but 10,052 false positives and Dice about .0028.
- **Did Focal Tversky work?** NOT YET MEASURED: the interrupted run has zero complete outer folds. Smoke tests show functionality only.
- **Can you measure clinical stone size?** Not from these exported pixel masks alone: no HU, spacing or scan mapping is available; pixel area is only descriptive.
- **Who did what?** Team must supply actual individual contributions; placeholders are explicit and not inferred.
