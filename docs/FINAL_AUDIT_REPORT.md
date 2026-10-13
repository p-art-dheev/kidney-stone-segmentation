# Final course-compliance audit — 13 October 2026

## A. Overall status

**Partially ready for submission; ready for technical draft review.** The
official three-page guideline was read in full. The actual Elsevier CAS
single-column report and actual 15-slide LaTeX Beamer presentation are compiled.
Numerical evidence, source bibliography, figures, full results and the separate
slide preparation brief are available. Group identity, all names/roll numbers
and actual individual contributions remain explicitly marked placeholders.
Those human requirements are not passed merely because a placeholder is present.

The user's requested workload wording is **250 completed training epochs across
25 fold runs**. The settings retain ten epochs per fold, accurately matching the
logs. Literature comparison tables focus on Dice and IoU; training settings and
protocol differences are documented separately. No run is relabelled as a
150-epoch experiment.

## B. Requirement-by-requirement evidence

`GUIDELINE_COMPLIANCE_MATRIX.md` contains 30 official-course gates and 10
additional master-prompt gates, their initial and final status, page/section
locators, evidence and action. PASS means a verified implementation/artifact.
BLOCKED means unavailable human/external information. PARTIAL and UNVERIFIED
are preserved for incomplete reproduction, deadlines, live demonstration and
authorship facts. Conditional improvement ablation is not triggered by the
measured negative result.

## C. Verified experimental data

- Original data: 838 matched 512x512 uint8 grayscale TIFF pairs, no empty masks;
  50 masks contain at most ten foreground pixels. Seven exact redundant images
  excluded, leaving 831 eligible samples. Area min/Q1/median/Q3/max:
  3/38.25/101.5/231/755 pixels; not clinical sizes.
- Corrected target: palette-decoded grayscale >127. Image normalization /255;
  no resizing. Original mask values have bands 0-9 and 247-255.
- Frozen content groups: 39 clusters, not verified patient IDs. Outer test
  sizes 179/279/118/172/83; inner selection sizes 206/97/48/70/142; training
  sizes 446/455/665/589/606. Seeds 43-47, base 42. No separate external cohort.
- TensorFlow 2.20, Keras 3.13.2, float32, Adam LR .001, batch two, Dice loss,
  training-only aligned augmentation and prediction threshold >=.5.
- Each fold uses its earliest maximum inner hard-Dice checkpoint; outer data
  never select the epoch. Fold-pooled counts yield metrics; unweighted fold
  mean and sample SD use ddof=1. Macro and all-fold-pooled scores are separate.

| Model | Dice mean ± SD | IoU mean ± SD |
|---|---|---|
| Modified U-Net | 0.7910 ± 0.0724 | 0.6589 ± 0.0973 |
| U-Net++ | 0.3678 ± 0.2567 | 0.2490 ± 0.1894 |
| U-Net3+ | 0.3230 ± 0.4535 | 0.2805 ± 0.4098 |
| TransUNet | 0.4842 ± 0.3380 | 0.3749 ± 0.3166 |
| Attention U-Net | 0.7585 ± 0.2205 | 0.6515 ± 0.2864 |

Full four-metric aggregate tables, all 25 per-fold scores, parameter counts,
runtime measurements, dataset/configuration tables and published comparisons
are in `EXPERIMENTAL_RESULTS.md`, `PPT_PREPARATION_BRIEF.md` and the article.
Original Table 3 was visually inspected in the primary IEEE paper, p.9.
Published Dice/IoU: .9706/.9465, .9663/.9397, .9668/.9402, .9553/.9221 for
the four respective architectures. These are literature values, not local runs.

## D. Modification assessment

The baseline is the paper-inspired lightweight Modified U-Net. The new local
adaptation adds four same-resolution additive spatial skip gates inspired by
Oktay et al. Encoder/decoder block design, split, seed values, loss, optimizer,
augmentation, threshold and training budget remain matched; architecture alone
differs in the recorded config. Matching seeds does not imply identical
corresponding initial weights after new layers consume random draws.

Parameter counts: baseline 1,945,233; Attention 1,967,357 (+22,124).
Mean Dice delta is -0.032463, mean IoU delta -0.007415. Individual fold outcomes
vary. There is no improvement/significance claim. The report shows actual false
positives and misses and distinguishes plausible mechanisms from proven causes.
The no-gate/all-gate comparison is controlled but not a gate-level ablation.

Focal Tversky functionality is implemented and smoke-tested, but its interrupted
run has **zero completed epochs and zero complete outer folds**. All Focal
performance cells remain NOT YET MEASURED. The master prompt allows retaining
a different already completed appropriate modification; Attention is used.

## E. Deliverables

| Path | Purpose |
|---|---|
| report/main.tex and main.pdf | Full actual CAS single-column article and compiled report |
| report/references.bib | Verified eight-source bibliography with all base-paper authors |
| report/cas-sc.cls, cas-common.sty, cas-model2-names.bst | Unmodified official Elsevier support files |
| report/figures/ and tables/ | Diagrams, curves, verified qualitative views, complete numerical tables |
| presentation/main.tex and main.pdf | Required 15-slide Beamer source and compiled PDF |
| presentation/references.bib | Matching full reference database |
| docs/PPT_PREPARATION_BRIEF.md | Fifteen slide-by-slide plans, exact facts, figures, tables and examiner Q&A |
| docs/GUIDELINE_COMPLIANCE_MATRIX.md | Initial and final statuses of every course and audit gate |
| docs/EXPERIMENTAL_RESULTS.md | Actual data/configuration/aggregate/per-fold/comparison tables |
| docs/REPRODUCIBILITY.md | Locked environment, audit/train/recovery/build commands and limits |
| docs/REPOSITORY_INVENTORY.md | File/asset inventory and classification of numerical sources |
| reports/compliance-20261013/ | Fresh independent verification, notebook/file inventory, tests and data ledger |
| scripts/ | Evidence inventory, data-driven figure/table/brief builder, integrity and PDF/package verifier |

The presentation shares `../report/figures/`; distribute both source directories
together. A generic-slide PDF or PowerPoint export was not substituted.

## F. Exact remaining blockers and limitations

1. Group number/name, all members' names and roll numbers, actual contributions
   and applicable corresponding-author details are unavailable; requested from
   the user. Drafts use explicit placeholders. Group size is not invented.
2. Exam deadline and separate lecture journal-writing guidance are not supplied;
   actual submission and classroom participation/demonstration are unverified.
3. Selected corrected-run H5 weights are locally backed up but not published on
   GitHub. Raw dataset is separately obtained. Record audit is reproducible from
   a clone; new inference requires recovery or retraining. Cloud recovery unverified.
4. Extended paper-budget experiments remain unrun. Linear extrapolation at the
   measured batch-two speed is 72.95 GPU-hours for four models at 150 epochs per
   fold. This estimate is not a measured run. No unavailable scores are invented.
5. Focal CV remains unexecuted after interruption; no resumable complete fold
   exists. It is not needed to substitute for the completed Attention modification.
6. Patient/scan identities, an external evaluation cohort, physical spacing and
   independent expert annotation validation are unavailable. Content grouping
   cannot establish patient-independent or clinical generalization.

## G. Verification and Git phases

Phase 1: `3ae3188` — Audit all course requirements and independently verify
existing experiment evidence. Pushed to `main`; live remote SHA matched
`3ae3188b8fc7eaf9de1ac2faa13b4b3f0867a71c`.

Fresh full regression: **80 passed**, 1166 upstream warnings, 406.96 seconds.
Fresh baseline audit: twenty selected checkpoints match SHA-256; full histories,
configurations, 831 outer IDs per architecture, metrics and thirty figure/index
identities verified. Fresh Attention audit: five selected checkpoints and all
831 saved masks checked against decoded ground truth. Fresh 838-pair dataset
manifest matches the frozen experimental manifest exactly.

The deliverable verifier checks CAS/Beamer source classes, source packaging,
15 slide pages, canonical table rounding, complete 25-fold/250-epoch coverage,
all generated figure hashes and resolved reference tokens. Build logs and final
PDF checksums are recorded in the same evidence directory. All rendered report
pages and slides are visually reviewed; small-slide figures receive separate
readable variants. Compilation warnings from the CAS front-matter box and
Beamer internal nullfont counters are tracked; rendered content was inspected
and shows no lost text or clipped table content.

Phase 2 commit/push identity is recorded in `git_release_verification.json`
after the full source/PDF package is committed. A final evidence-only closure
records the remote identity without rewriting history. See the final chat for
the closure commit hash; it cannot record its own hash within itself.

## H. Recommendation and 20 acceptance answers

| # | Acceptance question | Final answer and evidence |
|---|---|---|
| 1 | All three guideline pages checked? | YES; full extraction and requirement locators. |
| 2 | Every requirement in matrix? | YES; G01-G30 and U01-U10. |
| 3 | Beamer source compiled? | YES; actual beamer class, 15 pages. |
| 4 | CAS single-column report compiled? | YES; official cas-sc class and compiled article. |
| 5 | Correct title group details complete? | BLOCKED; explicit placeholders, user information required. |
| 6 | Base paper cited and explained? | YES; all eight authors, IEEE Access 2025, DOI, blocks and attributed diagram. |
| 7 | Paper/local Dice-IoU comparison? | YES; Table 3 visually verified and local counts recomputed. |
| 8 | Four architectures/results verified? | YES; twenty complete corrected fold evaluations and local weight hashes. |
| 9 | At least three required views? | YES; three full Input/GT/Prediction examples plus magnified insets/crops. |
| 10 | Logs, curves, settings documented? | YES; all histories, CSV excerpt, curves, provenance and configuration table. |
| 11 | Baseline and modification explicit? | YES; lightweight baseline plus four additive skip gates. |
| 12 | Modified diagram available? | YES; code-specific architecture and gate equation. |
| 13 | Fair comparison? | YES for recorded configuration; matching splits/seeds/budget verified. |
| 14 | Ablation if improvement claimed? | NOT TRIGGERED; no mean improvement claimed. Gate-level study is future work. |
| 15 | Negative result honestly discussed? | YES; actual deltas and failures, no manufactured benefit. |
| 16 | Reproducible code/environment? | PARTIAL; audits and exact commands available; fresh inference needs local weights/data. |
| 17 | Complete report sections/artifacts? | YES technically; author roster and real contributions BLOCKED. |
| 18 | Brief has actual data/slide instructions? | YES; 15 plans, tables A-E and examiner questions. |
| 19 | Numerical claims source checked? | YES for headline, fold, data, runtime and cited paper table claims; estimates and unavailable work labelled. |
| 20 | Completed tasks committed/pushed? | See git_release_verification.json and final closure hash; no force push. |

Review the technical draft now. Fill verified human information, recompile and
recheck the title/contribution slides, confirm the submission date and rehearse
the live demonstration before treating it as final-submission ready.
