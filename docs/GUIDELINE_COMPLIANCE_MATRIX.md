# Official guideline compliance matrix

Audit date: 2026-10-13, Asia/Kolkata. All three pages of the supplied
`22AIE437_Final_Project_Guidelines.pdf` were read in full. The course PDF is
the specification; the master prompt adds audit and documentation deliverables.
PASS means verified evidence, not filename existence. This is the initial audit;
the final deliverable verification updates the status after compilation.

| ID | Official page / section | Requirement | Initial status | Evidence / action |
|---|---|---|---|---|
| G01 | 1, Format | LaTeX Beamer source and compiled PDF | FAIL | No existing presentation; create and compile presentation/main.tex. |
| G02 | 1, Format | BibTeX if used | FAIL | Create verified references.bib for report and slides. |
| G03 | 1, Format; 2, Implementation | Reproducible implementation submission | PARTIAL | kssd/, configs/, tests/ and full saved records exist; selected weights remain local, raw dataset obtained separately. |
| G04 | 1, Format; 2, 2A; 3, Report | Elsevier CAS single-column source, PDF, bibliography and figures | FAIL | No CAS report exists; create report/main.tex using official cas-sc. |
| G05 | 1, Format; 3, Report | Group ID, all names and roll numbers; every author | BLOCKED | Not recovered from repository; requested from user; explicit placeholders required. |
| G06 | 1, Format; 3, Slide plan | Suggested 12-15 slides, 10-minute talk and 5-minute Q&A | FAIL | Prepare 15 slides and timing notes. Actual delivery unverified. |
| G07 | 1, Introduction | Problem, clinical motivation, reason for AI | PARTIAL | Base paper explains motivation; add attributed report/slide explanation without clinical deployment claims. |
| G08 | 1, Introduction | Dataset name, source and image count | PASS | docs/DATASET.md; reports/dataset/manifest.json: 838 pairs, 831 eligible unique samples. |
| G09 | 1, Introduction | Labels and actual split protocol | PASS | Binary stone/background, reports/splits.json; grouped outer test + inner selection; no separate external test cohort. |
| G10 | 1, Introduction | Objectives and metrics | PARTIAL | Corrected metrics exist; consolidate explicit objectives in both deliverables. |
| G11 | 1, Base paper | Full title, authors, venue, year and citation | PARTIAL | Local primary paper inspected; DOI 10.1109/ACCESS.2025.3610027; add complete BibTeX. |
| G12 | 1, Base paper | Base architecture diagram and key blocks | PARTIAL | Paper Fig. 5, PDF p.7; code matches lightweight encoder design; redraw with attribution. |
| G13 | 1-2, Results | Published vs local Dice and Jaccard table | PARTIAL | Paper Table 3, PDF p.9 visually checked; canonical JSON summaries verified; integrate side by side with protocol caveats. |
| G14 | 2, Segmentation figures | Three Input / Ground Truth / Predicted Mask examples | PASS | Existing selected-checkpoint panels with three explicit views; reports/figures/baseline10fp32_final and Attention figures/index.json. |
| G15 | 2, Implementation | Data loading, model, training loop and evaluation | PASS | kssd/data.py, models.py, train.py; existing regression suite; full saved evaluations. |
| G16 | 2, Implementation | Working-code demonstration | UNVERIFIED | Code and audit commands available; actual classroom demonstration requires team. |
| G17 | 2, Implementation | Training-log screenshots and loss/metric curves on slides | PARTIAL | history.csv/json and curves exist; add log excerpt figure and readable curves to slides. |
| G18 | 2, Implementation | Framework, optimizer, LR, batch, epochs, hardware | PASS | Per-fold provenance and reports/experiment_manifest.json: TF 2.20, Adam .001, batch 2, 10 epochs, RTX 4070 Laptop. |
| G19 | 2, Modification | Define baseline and our actual modification | PASS | Modified U-Net baseline; four additive skip gates in attention_unet, described in docs/ATTENTION_UNET.md. |
| G20 | 2, Modification | Modified architecture diagram | FAIL | Create accurate gate and overall architecture diagram. |
| G21 | 2, Modification; 3, Fair comparison | Baseline/modified comparison, identical splits, seeds, settings | PASS | comparison.json; five folds, architecture-only config difference, matched seed 43-47 and training budget. |
| G22 | 2, Analysis | Ablation if improvement is claimed | PASS | No improvement claimed: mean Attention Dice delta -0.032463. Component ablation is conditional, not required for this negative result. |
| G23 | 2, Analysis | Explain negative result and next steps | PARTIAL | docs/ATTENTION_RESULTS.md includes failures; integrate evidence and distinguish plausible causes from established findings. |
| G24 | 2, 2A; 3, Report | Abstract, introduction, replication, methodology, results, discussion, conclusion, references | FAIL | Create complete article, keywords, tables and figures. |
| G25 | 3, Report | Actual author contributions | BLOCKED | Human contributions unknown; do not invent. |
| G26 | 3, General rules | High-resolution, readable, labelled figures; table captions | PARTIAL | Existing figures reviewed previously; new report/deck require rendering and visual QA. |
| G27 | 3, General rules | Base paper and other sources cited | PARTIAL | Documentation links exist; add verified article bibliography. |
| G28 | 2-3, Deadline / writing guidance | Before-exam submission and lecture writing guidance | UNVERIFIED | Exam date and lecture writing slides not supplied. No submission claim. |
| G29 | 3, Teamwork | Every member presents and answers; contribution slide | BLOCKED | Team roster/roles unavailable; actual participation depends on team. |
| G30 | 3, Originality | Own code/results and attributed prior methods | PARTIAL | Recorded source commits and measured runs; architectures, dataset and template require explicit attribution; personal authorship cannot be inferred. |

## Additional master-prompt gates

| ID | Requirement | Initial status | Evidence / action |
|---|---|---|---|
| U01 | Inventory all notebooks, assets and numerical sources | PARTIAL | Existing docs/AUDIT.md and legacy_archive_audit.json; create current inventory and classify stale output claims. |
| U02 | Consistent hard metrics and edge cases | PASS | kssd/metrics.py, tf_metrics.py; tests/test_metrics.py includes empty, missed, FP, perfect and overlap. |
| U03 | Best inner checkpoint; no outer epoch selection | PASS | Full histories/provenance and final_audit.py verify earliest inner hard-Dice maximum. |
| U04 | Pairing, dimensions, masks, normalization, duplicates and split checks | PASS | Frozen 838-pair manifest, 831 eligible; corrected palette threshold >127; grouped inner/outer split validation. |
| U05 | Train-only aligned augmentation | PASS | kssd/augmentation.py and tests/test_augmentation.py; masks nearest-neighbour. |
| U06 | Corrected five-fold results for all four architectures | PASS | 20 complete float32 canonical evaluations; derived metrics and hashes independently checked. |
| U07 | Extend to paper budget where feasible | PARTIAL | 150-epoch configs exist; 10-epoch limitation explicit. Full paper-budget replication unrun; estimate runtime before decision. |
| U08 | Research-inspired modification, retaining existing appropriate experiment | PASS | Completed Attention comparison qualifies; Focal Tversky was interrupted with zero completed folds and is not substituted. |
| U09 | Full report and PPT_PREPARATION_BRIEF with all actual data | FAIL | Generate article and slide-by-slide brief with dataset/config/paper/fold/modification tables. |
| U10 | Phase commits, pushes and remote verification | UNVERIFIED | Starting main 480826c matches live GitHub; record each new completed phase. |

The project is initially partially ready. Missing human information remains
blocked even after a technically complete draft is generated. Training experiments
must not be represented as paper-equivalent or patient-independent evaluation.
