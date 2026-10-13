# Repository inventory and claim sources

Inventoried 2892 tracked files and 18 cells across three notebooks. Complete sources and retained text outputs are in `reports/compliance-20261013/notebook_claim_inventory.json`. File sizes and SHA-256 values are in `repository_inventory.json`. These hashes inventory the starting evidence, not new generated outputs.

## Published literature

Local IEEE paper, Table 3 p.9; externally reported 150-epoch scores.

## Verified canonical measurements

reports/experiments/baseline10fp32_final: four models, twenty folds; Attention export: five folds.

## Derived results tables

summary.json, comparison.json, docs/BASELINE_RESULTS.md and ATTENTION_RESULTS.md; independently recomputed from counts.

## Historical results

Stored notebook outputs, Results/*/*.png and twenty .keras archives; final-epoch, old mask/metric/split protocol.

## Exploratory results

baseline10amp_verified: mixed precision; TransUNet NaN failure; never mix with canonical float32.

## Incomplete experiments

Focal Tversky has zero completed outer folds; no legitimate comparative score. 150-epoch runs unavailable.

## Functionality only

CPU/GPU smoke and checkpoint round-trip logs; not CV or clinical performance.

## Unverified human facts

Group number, names, roll numbers and individual contributions absent; no pre-existing report or presentation found.

## Notebook review

- `kdss-modified-unet-10epoch.ipynb`: 6 cells, 0 cells with stored text outputs. Current notebooks delegate to the shared pipeline and have no stored execution outputs; legacy notebook versions remain in Git history.
- `kdss-unetpp-unet3p-10epochs.ipynb`: 6 cells, 0 cells with stored text outputs. Current notebooks delegate to the shared pipeline and have no stored execution outputs; legacy notebook versions remain in Git history.
- `kssd-transunet-10epochs.ipynb`: 6 cells, 0 cells with stored text outputs. Current notebooks delegate to the shared pipeline and have no stored execution outputs; legacy notebook versions remain in Git history.

The corrected package removes independent metric maxima and outer-fold epoch selection, validates strict pairs, uses decoded masks >127, retains incomplete batches, seeds runs, and separates inner checkpoint selection from outer evaluation. Architectural equivalence to all source-paper implementations is not established.
