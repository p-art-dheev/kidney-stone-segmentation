# Verified qualitative and error analysis

Implementation is available; measured corrected-CV figures have not been generated.
The old figures in Results/ use the superseded protocol and remain historical assets.

After all four baseline models have complete, matched five-fold evaluations:

```sh
python -m kssd.figures --root runs/baseline --data data/kssd2025/data --output runs/baseline-figures
```

Use the same numerical environment as training. All five selected checkpoints,
history selection epochs, source hashes, IDs, confusion counts and metrics are
validated before accepting results. Existing figure output directories are rejected.
The index records status running until all figures finish; partial generation is
not a completed qualitative analysis. Every restored prediction must reproduce
the saved confusion counts. Evaluation batch shape/order is preserved to avoid
changing floating-point kernels near the decision threshold.

For each model, deterministic case selectors span highest/lowest Dice, largest FN,
largest FP, smallest ground-truth mask and middle Dice. Ties use filenames. These
labels describe actual rankings, not assumed successful cases: if highest Dice is
zero, no good segmentation is implied; if largest FP is zero, no false-positive
example is implied. Duplicate selections remain recorded transparently. No cases
are manually selected to flatter a model. Comparisons use labels identifying each
model, sample and outer fold rather than implying different cases are identical.

Each image figure includes original normalized CT, binary ground truth, binary
prediction, TP/FN/FP overlay and a 128-pixel crop around the largest connected
ground-truth component. Colors: yellow TP, red FN, cyan FP. The crop helps reveal
tiny masks; full-image context stays visible. Title shows per-image Dice, threshold
and confusion counts. These are engineering error examples, not expert adjudication
of annotation ambiguity or clinical findings.

The four-model comparison shows mean outer-fold pooled-pixel metrics with **sample
SD across five folds**, not confidence intervals. The figure index stores the exact
plotted values and checkpoint hashes. Each model's curve figure shows all five
training/inner-selection losses and binary Dice histories. Outer-test curves do not
influence checkpoint selection and are not plotted as epoch validation histories.

For the controlled Modified U-Net loss run alone:

```sh
python -m kssd.figures --root runs/focal --architectures modified_unet --data data/kssd2025/data --output runs/focal-figures
```

Compare losses through `kssd.compare_losses`, not the architecture-comparison
protocol checker. Synthetic rendering tests use temporary, explicitly labelled
fixtures; those images are never published as segmentation results.
