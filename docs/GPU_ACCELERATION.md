# GPU acceleration check, 12 October 2026

All four architectures passed five actual GPU optimizer updates on one fixed
augmented 512x512 batch of two, finite float32 probability checks, and checkpoint
restoration under mixed_float16. All recorded optimizer iteration counts were 5.
The regression suite passed 63 tests. Evidence is in
`reports/resources/gpu_amp2_smoke.json` and `amp_tests.txt`.

Mean repeated-step seconds, excluding first compilation:

| Architecture | Earlier float32 batch 2 | mixed_float16 batch 2 |
|---|---:|---:|
| Modified U-Net | 0.206 | 0.102 |
| U-Net++ | 0.150 | 0.110 |
| U-Net3+ | 0.624 | 0.452 |
| TransUNet | 0.216 | 0.150 |

These are short fixed-batch timing checks, not complete epoch benchmarks or CV
results. Runtime depends on data loading, validation and compilation too.
Batch four mixed precision and batch eight float32 exceeded available memory
in U-Net3+. All final comparison configs therefore keep common batch two,
10 epochs, all five folds and the frozen split/augmentation protocol.
The two completed earlier float32 Modified U-Net folds and interrupted third
fold remain preserved in `runs/baseline`; they are excluded from the new matched
comparison. The initial `runs/baseline_amp` launch also reset the policy during
Keras session clearing; its runtime provenance correctly recorded float32/Adam.
It was stopped after this was discovered and excluded from comparison. The
corrected runner sets the policy after clearing the session and asserts that
mixed precision uses LossScaleOptimizer. A synthetic end-to-end runner test
now verifies actual runtime policy and optimizer for both precision modes.
New runs use `runs/baseline_amp_verified` with all folds restarted.

Output sigmoid computations and losses remain float32; model variables remain
float32. Keras supplies dynamic loss scaling automatically. See the
[TensorFlow mixed precision guide](https://www.tensorflow.org/guide/mixed_precision).
This protocol differs from the paper budget; 150-epoch/batch-16 reproduction
has not been performed. Policy selection used runtime and memory checks only.

## Longer-run failure and final fallback

The corrected mixed-precision run completed all five folds for Modified U-Net,
U-Net++ and U-Net3+. TransUNet fold 1 developed NaN inner-validation loss at
epoch 3 and NaN training loss at epoch 4, batch 39. TerminateOnNaN stopped
training, and strict JSON serialization rejected the non-finite history.
The actual trace is `reports/resources/transunet_amp_failure.txt`.
The short five-update checks did not establish longer-run numerical stability.
These mixed-precision results are retained as exploratory records and are not
combined with float32 results into a matched architecture comparison.

A four-epoch TransUNet fold-1 float32 diagnostic completed with finite histories
in 203.26 seconds and no outer-test evaluation. Its compact records are under
`reports/resources/transunet_fp32_diagnostic/`. This verifies the diagnostic
budget only, not all folds or longer training. The final common configs now
use float32, batch two, ten epochs and five folds for every architecture and
the matched Focal Tversky experiment. Final runs restart all folds under
`runs/baseline_fp32_final`; previously generated outputs remain preserved.
