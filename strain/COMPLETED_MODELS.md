# Completed model files

The completed run is **full256_20261013_v2**. All four models completed 20
additional epochs. Each model has its own output folder. Architecture entry
points share `kssd/models.py`; U-Net3+ and U-Net++ use the installed
`keras_unet_collection` builders through that file.

| Model | Architecture entry point | Completed output folder |
|---|---|---|
| Modified U-Net | [modified_unet()](../kssd/models.py#L18) | [modified_unet](results/full256_20261013_v2/modified_unet/) |
| U-Net3+ | [build_model("unet3plus")](../kssd/models.py#L80), calling `unet_3plus_2d` | [unet3plus](results/full256_20261013_v2/unet3plus/) |
| U-Net++ | [build_model("unetpp")](../kssd/models.py#L80), calling `unet_plus_2d` | [unetpp](results/full256_20261013_v2/unetpp/) |
| TransUNet | [transunet()](../kssd/models.py#L53) | [transunet](results/full256_20261013_v2/transunet/) |

## Modified U-Net

- [Final validation and test results](results/full256_20261013_v2/modified_unet/result.json)
- [Epoch loss and metric log, CSV](results/full256_20261013_v2/modified_unet/history.csv)
- [Epoch loss and metric log, JSON](results/full256_20261013_v2/modified_unet/history.json)
- [Loss and metric curves](results/full256_20261013_v2/modified_unet/training_curves.png)
- [Selected checkpoint metadata](results/full256_20261013_v2/modified_unet/selection.json)
- Local weights: `strain/checkpoints/full256_20261013_v2/modified_unet.best.weights.h5`

## U-Net3+

- [Final validation and test results](results/full256_20261013_v2/unet3plus/result.json)
- [Epoch loss and metric log, CSV](results/full256_20261013_v2/unet3plus/history.csv)
- [Epoch loss and metric log, JSON](results/full256_20261013_v2/unet3plus/history.json)
- [Loss and metric curves](results/full256_20261013_v2/unet3plus/training_curves.png)
- [Selected checkpoint metadata](results/full256_20261013_v2/unet3plus/selection.json)
- Local weights: `strain/checkpoints/full256_20261013_v2/unet3plus.best.weights.h5`

## U-Net++

- [Final validation and test results](results/full256_20261013_v2/unetpp/result.json)
- [Epoch loss and metric log, CSV](results/full256_20261013_v2/unetpp/history.csv)
- [Epoch loss and metric log, JSON](results/full256_20261013_v2/unetpp/history.json)
- [Loss and metric curves](results/full256_20261013_v2/unetpp/training_curves.png)
- [Selected checkpoint metadata](results/full256_20261013_v2/unetpp/selection.json)
- [Loss trial comparison](results/full256_20261013_v2/unetpp/loss_comparison.json)
- Local weights: `strain/checkpoints/full256_20261013_v2/unetpp.best.weights.h5`

## TransUNet

- [Final validation and test results](results/full256_20261013_v2/transunet/result.json)
- [Epoch loss and metric log, CSV](results/full256_20261013_v2/transunet/history.csv)
- [Epoch loss and metric log, JSON](results/full256_20261013_v2/transunet/history.json)
- [Loss and metric curves](results/full256_20261013_v2/transunet/training_curves.png)
- [Selected checkpoint metadata](results/full256_20261013_v2/transunet/selection.json)
- [Loss trial comparison](results/full256_20261013_v2/transunet/loss_comparison.json)
- Local weights: `strain/checkpoints/full256_20261013_v2/transunet.best.weights.h5`

## Shared experiment code

| File | Purpose |
|---|---|
| [full/run.py](full/run.py) | Four-model runner, deadline, checkpoint selection and final evaluation |
| [full/train.py](full/train.py) | Model compilation, paired augmentation, losses, pilots and fitting |
| [full/data.py](full/data.py) | Input/cache verification and partition loading |
| [targeted/diagnose.py](targeted/diagnose.py) | Output diagnostics and validation threshold checks |
| [plot_history.py](plot_history.py) | Training/validation loss and all four metric curves |
| [full/evaluate_sealed.py](full/evaluate_sealed.py) | Evaluate already selected checkpoints |
| [full/audit_results.py](full/audit_results.py) | Independently verify saved results |
| [full/finalize_results.py](full/finalize_results.py) | Generate the final report |

The [final report](results/full256_20261013_v2/report.md) includes the comparison,
timing, validation scores, held-out test scores and limitations. Weight files,
caches and raw CT panels are local and excluded from Git. Attention U-Net is
outside this four-model run.
