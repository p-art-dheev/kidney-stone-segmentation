# Five named model files

| File | Model | What the architecture does |
|---|---|---|
| [modified_unet.py](modified_unet.py) | Modified U-Net | Convolutional encoder/decoder with skip connections, batch normalization and dropout |
| [unet3plus.py](unet3plus.py) | U-Net3+ | Full-scale skip connections combine features from multiple resolutions |
| [unetpp.py](unetpp.py) | U-Net++ | Nested skip connections connect intermediate decoder features |
| [transunet.py](transunet.py) | TransUNet | Convolutional features pass through transformer blocks before decoding |
| [attention_unet.py](attention_unet.py) | Attention U-Net | Four attention gates weight encoder skip features before decoding |

Each file is an explicit entry point to the existing architecture implementation
in `kssd/models.py`. It exposes `build()` for a fresh 256×256 grayscale model,
`load_best()` for the completed trained model, `BEST_CHECKPOINT` for the local
weight path and `RESULTS` for its saved logs and metrics. The shared implementation
keeps the named entry points consistent with the architectures that were trained.

From the repository root:

```python
from strain.models.attention_unet import load_best, RESULTS

model = load_best()
print(RESULTS)
# Input: float32 image batches shaped (N, 256, 256, 1), normalized to [0, 1].
probabilities = model(images, training=False)
masks = probabilities >= 0.5
```

The files do not start training when imported. Training runners remain
`strain/full/run.py` for the first four models and `strain/attention/run.py`
for Attention U-Net. The first four completed 20 additional epochs; Attention
U-Net completed a separate fresh 50-epoch run.

| Saved file | Meaning |
|---|---|
| `*.best.weights.h5` | Best validation-selected trained weights; stored locally under `strain/checkpoints/` |
| `history.csv` / `history.json` | Training and validation loss, Dice, IoU, precision and recall for each epoch |
| `training_curves.png` | Curves drawn from those epoch logs |
| `selection.json` | Selected checkpoint, validation scores and evaluation settings |
| `result.json` | Final validation and held-out test scores, including thresholds and pooled counts |
| `report.md` | Experiment summary, measured runtime and limitations |

Code, logs, curves and reports are pushed to Git. Checkpoints remain local and
are excluded from Git. All five output folders and checkpoint paths are linked
in [the completed model index](../COMPLETED_MODELS.md).
