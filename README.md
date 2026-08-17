# Kidney Stone Segmentation — KSSD2025

Replication of the paper: **"KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models"** (IEEE Access, 2025).

## Overview

Automatic segmentation of kidney stones from axial CT images using four U-Net-based deep learning architectures. All models are evaluated using **5-fold cross-validation** on the KSSD2025 dataset.

## Dataset — KSSD2025

| Property | Value |
|----------|-------|
| **Total images** | 838 axial CT scans |
| **Image size** | 512 × 512 pixels |
| **Format** | Grayscale TIFF (.tif) |
| **Labels** | Binary masks (stone / background) |
| **Source** | Derived from "CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone" (Islam et al., 2022) |
| **Annotation** | Threshold → connected component filtering → expert manual refinement |

The dataset is publicly available on Kaggle: [`murillobouzon/kssd2025-kidney-stone-segmentation-dataset`](https://www.kaggle.com/datasets/murillobouzon/kssd2025-kidney-stone-segmentation-dataset)

## Models

### 1. Modified U-Net (Custom)
Standard U-Net encoder-decoder with:
- **Encoder**: 4 blocks `[16, 32, 64, 128]` filters — each with 2×(Conv3×3 + BatchNorm + ReLU) + Dropout + MaxPool
- **Bottleneck**: 256 filters
- **Decoder**: Conv2DTranspose upsampling + skip connections + 2×(Conv3×3 + BatchNorm + ReLU)
- **Dropout rates**: 0.1, 0.2, 0.3, 0.3, 0.3
- **Parameters**: ~1.4M

### 2. U-Net++
Nested U-Net with dense skip connections. Built using `keras-unet-collection` with filters `[16, 32, 64, 128, 256]`.
- **Parameters**: ~2.0M

### 3. U-Net3+
Full-scale skip connections aggregating features from all encoder-decoder levels.
- **Parameters**: ~2.4M

### 4. TransU-Net
Hybrid CNN-Transformer architecture with Vision Transformer (ViT) blocks at the bottleneck.
- `embed_dim=192`, `num_heads=4`, `num_transformer=3`
- **Parameters**: ~4.7M

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 10 (sanity check; paper uses 150) |
| **Batch size** | 8 |
| **Optimizer** | Adam |
| **Learning rate** | 0.001 |
| **Loss function** | Dice Loss |
| **Cross-validation** | 5-fold (KFold, shuffle=True, seed=42) |
| **Hardware** | 2× NVIDIA Tesla T4 (Kaggle) |
| **Strategy** | `tf.distribute.MirroredStrategy` |
| **Mixed precision** | Disabled (causes sigmoid saturation on small objects) |

## Results (10 Epochs)

| Model | Dice (Mean±Std) | IoU (Mean±Std) | Precision | Recall |
|-------|----------------|----------------|-----------|--------|
| **Modified U-Net** | 0.5799 ± 0.0126 | 0.4256 ± 0.0130 | 0.9610 ± 0.0781 | 0.7371 ± 0.1241 |
| **U-Net++** | 0.5361 ± 0.0187 | 0.3843 ± 0.0184 | 0.7727 ± 0.1200 | 0.9964 ± 0.0027 |
| **U-Net3+** | 0.5891 ± 0.0150 | 0.4337 ± 0.0143 | 1.0000 ± 0.0000 | 0.6623 ± 0.0417 |
| **TransU-Net** | 0.2850 ± 0.1551 | 0.1914 ± 0.1156 | 0.9916 ± 0.0169 | 0.8743 ± 0.2435 |

> **Note**: These results are from only **10 epochs** as a sanity check. The paper trains for **150 epochs** and reports Dice scores above 95% for all models. Full training is expected to significantly improve these numbers.

## Project Structure

```
├── kdss-modified-unet-10epoch.ipynb     # Modified U-Net — 5-fold CV
├── kdss-unetpp-unet3p-10epochs.ipynb    # U-Net++ & U-Net3+ — 5-fold CV
├── kssd-transunet-10epochs.ipynb        # TransU-Net — 5-fold CV
└── README.md
```

## How to Run

1. Open any notebook on [Kaggle](https://www.kaggle.com/)
2. Add dataset: `murillobouzon/kssd2025-kidney-stone-segmentation-dataset`
3. Set accelerator to **GPU T4 × 2**
4. Enable **Internet** (for `pip install keras-unet-collection`)
5. **Run All**

| Notebook | Estimated time |
|----------|---------------|
| Modified U-Net | ~30 min |
| U-Net++ & U-Net3+ | ~75 min |
| TransU-Net | ~60 min |

## Key Implementation Notes

- **No mixed precision**: Float16 causes sigmoid saturation for small kidney stone regions, resulting in the model predicting all-background during validation. All training uses float32.
- **Dice loss returns per-sample tensor**: Shape `(batch,)` so Keras handles distributed reduction correctly with `MirroredStrategy`.
- **TransU-Net Keras 3 patch**: The `keras_unet_collection` library uses `tf.reshape` which is incompatible with Keras 3 KerasTensors. We monkey-patch it to use `keras.ops.reshape`.
- **TransU-Net model saving**: Saved as weights-only (`.weights.h5`) because `keras_unet_collection` custom layers aren't registered with Keras 3 serialization.

## References

1. Bouzon, M., et al. (2025). *"KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models."* IEEE Access, Vol. 13. DOI: [10.1109/ACCESS.2025.3610027](https://doi.org/10.1109/ACCESS.2025.3610027)
2. Ronneberger, O., et al. (2015). *"U-Net: Convolutional Networks for Biomedical Image Segmentation."*
3. Zhou, Z., et al. (2018). *"UNet++: A Nested U-Net Architecture for Medical Image Segmentation."*
4. Huang, H., et al. (2020). *"UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation."*
5. Chen, J., et al. (2021). *"TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation."*
