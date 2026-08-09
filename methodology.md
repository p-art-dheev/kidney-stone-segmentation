# Methodology: Replicating KSSD2025 Paper

## Paper Reference
**"KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models"**
Published in *IEEE Access*, 2025.

---

## 1. Problem Statement

Kidney stones are a common urological condition diagnosed via Computed Tomography (CT) scans. Manual segmentation is time-consuming and suffers from inter-observer variability. This paper introduces the KSSD2025 dataset and benchmarks four modified U-Net architectures for automatic kidney stone segmentation.

---

## 2. Dataset: KSSD2025

| Property | Details |
|---|---|
| **Total Images** | 838 axial CT slices |
| **Source** | Refined subset of "CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone" (Islam et al., 2022) |
| **Image Format** | TIF (grayscale, mode L) |
| **Mask Format** | TIF (binary, mode P) |
| **Original Resolution** | 512 × 512 pixels |
| **Classes** | 2 (Background = 0, Kidney Stone = 1) |

### 2.1 Annotation Process
The ground-truth masks were created using a semi-automatic approach:
1. **Thresholding**: Pixel intensity thresholding at 150 HU to isolate high-density structures
2. **Connected Component Filtering**: Regions larger than 300 pixels discarded to remove bones/non-stone tissues
3. **Manual Refinement**: Expert radiologists and urologists manually refined masks for clinical accuracy

---

## 3. Preprocessing Pipeline

### Step 1: Image Loading
- Load grayscale CT images and corresponding binary masks from TIF files
- Ensure image-mask filename correspondence

### Step 2: Resizing
- Resize all images and masks to **256 × 256** pixels
- Use bilinear interpolation for images, nearest-neighbor for masks (to preserve binary labels)

### Step 3: Normalization
- Normalize image pixel values to [0, 1] range by dividing by 255.0
- Masks are binarized: any pixel > 0 is set to 1 (stone), rest remain 0 (background)

### Step 4: Tensor Conversion
- Convert images to PyTorch tensors with shape `(1, 256, 256)` (single-channel grayscale)
- Convert masks to tensors with shape `(1, 256, 256)` (single-channel binary)

---

## 4. Data Augmentation

To improve model generalization and address the relatively small dataset size, the following augmentations are applied to training data only:

| Augmentation | Parameters |
|---|---|
| Horizontal Flip | p = 0.5 |
| Vertical Flip | p = 0.5 |
| Random Rotation | ±15 degrees |
| Random Affine | Scale 0.9–1.1, Translate ±10% |
| Brightness/Contrast | Factor ±0.2 |

> **Note**: All spatial augmentations are applied identically to both images and masks to maintain alignment. Intensity augmentations are applied only to images.

---

## 5. Data Splitting: 5-Fold Cross-Validation

- The 838 images are divided into **5 stratified folds**
- In each fold: **~670 images for training**, **~168 images for validation**
- Each model is trained 5 times (once per fold), and metrics are averaged
- This provides robust performance estimates with confidence intervals

```
Fold 1: Train [2,3,4,5] -> Val [1]
Fold 2: Train [1,3,4,5] -> Val [2]
Fold 3: Train [1,2,4,5] -> Val [3]
Fold 4: Train [1,2,3,5] -> Val [4]
Fold 5: Train [1,2,3,4] -> Val [5]
```

---

## 6. Model Architectures

### 6.1 U-Net (Baseline)
The standard U-Net architecture with:
- **Encoder** (contracting path): 4 downsampling blocks, each with two 3x3 convolutions + ReLU + BatchNorm, followed by 2x2 max pooling
- **Bottleneck**: Two 3x3 convolutions at the deepest level
- **Decoder** (expanding path): 4 upsampling blocks with 2x2 transposed convolutions + skip connections + two 3x3 convolutions
- **Output**: 1x1 convolution + Sigmoid activation
- **Filter progression**: 64 -> 128 -> 256 -> 512 -> 1024

### 6.2 U-Net++
Enhanced U-Net with **nested, dense skip pathways**:
- Redesigned skip connections with dense convolutional blocks between encoder and decoder
- Multiple intermediate nodes at each level form a dense connectivity pattern
- Each node receives feature maps from all preceding nodes at the same level AND below
- Deep supervision applied at multiple output layers
- This allows the model to capture multi-scale features more effectively

### 6.3 U-Net3+
U-Net with **full-scale skip connections**:
- Every decoder node receives feature maps from ALL encoder levels AND all previous decoder levels
- Combines fine-grained details (shallow layers) with coarse semantic information (deep layers)
- Uses classification-guided module (CGM) to handle false-positive reduction
- Each decoder level aggregates full-scale information, making it more robust for small objects like kidney stones

### 6.4 TransUNet
Hybrid **Transformer + U-Net** architecture:
- **Encoder**: CNN (first few layers for local features) -> Vision Transformer (ViT) for global context
- The image is divided into patches (16x16), embedded, and processed through Transformer layers (12 multi-head self-attention blocks)
- **Decoder**: Standard U-Net decoder with skip connections from CNN encoder layers
- Captures both local spatial details (CNN) and long-range dependencies (Transformer)
- Uses pre-trained ViT weights (optional, for faster convergence)

---

## 7. Loss Function

### Combined Binary Cross-Entropy + Dice Loss

```
L_total = L_BCE + L_Dice
```

**Binary Cross-Entropy (BCE)**:
```
L_BCE = -1/N * sum [y*log(y_hat) + (1-y)*log(1-y_hat)]
```
- Provides pixel-wise supervision
- Good for boundary precision

**Dice Loss**:
```
L_Dice = 1 - (2*|y intersection y_hat| + eps) / (|y| + |y_hat| + eps)
```
- Directly optimizes the Dice coefficient
- Handles class imbalance (stone pixels << background pixels)
- eps = 1e-7 for numerical stability

---

## 8. Training Configuration

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Batch Size** | 16 |
| **Epochs** | 150 |
| **Input Size** | 256 x 256 |
| **Weight Decay** | 1e-5 |
| **Early Stopping** | Patience = 20 epochs (based on validation Dice) |
| **Learning Rate Scheduler** | ReduceLROnPlateau (factor=0.5, patience=10) |

---

## 9. Evaluation Metrics

All metrics are computed per-image and then averaged across the validation set:

### 9.1 Dice Similarity Coefficient (DSC)
```
DSC = 2*TP / (2*TP + FP + FN)
```
Primary metric. Measures overlap between predicted and ground-truth masks. Range: [0, 1].

### 9.2 Intersection over Union (IoU / Jaccard Index)
```
IoU = TP / (TP + FP + FN)
```
Stricter overlap metric. Always <= DSC for the same prediction.

### 9.3 Pixel Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Percentage of correctly classified pixels. Can be misleadingly high due to class imbalance.

### 9.4 Precision
```
Precision = TP / (TP + FP)
```
Of all pixels predicted as stone, what fraction are truly stone.

### 9.5 Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
Of all actual stone pixels, what fraction are correctly detected.

---

## 10. Expected Results (from Paper)

| Model | Mean Dice (%) | Mean IoU (%) |
|---|---|---|
| U-Net | ~95-96 | ~91-93 |
| U-Net++ | ~96-97 | ~92-94 |
| U-Net3+ | ~96-97 | ~93-95 |
| TransUNet | ~95-97 | ~91-94 |

> All models achieved mean Dice scores above 95% with the modified U-Net architecture reaching 97.06% in cross-validation.

---

## 11. Workflow Summary

```
Load KSSD2025 Dataset (838 TIF images + masks)
    |
    v
Preprocessing (Resize 256x256, Normalize, Binarize)
    |
    v
5-Fold Cross-Validation Split
    |
    v
Data Augmentation (Flips, Rotation, Affine)
    |
    v
Train Models: U-Net | U-Net++ | U-Net3+ | TransUNet
    |
    v
Evaluate per Fold (Dice, IoU, Accuracy, Precision, Recall)
    |
    v
Average Metrics Across 5 Folds
    |
    v
Compare Models (Tables + Plots + Visualizations)
```
