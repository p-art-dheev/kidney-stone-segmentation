# Dataset audit and preprocessing

Official source: [KSSD2025 on Kaggle](https://www.kaggle.com/datasets/murillobouzon/kssd2025-kidney-stone-segmentation-dataset), downloaded 2026-10-11.
Archive SHA256: `f9a672840d1a93e3dd3583c129e2ed9ac17e054460a58ce6c9596b0aa8417fb6`.
Archive CRC check passed; 1,676 files, 305,376,428 uncompressed bytes.
Citation: Bouzon et al., IEEE Access (2025), DOI 10.1109/ACCESS.2025.3610027;
source imagery: Islam et al., Scientific Reports (2022). Respect source attribution/licensing;
downloaded dataset files are local ignored assets, not redistributed here.

## Reproduce

Download/unzip the official archive into `data/kssd2025/` (the archive contains another `data/`).
Then from the repository root:

```sh
python -m kssd.validate_data --data data/kssd2025/data --output runs/dataset-audit --mask-threshold 127
python -m pytest tests/test_data.py -q --basetemp=tmp/data-tests
```

Create the `tmp/` parent first if absent. The committed `reports/dataset/manifest.json`
contains reproducible decoded-pixel hashes, filenames, intensities, areas, eligibility,
duplicate pairs, similarity candidates and group assignments. Five overlays are selected
deterministically at mask-area ranks 0, 25%, 50%, 75% and maximum, without performance selection.

## Measured findings

- Exactly 838 matched TIFF pairs; all 512x512 uint8 grayscale; no missing pairs, invalid values or dimension mismatches. No resizing is needed. Images normalize by division by 255; these exported intensities are **not Hounsfield units**.
- Indexed masks have grayscale palettes. Preprocessing decodes the palette before binarization. All mask intensities lie in 0-9 or 247-255, with no midrange values. `mask >127` consistently separates the two bands. Original pixels are preserved on disk.
- Legacy `mask >0` yields 313,149 foreground pixels versus 138,922 with the corrected policy. Faint background artifacts were incorrectly included by the old notebooks. The threshold choice is supported by the observed gap; any integer threshold 9-246 produces the same binary mask for this version.
- No empty masks. 50 masks have 1-10 foreground pixels; mask areas have min/Q1/median/Q3/max = 3/38.25/101.5/231/755 pixels. These are descriptive pixel areas, not clinical stone-size classes.
- Seven decoded-image duplicate pairs: 729/735, 730/736, 731/737, 732/738, 733/739, 734/740, 993/994. Their raw mask hashes differ in faint intensities, but all corresponding binary masks are identical. Canonical lexicographically first members remain eligible; seven redundant copies are excluded in the manifest, leaving **831 unique samples**. No source files are removed.
- 3,326 thumbnail-similarity candidate pairs. Exact duplicates plus candidate edges form 39 content clusters (largest 173 images). The rule is fixed before training: 32x32 Pillow bicubic thumbnails of uint8/255 images, mean absolute difference <=0.01, connected components. Use these clusters in both outer and inner group splits; retain exact duplicates in one cluster even when one is excluded.
- Image metadata contains compression/dpi/resolution only; no patient or scan mapping is supplied. Filenames are not patient IDs. The paper also says identifying metadata was removed before export.

## Leakage limitations

Content clusters are **not patient IDs**. Similarity chaining can merge different exams;
related slices may also evade this heuristic. Grouped evaluation is a conservative
alternative to random image splitting, but cannot support patient-independent or
external-hospital generalization claims. The largest clusters may produce imbalanced
fold sizes; exact sizes are recorded by the split utility. The corrected protocol
differs from legacy image-level folds and from an exact paper reproduction.

Representative overlays were visually inspected for image/mask alignment and clean,
binary foreground. This is an engineering check, not expert re-annotation or clinical validation.
