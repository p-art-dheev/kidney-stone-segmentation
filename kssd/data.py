"""Strict file pairing and non-resizing preprocessing for exported 8-bit CT images."""
from pathlib import Path
import numpy as np
from PIL import Image

EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def paired_files(root):
    root = Path(root)
    directories = [root / "image", root / "label"]
    if any(not p.is_dir() for p in directories):
        raise ValueError("Dataset must have image/ and label/ directories")
    listings = [{p.name: p for p in d.iterdir() if p.suffix.lower() in EXTENSIONS} for d in directories]
    for listing in listings:
        names = list(listing)
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError("Case-insensitive duplicate filenames")
        if len({Path(n).stem.casefold() for n in names}) != len(names):
            raise ValueError("Duplicate sample stems across image extensions")
    a, b = listings
    if set(a) != set(b):
        raise ValueError(f"Pair mismatch: images without masks={sorted(set(a)-set(b))}, masks without images={sorted(set(b)-set(a))}")
    if not a:
        raise ValueError("Dataset contains no image-mask pairs")
    return [(name, a[name], b[name]) for name in sorted(a)]


def read_grayscale(path):
    with Image.open(path) as image:
        if image.mode == "P":
            palette = np.asarray(image.getpalette(), dtype=np.uint8).reshape(-1, 3)
            if not np.all(palette[:, 0] == palette[:, 1]) or not np.all(palette[:, 0] == palette[:, 2]):
                raise ValueError(f"Non-grayscale palette: {path}")
            image = image.convert("L")
        raw = np.asarray(image)
        if raw.dtype != np.uint8:
            raise ValueError(f"Expected exported uint8 image; got {raw.dtype}: {path}")
        if raw.ndim == 3:
            if raw.shape[-1] not in (3, 4) or not np.array_equal(raw[..., 0], raw[..., 1]) or not np.array_equal(raw[..., 0], raw[..., 2]):
                raise ValueError(f"Non-grayscale color image: {path}")
            raw = raw[..., 0]
        if raw.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image: {path}")
        return raw.copy()


def load_pair(image_path, mask_path, mask_threshold=127):
    image, mask = read_grayscale(image_path), read_grayscale(mask_path)
    if image.shape != mask.shape:
        raise ValueError("Image and mask dimensions differ")
    if not 0 <= mask_threshold <= 254:
        raise ValueError("Mask threshold must lie in [0,254]")
    return image[..., None].astype(np.float32)/255.0, (mask > mask_threshold)[..., None].astype(np.float32)
