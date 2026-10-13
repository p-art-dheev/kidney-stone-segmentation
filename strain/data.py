"""Audit original encodings, freeze grouped splits, and cache paired resizing once."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from kssd.data import paired_files, read_grayscale
from strain.config import SIZE, SEED, FRACTION


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(root, output):
    output = Path(output)
    manifest = json.loads(Path('reports/dataset/manifest.json').read_text())
    records = {s['id']: s for s in manifest['samples']}
    cache = output.parent.parent / 'cache' / output.name
    cache.mkdir(parents=True, exist_ok=False)
    audit = []
    for name, ip, mp in paired_files(root):
        x, raw = read_grayscale(ip), read_grayscale(mp)
        record = records[name]
        for array, key in [(x, 'image_sha256'), (raw, 'mask_sha256')]:
            if hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest() != record[key]:
                raise ValueError(f'Original pixels differ from frozen audit: {name}')
        if x.shape != raw.shape or not np.all((raw <= 9) | (raw >= 247)):
            raise ValueError(f'Unexpected dimensions or mask encoding: {name}')
        y = (raw > 127).astype(np.uint8)
        xr = np.asarray(Image.fromarray(x).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        yr = np.asarray(Image.fromarray(y).resize((SIZE, SIZE), Image.Resampling.NEAREST))
        components, count = label(y)
        kept = set(np.asarray(Image.fromarray(components.astype(np.int32)).resize(
            (SIZE, SIZE), Image.Resampling.NEAREST)).ravel()) - {0}
        # Round-trip disagreement is descriptive, never a replacement annotation.
        back = np.asarray(Image.fromarray(yr).resize(x.shape[::-1], Image.Resampling.NEAREST))
        intersect = int((y.astype(bool) & back.astype(bool)).sum())
        denom = int(y.sum() + back.sum())
        audit.append(dict(id=name, image_range=[int(x.min()), int(x.max())],
            mask_values=np.unique(raw).tolist(), original_shape=list(x.shape),
            original_area=int(y.sum()), resized_area=int(yr.sum()),
            vanished_components=int(count - len(kept)), vanished=bool(y.sum() and not yr.sum()),
            nearest_roundtrip_dice=2*intersect/denom if denom else 1.0))
        np.savez_compressed(cache / (Path(name).stem + '.npz'), image=xr, mask=yr)
    rng = np.random.default_rng(SEED)
    eligible = [s for s in manifest['samples'] if s['eligible']]
    groups = sorted({s['similarity_group'] for s in eligible})
    rng.shuffle(groups)
    # Partition groups before selecting images; all models receive identical IDs.
    partitions = dict(train=groups[:23], validation=groups[23:31], test=groups[31:])
    total = int(len(eligible) * FRACTION)
    targets = dict(train=total - 2*(total//5), validation=total//5, test=total//5)
    splits = {}
    for part, gs in partitions.items():
        pool = [s['id'] for s in eligible if s['similarity_group'] in gs]
        if len(pool) < targets[part]:
            raise ValueError('Insufficient group partition size')
        splits[part] = sorted(rng.choice(pool, targets[part], replace=False).tolist())
    assert not any(set(splits[a]) & set(splits[b]) for a, b in
                   [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    save(output / 'preprocessing.json', dict(samples=audit,
        disappeared=[r['id'] for r in audit if r['vanished']],
        components_lost=[r['id'] for r in audit if r['vanished_components']],
        normalization='Exported uint8 grayscale (not HU): bilinear resize then float32 / 255',
        mask_policy='Decode grayscale palette; >127; nearest-neighbor binary resize',
        cache=str(cache), original_files_modified=False))
    save(output / 'splits.json', dict(seed=SEED, eligible_count=len(eligible),
        fraction=len(sum(splits.values(), []))/len(eligible), splits=splits,
        groups=partitions, counts=targets, limitation=manifest['grouping_limitation'],
        sampling='Seeded uniform sampling inside fixed disjoint content-group partitions; no area filtering'))
    preview(root, cache, audit, output)
    arrays = {}
    for part, ids in splits.items():
        pairs = [np.load(cache / (Path(n).stem + '.npz')) for n in ids]
        arrays[part] = (np.stack([p['image'] for p in pairs])[..., None].astype('float32')/255,
                        np.stack([p['mask'] for p in pairs])[..., None].astype('float32'))
        assert arrays[part][0].shape[1:] == (256, 256, 1)
        assert np.isin(arrays[part][1], [0, 1]).all()
    return arrays, splits


def preview(root, cache, audit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ranked = sorted(audit, key=lambda r: r['original_area'])
    selected = [ranked[0], ranked[len(ranked)//2], ranked[-1]]
    vanished = next((r for r in ranked if r['vanished']), None)
    if vanished and vanished not in selected:
        selected.append(vanished)
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 4*len(selected)))
    for row, r in enumerate(selected):
        x = read_grayscale(Path(root)/'image'/r['id'])
        y = read_grayscale(Path(root)/'label'/r['id']) > 127
        pair = np.load(cache/(Path(r['id']).stem+'.npz'))
        for col, (image, mask) in enumerate([(x, y), (pair['image'], pair['mask'])]):
            ax = axes[row, col]
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='autumn', alpha=.8, vmin=0, vmax=1)
            ax.set_title(f"{r['id']} | {image.shape} | foreground={int(mask.sum())}")
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(output/'resize_preview.png', dpi=160)
    plt.close(fig)
