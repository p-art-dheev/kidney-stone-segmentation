"""CLI dataset integrity, decoded-image duplicates, perceptual candidates and overlays."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from kssd.data import paired_files, read_grayscale, load_pair


def digest(array):
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def validate(root, expected_count=838, expected_size=512, mask_threshold=127, output=None):
    pairs = paired_files(root)
    errors = []
    if len(pairs) != expected_count:
        errors.append(f"Expected {expected_count} pairs, found {len(pairs)}")
    samples, hash_groups, thumbnails = [], defaultdict(list), []
    for name, image_path, mask_path in pairs:
        image, mask = read_grayscale(image_path), read_grayscale(mask_path)
        if image.shape != mask.shape or image.shape != (expected_size, expected_size):
            errors.append(f"Unexpected dimensions: {name}: {image.shape}/{mask.shape}")
        values, counts = np.unique(mask, return_counts=True)
        area = int(np.count_nonzero(mask > mask_threshold))
        image_hash, mask_hash = digest(image), digest(mask)
        hash_groups[image_hash].append(name)
        thumbnails.append(np.asarray(Image.fromarray(image).resize((32,32)), dtype=np.float32)/255)
        with Image.open(image_path) as opened:
            metadata_keys = sorted(str(k) for k in opened.info)
        samples.append({"id": name, "image_sha256": image_hash, "mask_sha256": mask_hash,
                        "binary_mask_sha256": digest((mask > mask_threshold).astype(np.uint8)),
                        "shape": list(image.shape), "image_min": int(image.min()), "image_max": int(image.max()),
                        "mask_values": {str(v): int(c) for v,c in zip(values,counts)}, "stone_area": area,
                        "area_gt0": int(np.count_nonzero(mask)), "area_gt127": int(np.count_nonzero(mask > 127)),
                        "metadata_keys": metadata_keys})
    duplicates = [group for group in hash_groups.values() if len(group) > 1]
    by_id = {s['id']: s for s in samples}
    for group in duplicates:
        if len({by_id[n]['binary_mask_sha256'] for n in group}) > 1:
            errors.append(f"Identical images have conflicting labels: {group}")
    # Only candidate screening; thumbnail similarity cannot establish patient identity.
    near = []
    for i, thumb in enumerate(thumbnails):
        for j in range(i):
            mae = float(np.mean(np.abs(thumb-thumbnails[j])))
            if mae <= 0.01 and samples[i]['image_sha256'] != samples[j]['image_sha256']:
                near.append({"ids": [samples[j]['id'], samples[i]['id']], "thumbnail_mae": mae})
    parents = {s['id']: s['id'] for s in samples}
    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name
    for group in duplicates + [item['ids'] for item in near]:
        for name in group[1:]:
            parents[find(name)] = find(group[0])
    clusters = defaultdict(list)
    for name in parents:
        clusters[find(name)].append(name)
    groups = sorted([sorted(g) for g in clusters.values()], key=lambda g:g[0])
    excluded = {name for group in duplicates for name in group[1:]}
    for i, group in enumerate(groups):
        for name in group:
            by_id[name]['similarity_group'] = f'content-{i:03d}'
            by_id[name]['eligible'] = name not in excluded
    report = {"schema_version": 1, "count": len(samples), "expected_count": expected_count,
              "mask_threshold": mask_threshold, "normalization": "uint8 / 255; no resizing",
              "empty_masks": sum(s['stone_area']==0 for s in samples),
              "small_masks_le10_pixels": sum(0<s['stone_area']<=10 for s in samples),
              "mask_value_histogram": dict(sum((Counter(s['mask_values']) for s in samples), Counter())),
              "exact_duplicate_groups": duplicates, "near_duplicate_candidates": near,
              "similarity_group_count": len(groups), "eligible_count": len(samples)-len(excluded),
              "excluded_exact_duplicate_ids": sorted(excluded),
              "grouping_policy": "Connected components of exact hashes and 32x32 Pillow bicubic thumbnail mean absolute difference <=0.01 (uint8/255).",
              "grouping_limitation": "Content groups are conservative similarity clusters, NOT patient/scan IDs. May merge distinct patients or miss related slices; patient-independent generalization is unverified.",
              "samples": samples, "errors": errors, "passed": not errors}
    if output:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        (output/'dataset_manifest.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
        overlays(root, report, output/'overlays')
    return report


def overlays(root, report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(report['samples'], key=lambda s:(s['stone_area'],s['id']))
    chosen = sorted(set([0, len(ordered)//4, len(ordered)//2, 3*len(ordered)//4, len(ordered)-1]))
    for i in chosen:
        sample = ordered[i]; name=sample['id']
        image, mask = load_pair(Path(root)/'image'/name,Path(root)/'label'/name,report['mask_threshold'])
        fig, axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(image[...,0],cmap='gray',vmin=0,vmax=1)
        axes[1].imshow(mask[...,0],cmap='gray',vmin=0,vmax=1)
        axes[2].imshow(image[...,0],cmap='gray',vmin=0,vmax=1)
        axes[2].imshow(np.ma.masked_where(mask[...,0]==0,mask[...,0]),cmap='autumn',alpha=.65,vmin=0,vmax=1)
        for ax,title in zip(axes,['CT image','Binary ground truth','Overlay']): ax.set_title(title); ax.axis('off')
        fig.suptitle(f"{name} | area={sample['stone_area']} pixels | mask > {report['mask_threshold']}")
        fig.tight_layout(); fig.savefig(output/f'{Path(name).stem}.png',dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--expected-count',type=int,default=838); parser.add_argument('--expected-size',type=int,default=512)
    parser.add_argument('--mask-threshold',type=int,default=127)
    args=parser.parse_args(); report=validate(args.data,args.expected_count,args.expected_size,args.mask_threshold,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ['samples','near_duplicate_candidates','mask_value_histogram']},indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
