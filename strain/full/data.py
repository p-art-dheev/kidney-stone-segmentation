import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from kssd.data import paired_files,read_grayscale
from strain.data import save

BASE=Path('strain/results/experiment256_final_20261013')


def prepare(root,output):
    manifest=json.loads(Path('reports/dataset/manifest.json').read_text())
    previous=json.loads((BASE/'splits.json').read_text())
    audit=json.loads((BASE/'preprocessing.json').read_text())
    records={s['id']:s for s in manifest['samples']}
    source_check=[]
    for name,ip,mp in paired_files(root):
        x,y=read_grayscale(ip),read_grayscale(mp)
        for raw,key in [(x,'image_sha256'),(y,'mask_sha256')]:
            assert hashlib.sha256(str(raw.shape).encode()+raw.tobytes()).hexdigest()==records[name][key]
        assert x.shape==y.shape==(512,512) and np.all((y<=9)|(y>=247))
        cached=np.load(Path(audit['cache'])/(Path(name).stem+'.npz'))
        expected_x=np.asarray(Image.fromarray(x).resize((256,256),Image.Resampling.BILINEAR))
        expected_y=np.asarray(Image.fromarray((y>127).astype('uint8')).resize((256,256),Image.Resampling.NEAREST))
        assert np.array_equal(cached['image'],expected_x) and np.array_equal(cached['mask'],expected_y)
        source_check.append(name)
    splits={p:sorted(s['id'] for s in manifest['samples'] if s['eligible'] and
        s['similarity_group'] in previous['groups'][p]) for p in ['train','validation','test']}
    assert set(sum(splits.values(),[]))=={s['id'] for s in manifest['samples'] if s['eligible']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(splits[a])&set(splits[b])
        assert not set(previous['groups'][a])&set(previous['groups'][b])
    for p in splits:
        assert set(previous['splits'][p])<=set(splits[p])
    save(output/'splits.json',dict(splits=splits,groups=previous['groups'],
        counts={p:len(ids) for p,ids in splits.items()},eligible_count=manifest['eligible_count'],
        dataset_fraction=1.0,training_partition_fraction_used=1.0,seed=previous['seed'],
        policy='Expand previous frozen content-group assignments to every eligible image',
        limitation=manifest['grouping_limitation'], warm_start_leakage_check_passed=True))
    save(output/'preprocessing.json',dict(cache=audit['cache'],
        checked_original_pairs=len(source_check),normalization='Pillow bilinear uint8 resize; float32 /255 (not HU)',
        mask_policy='grayscale palette decoding; >127; nearest-neighbor binary resize',
        all_cached_pairs_match_originals=True, lost_components=audit['components_lost'],
        vanished_masks=audit['disappeared'],originals_modified=False))
    # Reuse the byte-identical preview already inspected, not unverified new overlays.
    preview=BASE/'resize_preview.png'
    assert json.loads((BASE/'preview_verified.json').read_text())['verified']
    (output/'resize_preview.png').write_bytes(preview.read_bytes())
    save(output/'preview_verified.json',dict(verified=True, reused_from=str(BASE),
        sha256=hashlib.sha256(preview.read_bytes()).hexdigest(),source_cache_reverified=True))
    # Deliberately do not materialize test images/masks during fitting or tuning.
    arrays={p:load_partition(splits[p],audit['cache']) for p in ['train','validation']}
    return arrays,splits,audit['cache']


def load_partition(ids,cache):
    pairs=[np.load(Path(cache)/(Path(n).stem+'.npz')) for n in ids]
    return (np.stack([p['image'] for p in pairs])[...,None].astype('float32')/255,
            np.stack([p['mask'] for p in pairs])[...,None].astype('float32'))
