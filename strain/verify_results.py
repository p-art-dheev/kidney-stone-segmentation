"""Independent count-based audit of saved final held-out evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from kssd.metrics import metrics_from_counts
from strain.config import ARCHITECTURES
from strain.data import save


def verify(output):
    output = Path(output)
    splits = json.loads((output/'splits.json').read_text())['splits']
    prep = json.loads((output/'preprocessing.json').read_text())
    records = []
    for name in ARCHITECTURES:
        folder = output/name
        result = json.loads((folder/'result.json').read_text())
        test = result['test']
        assert test is not None, f'No held-out result: {name}'
        assert [row['id'] for row in test['per_image']] == splits['test']
        total = dict(tp=0, fp=0, fn=0, tn=0)
        for row in test['per_image']:
            cache = Path(prep['cache'])/(Path(row['id']).stem+'.npz')
            mask = np.load(cache)['mask']
            assert row['tp']+row['fn'] == int(mask.sum())
            assert sum(row[k] for k in total) == 256*256
            computed = metrics_from_counts(**{k:row[k] for k in total})
            for key, value in computed.items():
                assert np.isclose(row[key],value,rtol=0,atol=1e-12)
            for key in total:
                total[key] += row[key]
        assert total == test['counts']
        for key, value in metrics_from_counts(**total).items():
            assert np.isclose(test['micro'][key],value,rtol=0,atol=1e-12)
            assert np.isclose(test['macro'][key],np.mean([r[key] for r in test['per_image']]),atol=1e-12)
        history = json.loads((folder/'history.json').read_text())
        best = int(np.argmax(history['val_hard_mask_dice']))+1
        assert result['best_validation_epoch'] == best
        assert result['epochs_completed'] == len(history['loss'])
        assert all(np.isfinite(values).all() for values in history.values())
        checkpoint = Path(result['checkpoint'])
        records.append(dict(architecture=name, passed=True,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            epochs=result['epochs_completed'], best_epoch=best, test_images=len(test['per_image'])))
    comparison = json.loads((output/'comparison.json').read_text())
    assert comparison['elapsed_seconds'] < 3600 and comparison['under_60_minutes']
    save(output/'independent_verification.json', dict(passed=True, models=records,
        shared_elapsed_seconds=comparison['elapsed_seconds'],
        checks=['exact held-out IDs', 'cached mask areas', 'total pixel counts',
            'per-image and pooled metrics', 'macro means', 'finite histories',
            'earliest best validation checkpoint', 'checkpoint hashes', 'shared one-hour budget']))
    print('All four final held-out evidence audits passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    verify(parser.parse_args().output)
