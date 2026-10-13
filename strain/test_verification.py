"""Meaningful checks for deadline termination and frozen cached evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from strain.train import Deadline


def test_deadline_stops_batch_and_avoids_new_epoch():
    callback = Deadline(0)
    fake = SimpleNamespace(stop_training=False)
    callback.set_model(fake)
    callback.on_train_batch_end(0)
    assert fake.stop_training and callback.hit
    fake.stop_training = False
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0)
    assert fake.stop_training and callback.epoch_seconds[0] >= 0


def test_frozen_splits_and_cache():
    output = Path('strain/results/experiment256_20261013')
    if not (output/'splits.json').exists():
        import pytest
        pytest.skip('Measured preprocessing evidence not yet available')
    split = json.loads((output/'splits.json').read_text())
    audit = json.loads((output/'preprocessing.json').read_text())
    groups, ids = split['groups'], split['splits']
    for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
        assert not set(groups[a]) & set(groups[b])
        assert not set(ids[a]) & set(ids[b])
    assert sum(map(len,ids.values())) == 83
    for name in sum(ids.values(), []):
        pair = np.load(Path(audit['cache'])/(Path(name).stem+'.npz'))
        assert pair['image'].shape == pair['mask'].shape == (256,256)
        assert pair['image'].dtype == np.uint8
        assert np.isin(pair['mask'], [0,1]).all()
    # Vanished masks are documented, not silently dropped from the audit.
    assert len(audit['samples']) == 838
    assert set(audit['disappeared']) == {r['id'] for r in audit['samples'] if r['vanished']}


def test_supervisor_rejects_budget_above_one_hour(monkeypatch):
    import sys
    import pytest
    from strain.run_experiment import main
    monkeypatch.setattr(sys, 'argv', ['experiment', '--data', 'unused', '--budget', '3601'])
    with pytest.raises(ValueError, match='Budget'):
        main()
