"""Supervisor deadline includes imports, preprocessing, pilots, training and reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def report(output, started, status):
    from strain.data import save
    from strain.plot_history import plot
    plot(output)
    rows = []
    for name in ['modified_unet','unet3plus','unetpp','transunet']:
        path = output/name/'result.json'
        row = json.loads(path.read_text()) if path.exists() else dict(architecture=name,
            status='unavailable', test=None, reason='No completed result; inspect worker.log and errors.json')
        rows.append(row)
    elapsed = time.monotonic()-started
    save(output/'comparison.json', dict(models=rows, elapsed_seconds=elapsed,
        supervisor_status=status, under_60_minutes=elapsed < 3600,
        evaluation='Held-out test; global pooled pixel counts on nearest-neighbor 256 masks',
        original_resolution_metrics='Not comparable directly to resized masks; see resizing audit',
        uncertainty='Single tiny grouped split; no confidence interval or patient independence claim'))
    text = ['# Isolated 256 × 256 experiment', '', f'Elapsed: {elapsed:.2f} seconds. Status: {status}.', '',
        'Held-out test; threshold 0.5; all metrics from the same validation-selected checkpoint.', '',
        '| Model | Epochs | Dice | IoU | Precision | Recall | Status |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        m = (row.get('test') or {}).get('micro', {})
        values = [f"{m[k]:.4f}" if k in m else 'unavailable' for k in
            ['hard_mask_dice','hard_mask_iou','hard_mask_precision','hard_mask_recall']]
        text.append('| '+' | '.join([row['architecture'],str(row.get('epochs_completed','—')),*values,row['status']])+' |')
    text += ['', 'Tiny regions erased by nearest-neighbor resizing remain in all partitions. See preprocessing.json.',
        'Content groups are not patient IDs. The small test sample cannot establish clinical generalization.',
        'Dice/IoU are 1 for empty truth and prediction; precision/recall are 0 for zero denominators.',
        'The existing image illustrates legacy scores; these resized test scores use a different split and resolution.']
    (output/'summary.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def worker(args):
    from strain.data import prepare, save
    import tensorflow as tf
    from strain.train import pilot, train_model
    from strain.config import ARCHITECTURES
    import importlib.metadata
    import platform
    started = args.started
    deadline = started + args.budget - 50
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('GPU unavailable; no full training started')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    output = Path(args.output)
    save(output/'environment.json', dict(python=platform.python_version(),
        versions={name:importlib.metadata.version(name) for name in
            ['tensorflow','keras','numpy','scipy','Pillow','keras-unet-collection']},
        gpu_details=[tf.config.experimental.get_device_details(g) for g in gpus],
        build=tf.sysconfig.get_build_info(), seed=20261013, deterministic=True))
    save(output/'gpu.txt.json', dict(nvidia_smi=subprocess.check_output(
        ['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv'],text=True)))
    arrays, splits = prepare(args.data, output)
    save(output/'preview_ready.json', dict(status='Awaiting local visual inspection', elapsed=time.monotonic()-started))
    while not (output/'preview_verified.json').exists():
        if time.monotonic() > deadline-120:
            raise TimeoutError('Preview inspection not completed before deadline')
        time.sleep(1)
    timings, errors = {}, {}
    for name in ARCHITECTURES:
        folder = output/name
        folder.mkdir(exist_ok=True)
        try:
            timings[name] = pilot(name, arrays, folder, deadline)
        except Exception as error:
            errors[name] = repr(error)
            save(output/'errors.json', errors)
    # Timing first for every architecture. Separate training budgets reserve evaluation.
    for index, name in enumerate(ARCHITECTURES):
        if name not in timings:
            continue
        remaining = len([n for n in ARCHITECTURES[index:] if n in timings])
        end = min(deadline, time.monotonic() + (deadline-time.monotonic())/remaining)
        if end-time.monotonic() < 90:
            errors[name] = 'Insufficient measured time remaining'
            save(output/'errors.json', errors)
            continue
        try:
            train_model(name, arrays, splits, output/name,
                Path('strain/checkpoints')/output.name/(name+'.weights.h5'), end, timings[name])
        except Exception as error:
            errors[name] = repr(error)
            save(output/'errors.json', errors)
    save(output/'worker_complete.json', dict(elapsed_seconds=time.monotonic()-started, errors=errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='strain/results/run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--budget', type=float, default=3600)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--started', type=float)
    args = parser.parse_args()
    if not 120 <= args.budget <= 3600:
        raise ValueError('Budget must be between 120 and 3600 seconds')
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path('strain/results').resolve()):
        raise ValueError('Output must remain inside strain/results')
    args.output = str(output)
    if args.worker:
        worker(args)
        return
    started = args.started if args.started is not None else time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONHASHSEED='20261013', TF_NUM_INTRAOP_THREADS='2',
               TF_NUM_INTEROP_THREADS='2', MPLCONFIGDIR=str(output/'matplotlib'))
    command = [sys.executable, '-m', 'strain.run_experiment', '--worker', '--data', args.data,
        '--output', str(output), '--budget',str(args.budget), '--started',str(started)]
    with (output/'worker.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=max(1,args.budget-35-(time.monotonic()-started)))
            status = 'finished' if code == 0 else f'worker_exit_{code}'
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = 'hard_deadline_terminated'
    report(output, started, status)
    print((output/'summary.md').read_text())


if __name__ == '__main__':
    main()
