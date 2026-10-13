"""One shared diagnostic/tuning deadline; test loading follows a sealed selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def protections():
    paths = list(Path('strain/checkpoints/experiment256_final_20261013').glob('*.weights.h5'))
    paths += [Path('kssd/models.py'),Path('strain/train.py'),Path('strain/data.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def worker(args):
    import tensorflow as tf
    from strain.data import save
    from strain.targeted.diagnose import diagnose,load_cached,MODELS
    output=Path(args.output)
    gpu=tf.config.list_physical_devices('GPU')
    if not gpu:
        raise RuntimeError('GPU required for targeted experiments')
    for device in gpu:
        tf.config.experimental.set_memory_growth(device,True)
    before=protections()
    save(output/'protected_files.json',before)
    save(output/'environment.json',dict(tensorflow=tf.__version__,keras=tf.keras.__version__,
        gpu=[tf.config.experimental.get_device_details(g) for g in gpu], precision='float32',
        device_selection='GPU:0; sequential',nvidia_smi=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,memory.free,driver_version','--format=csv'],text=True)))
    arrays,split=load_cached()
    for name in MODELS:
        diagnose(name,arrays,split,args.data,output/name)
    save(output/'diagnosis_complete.json',dict(elapsed_seconds=time.monotonic()-args.started,
        tuning_test_data_loaded=False))
    while not (output/'plan.json').exists():
        if time.monotonic()>args.started+args.budget-120:
            raise TimeoutError('No evidence-based tuning plan before deadline')
        time.sleep(1)
    # Imported after diagnosis and plan review, allowing targeted implementation.
    from strain.targeted.optimize import optimize
    plan=json.loads((output/'plan.json').read_text())
    optimize(plan,arrays,split,output,args.started+args.budget-60)
    after=protections()
    assert before==after, 'A protected original file changed'
    save(output/'protection_verification.json',dict(passed=True,before=before,after=after))
    save(output/'worker_complete.json',dict(elapsed_seconds=time.monotonic()-args.started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--budget',type=float,default=3600)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--started',type=float)
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if not output.is_relative_to(Path('strain/diagnostics').resolve()):
        raise ValueError('Targeted evidence must remain inside strain/diagnostics')
    if not 120<=args.budget<=3600:
        raise ValueError('Shared budget must be 120–3600 seconds')
    args.output=str(output)
    if args.worker:
        worker(args);return
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONHASHSEED='20261013',TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='2',MPLCONFIGDIR=str(output/'matplotlib'))
    command=[sys.executable,'-m','strain.targeted.run','--worker','--data',args.data,
        '--output',str(output),'--budget',str(args.budget),'--started',str(started)]
    with (output/'worker.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            code=process.wait(timeout=args.budget-30)
            status='finished' if code==0 else f'worker_exit_{code}'
        except subprocess.TimeoutExpired:
            process.kill();process.wait();status='hard_deadline_terminated'
    from strain.data import save
    save(output/'execution.json',dict(status=status,elapsed_seconds=time.monotonic()-started,
        shared_budget_seconds=args.budget))
    print(json.dumps(dict(status=status,elapsed_seconds=time.monotonic()-started,output=str(output))))


if __name__=='__main__':
    main()
