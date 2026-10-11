"""Seeds and machine-readable runtime provenance."""
import importlib.metadata
import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path
import numpy as np


def set_seeds(seed):
    # PYTHONHASHSEED must be set by the launching shell to affect hash randomization.
    random.seed(seed);np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def environment():
    import tensorflow as tf
    versions={}
    for name in ['tensorflow','keras','numpy','scipy','Pillow','scikit-learn','keras-unet-collection']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    try:
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    except (OSError,subprocess.CalledProcessError):git_commit=None;dirty=None
    return {'python':platform.python_version(),'platform':platform.platform(),'versions':versions,
            'tensorflow_gpu_devices':[d.name for d in tf.config.list_physical_devices('GPU')],
            'gpu_details':[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices('GPU')],
            'git_commit':git_commit,'git_dirty':dirty,'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'all_installed_packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'deterministic_ops':True}
