"""Seeds and machine-readable runtime provenance."""
import importlib.metadata
import os
import platform
import random
import subprocess
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
    try:git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):git_commit=None
    return {'python':platform.python_version(),'platform':platform.platform(),'versions':versions,
            'tensorflow_gpu_devices':[d.name for d in tf.config.list_physical_devices('GPU')],
            'gpu_details':[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices('GPU')],
            'git_commit':git_commit,'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
            'deterministic_ops':True}
