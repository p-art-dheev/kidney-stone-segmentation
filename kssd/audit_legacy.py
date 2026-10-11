"""Restore every historical archive without treating it as validated CV evidence."""
import argparse
import gc
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from kssd.data import load_pair
from kssd.models import build_model
from kssd.train import write_json

DIRECTORIES={'Modified_Unet':'modified_unet','UNet++':'unetpp','Unet 3+':'unet3plus','Trans_UNet':'transunet'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True)
    parser.add_argument('--output',default='reports/legacy_archive_audit.json')
    parser.add_argument('--inference',action='store_true',help='Also run one full-size prediction per architecture; may be slow on CPU.')
    args=parser.parse_args()
    Path('tmp').mkdir(exist_ok=True)
    image,_=load_pair(Path(args.data)/'image/1.tif',Path(args.data)/'label/1.tif');records=[]
    for directory,architecture in DIRECTORIES.items():
        for i,path in enumerate(sorted((Path('Results')/directory/'Model').glob('*.keras'))):
            tf.keras.backend.clear_session();gc.collect();model=build_model(architecture)
            with zipfile.ZipFile(path) as archive:
                metadata=json.loads(archive.read('metadata.json'))
                with tempfile.TemporaryDirectory(dir='tmp') as temporary:
                    weights=Path(temporary)/'model.weights.h5';weights.write_bytes(archive.read('model.weights.h5'))
                    model.load_weights(weights)
            if not all(np.isfinite(w).all() for w in model.get_weights()):raise ValueError(f'Non-finite archive weights: {path}')
            if args.inference and i==0:
                pred=model(image[None],training=False).numpy()
                if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1:raise ValueError('Invalid restored inference')
            records.append({'path':path.as_posix(),'architecture':architecture,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'metadata':metadata,'parameters':model.count_params(),'weights_restore_passed':True,
                            'full_size_inference_checked':args.inference and i==0})
            print(f'Restored {path}',flush=True);del model
    if len(records)!=20:raise ValueError('Expected 20 historical archives')
    write_json(args.output,{'scope':'Archive compatibility only. Legacy final-epoch checkpoints have no verified nested-CV provenance.',
                           'archives':records,'weights_restoration_passed':True,'performance_claim':None,
                           'inference_scope':'One archive per architecture' if args.inference else 'Not checked by this run'})


if __name__=='__main__':main()
