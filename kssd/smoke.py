"""Real 512x512 training/inference/checkpoint smoke checks; not CV evidence."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from kssd.augmentation import augment_pair
from kssd.data import load_pair
from kssd.models import ARCHITECTURES,build_model
from kssd.reproducibility import environment,set_seeds
from kssd.tf_metrics import dice_loss,make_metrics
from kssd.train import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--require-gpu',action='store_true')
    args=parser.parse_args()
    gpus=tf.config.list_physical_devices('GPU')
    if args.require_gpu and not gpus:raise RuntimeError('GPU required but not visible')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    splits=json.loads(Path('reports/splits.json').read_text());ids=splits['folds'][0]['train'][:2]
    pairs=[augment_pair(*load_pair(Path(args.data)/'image'/n,Path(args.data)/'label'/n),seed=43+i) for i,n in enumerate(ids)]
    x=np.stack([p[0] for p in pairs]);y=np.stack([p[1] for p in pairs]);results=[]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for architecture in ARCHITECTURES:
        tf.keras.backend.clear_session();gc.collect();set_seeds(43)
        model=build_model(architecture)
        model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss=dice_loss,metrics=make_metrics(),jit_compile=False)
        started=time.perf_counter();values=model.train_on_batch(x,y,return_dict=True);seconds=time.perf_counter()-started
        if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Non-finite training smoke result')
        prediction=model(x,training=False).numpy()
        path=output/f'{architecture}.weights.h5';model.save_weights(path);model.load_weights(path)
        np.testing.assert_allclose(model(x,training=False),prediction,atol=1e-7)
        result={'architecture':architecture,'parameters':model.count_params(),'training_batch_seconds':seconds,
                'input_shape':list(x.shape),'sample_ids':ids,'loss':float(values['loss']),
                'passed':True,'scope':'one augmented training batch and checkpoint round trip; NOT performance evidence'}
        results.append(result);write_json(output/'smoke.json',{'environment':environment(),'models':results,'complete':len(results)==4})
        print(json.dumps(result),flush=True)
        del model


if __name__=='__main__':main()
