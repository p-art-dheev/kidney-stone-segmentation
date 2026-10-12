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
    parser.add_argument('--architectures',nargs='+',choices=ARCHITECTURES,default=ARCHITECTURES)
    parser.add_argument('--loss',choices=['dice','focal_tversky'],default='dice')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--training-steps',type=int,default=1)
    parser.add_argument('--precision',choices=['float32','mixed_float16'],default='float32')
    args=parser.parse_args()
    if args.batch_size<1 or args.training_steps<1:raise ValueError('Batch size and training steps must be positive')
    gpus=tf.config.list_physical_devices('GPU')
    if args.require_gpu and not gpus:raise RuntimeError('GPU required but not visible')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    splits=json.loads(Path('reports/splits.json').read_text());ids=splits['folds'][0]['train'][:args.batch_size]
    if len(ids)!=args.batch_size:raise ValueError('Not enough training images for requested batch size')
    pairs=[augment_pair(*load_pair(Path(args.data)/'image'/n,Path(args.data)/'label'/n),seed=43+i) for i,n in enumerate(ids)]
    x=np.stack([p[0] for p in pairs]);y=np.stack([p[1] for p in pairs]);results=[]
    output=Path(args.output)
    if output.exists():raise ValueError('Smoke output exists; choose a new directory')
    output.mkdir(parents=True)
    if args.loss=='dice':loss=dice_loss
    else:
        from kssd.losses import FocalTverskyLoss
        loss=FocalTverskyLoss()
    for architecture in args.architectures:
        tf.keras.backend.clear_session();gc.collect();set_seeds(43)
        tf.keras.mixed_precision.set_global_policy(args.precision)
        if gpus:tf.config.experimental.reset_memory_stats('GPU:0')
        model=build_model(architecture)
        model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss=loss,metrics=make_metrics(),jit_compile=False)
        started=time.perf_counter();values=model.train_on_batch(x,y,return_dict=True);seconds=time.perf_counter()-started
        if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Non-finite training smoke result')
        extra_seconds=[]
        for _ in range(args.training_steps-1):
            started=time.perf_counter();values=model.train_on_batch(x,y,return_dict=True);extra_seconds.append(time.perf_counter()-started)
            if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Non-finite repeated training result')
        prediction=model(x,training=False).numpy()
        if prediction.dtype!=np.float32:raise RuntimeError('Output probabilities must remain float32')
        if prediction.shape!=y.shape or not np.all(np.isfinite(prediction)) or np.any((prediction<0)|(prediction>1)):
            raise RuntimeError('Invalid prediction shape or probabilities in smoke check')
        path=output/f'{architecture}.weights.h5';model.save_weights(path)
        model.set_weights([np.zeros_like(w) for w in model.get_weights()]);model.load_weights(path)
        np.testing.assert_allclose(model(x,training=False),prediction,atol=1e-7,equal_nan=False)
        result={'architecture':architecture,'parameters':model.count_params(),'training_batch_seconds':seconds,
                'extra_training_batch_seconds':extra_seconds,'training_steps':args.training_steps,
                'precision':args.precision,'optimizer_class':type(model.optimizer).__name__,
                'optimizer_iterations':int(model.optimizer.iterations.numpy()),
                'gpu_peak_allocated_bytes':tf.config.experimental.get_memory_info('GPU:0')['peak'] if gpus else None,
                'input_shape':list(x.shape),'sample_ids':ids,'seed':43,'loss':float(values['loss']),
                'loss_function':args.loss,'loss_parameters':loss.get_config() if args.loss=='focal_tversky' else {'smooth':1e-6,'reduction':'per image then Keras batch mean'},
                'passed':True,'scope':f'{args.training_steps} optimizer updates on one fixed augmented batch and checkpoint round trip; NOT CV performance evidence'}
        results.append(result);write_json(output/'smoke.json',{'environment':environment(),'models':results,
                                                            'requested_architectures':args.architectures,'loss_function':args.loss,
                                                            'complete':len(results)==len(args.architectures)})
        print(json.dumps(result),flush=True)
        del model


if __name__=='__main__':main()
