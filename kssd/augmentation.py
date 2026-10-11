"""Small paired affine transforms; image-only Gaussian noise, train-only mapping."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import affine_transform


@dataclass(frozen=True)
class AugmentationParameters:
    rotation_degrees: float=2.5
    shift_fraction: float=.0075
    zoom_fraction: float=.0075
    noise_max_std: float=.0075
    horizontal_flip_probability: float=.5


def augment_pair(image,mask,seed,parameters=AugmentationParameters()):
    rng=np.random.default_rng(seed)
    if image.shape!=mask.shape or image.ndim!=3 or image.shape[-1]!=1:
        raise ValueError('Paired grayscale arrays must have matching HxWx1 shape')
    if not np.isin(mask,[0,1]).all():raise ValueError('Augmentation mask must be binary')
    if min(parameters.rotation_degrees,parameters.shift_fraction,parameters.zoom_fraction,parameters.noise_max_std)<0:
        raise ValueError('Augmentation ranges cannot be negative')
    if parameters.zoom_fraction>=1 or not 0<=parameters.horizontal_flip_probability<=1:
        raise ValueError('Invalid zoom/flip parameters')
    theta=np.deg2rad(rng.uniform(-parameters.rotation_degrees,parameters.rotation_degrees))
    zoom=rng.uniform(1-parameters.zoom_fraction,1+parameters.zoom_fraction)
    shift=rng.uniform(-parameters.shift_fraction,parameters.shift_fraction,2)*np.asarray(image.shape[:2])
    flip=-1 if rng.random()<parameters.horizontal_flip_probability else 1
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    inverse=np.linalg.inv(zoom*rotation@np.diag([1,flip]))
    center=(np.asarray(image.shape[:2])-1)/2
    offset=center-inverse@(center+shift)
    x=affine_transform(image[...,0],inverse,offset,order=1,mode='constant',cval=0,prefilter=False)
    y=affine_transform(mask[...,0],inverse,offset,order=0,mode='constant',cval=0,prefilter=False)
    sigma=rng.uniform(0,parameters.noise_max_std)
    if sigma:x=x+rng.normal(0,sigma,x.shape)
    return np.clip(x,0,1)[...,None].astype(np.float32),(y>0.5)[...,None].astype(np.float32)


def augment_dataset(ds,seed,parameters=AugmentationParameters()):
    import tensorflow as tf
    def mapping(index,pair):
        x,y=pair
        def numpy_augment(image,mask,position):
            return augment_pair(image,mask,np.random.SeedSequence([seed,int(position)]),parameters)
        image,mask=tf.numpy_function(numpy_augment,[x,y,index],[tf.float32,tf.float32])
        image.set_shape(x.shape);mask.set_shape(y.shape)
        return image,mask
    # Enumeration after seeded shuffling maps deterministic transforms to shuffled images.
    # Shuffling changes image/position associations each epoch, so transforms vary per image.
    return ds.enumerate().map(mapping,num_parallel_calls=2,deterministic=True)
