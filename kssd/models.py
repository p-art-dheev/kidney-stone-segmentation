"""Architecture builders matching the legacy notebook settings, without source exec."""
import tensorflow as tf
from tensorflow.keras import layers, Model

FILTERS=[16,32,64,128,256]
BASELINE_ARCHITECTURES=['modified_unet','unetpp','unet3plus','transunet']
ARCHITECTURES=BASELINE_ARCHITECTURES+['attention_unet']


def conv_block(x, filters, dropout=0):
    x=layers.Conv2D(filters,3,padding='same',use_bias=False)(x)
    x=layers.BatchNormalization()(x);x=layers.ReLU()(x)
    if dropout:x=layers.Dropout(dropout)(x)
    x=layers.Conv2D(filters,3,padding='same',use_bias=False)(x)
    return layers.ReLU()(layers.BatchNormalization()(x))


def modified_unet(input_shape):
    inputs=layers.Input(input_shape);x=inputs;skips=[]
    for filters,dropout in zip(FILTERS[:-1],[.1,.2,.3,.3]):
        x=conv_block(x,filters,dropout);skips.append(x);x=layers.MaxPooling2D(2)(x)
    x=conv_block(x,256,.3)
    for filters,skip in zip(FILTERS[-2::-1],skips[::-1]):
        x=layers.Conv2DTranspose(filters,2,strides=2,padding='same')(x)
        x=conv_block(layers.Concatenate()([x,skip]),filters)
    return Model(inputs,layers.Conv2D(1,1,activation='sigmoid',dtype='float32')(x),name='Modified_UNet')


def attention_gate(skip,decoder,channels,name):
    """Additive spatial gate at the upsampled decoder resolution (Oktay et al.)."""
    if skip.shape[1:3]!=decoder.shape[1:3]:raise ValueError('Gate spatial shapes must match')
    projected_skip=layers.Conv2D(channels,1,padding='same',name=f'{name}_skip')(skip)
    projected_decoder=layers.Conv2D(channels,1,padding='same',name=f'{name}_decoder')(decoder)
    score=layers.ReLU(name=f'{name}_relu')(layers.Add()([projected_skip,projected_decoder]))
    score=layers.Conv2D(1,1,padding='same',name=f'{name}_score')(score)
    coefficient=layers.Activation('sigmoid',name=f'{name}_coefficient')(score)
    return layers.Multiply(name=f'{name}_gated_skip')([skip,coefficient])


def attention_unet(input_shape):
    # Preserve Modified U-Net's encoder, decoder, dropout and convolution blocks.
    inputs=layers.Input(input_shape);x=inputs;skips=[]
    for filters,dropout in zip(FILTERS[:-1],[.1,.2,.3,.3]):
        x=conv_block(x,filters,dropout);skips.append(x);x=layers.MaxPooling2D(2)(x)
    x=conv_block(x,256,.3)
    for level,(filters,skip) in enumerate(zip(FILTERS[-2::-1],skips[::-1])):
        x=layers.Conv2DTranspose(filters,2,strides=2,padding='same')(x)
        skip=attention_gate(skip,x,max(filters//2,1),f'attention_{level}')
        x=conv_block(layers.Concatenate()([x,skip]),filters)
    return Model(inputs,layers.Conv2D(1,1,activation='sigmoid',dtype='float32')(x),name='Attention_UNet')


def transunet(input_shape):
    # Same no-backbone, patch-size-1 topology as keras-unet-collection 0.1.13.
    # Keras Reshape replaces the illegal symbolic tf.reshape; library source is untouched.
    from keras_unet_collection.layer_utils import CONV_stack, CONV_output
    from keras_unet_collection._model_unet_2d import UNET_left, UNET_right
    from keras_unet_collection._model_transunet_2d import ViT_block
    from keras_unet_collection.transformer_layers import patch_embedding
    inputs=layers.Input(input_shape);name='TransUNet';skips=[]
    x=CONV_stack(inputs,FILTERS[0],stack_num=2,activation='ReLU',batch_norm=True,name=f'{name}_down0')
    skips.append(x)
    for i,filters in enumerate(FILTERS[1:],1):
        x=UNET_left(x,filters,stack_num=2,activation='ReLU',pool=True,batch_norm=True,name=f'{name}_down{i}')
        skips.append(x)
    x=layers.Conv2D(FILTERS[-1],1,padding='valid',use_bias=False,name=f'{name}_conv_trans_before')(x)
    size=input_shape[0]//16
    x=layers.Reshape((size*size,FILTERS[-1]),name='patch_extract_reshape')(x)
    x=patch_embedding(size*size,192)(x)
    for i in range(3):x=ViT_block(x,4,192,[384,192],activation='GELU',name=f'{name}_ViT_{i}')
    x=layers.Reshape((size,size,192),name='token_grid')(x)
    x=layers.Conv2D(FILTERS[-1],1,padding='valid',use_bias=False,name=f'{name}_conv_trans_after')(x)
    for i,(skip,filters) in enumerate(zip(skips[-2::-1],FILTERS[-2::-1])):
        x=UNET_right(x,[skip],filters,stack_num=2,activation='ReLU',unpool=True,batch_norm=True,name=f'{name}_up{i}')
    x=CONV_output(x,1,kernel_size=1,activation=None,name=f'{name}_output')
    x=layers.Activation('sigmoid',dtype='float32',name=f'{name}_output_activation')(x)
    return Model(inputs,x,name='TransUNet_model')


def build_model(architecture, input_shape=(512,512,1)):
    if input_shape[0]!=input_shape[1] or input_shape[0]%16 or input_shape[-1]!=1:
        raise ValueError('Require square, grayscale inputs divisible by 16')
    if architecture=='modified_unet':return modified_unet(input_shape)
    if architecture=='attention_unet':return attention_unet(input_shape)
    if architecture=='transunet':return transunet(input_shape)
    from keras_unet_collection import models as kuc
    common=dict(input_size=input_shape,n_labels=1,stack_num_down=2,stack_num_up=2,
                activation='ReLU',output_activation=None,batch_norm=True,pool=True,unpool=True,
                deep_supervision=False,backbone=None)
    if architecture=='unetpp':model=kuc.unet_plus_2d(filter_num=FILTERS,name='UNetPP',**common)
    elif architecture=='unet3plus':model=kuc.unet_3plus_2d(filter_num_down=FILTERS,filter_num_skip=[22]*4,filter_num_aggregate=110,name='UNet3Plus',**common)
    else:raise ValueError(f'Unknown architecture: {architecture}')
    output=layers.Activation('sigmoid',dtype='float32',name=f'{model.outputs[0]._keras_history.operation.name}_activation')(model.outputs[0])
    return Model(model.inputs[0],output,name=model.name)
