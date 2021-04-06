import os
import sys
from util.opt import Options
import util.utilities as utl
import torch
import torch.nn as nn
from model import models3 as m3
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import *
import scipy.io
from keras_contrib.layers import InstanceNormalization

opt = Options(sys.argv[0])

def KerasModelSimple(inp_shape=[100,100,3]):
    x = kl.Input(inp_shape)
    net = kl.Conv2D(3, 3, strides=(1,1), padding='same', use_bias=False, name='conv1')(x)
    # net = kl.Conv2D(3, 3, strides=(1,1), padding='same', name='conv2')(net)

    model = Model(inputs=x, outputs=net)
    return model

kmodel = KerasModelSimple()
for lyr in kmodel.layers:
    a = lyr.get_weights()
input('....')

# model_path = os.path.join(opt.model_path, 'model_simple.h5')
# tflite_path = os.path.join(opt.model_path, 'model_simple.tflite')
# model = KerasModelSimple()
# tf.keras.models.save_model(model, model_path)
# converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
# tflite_model = converter.convert()
# open(tflite_path, 'wb').write(tflite_model)
# input('....')

# model_path = os.path.join(opt.model_path, 'test_model.h5')
# tflite_model_path = os.path.join(opt.model_path, 'test_model.tflite')
# converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
# tflite_model = converter.convert()
# open(tflite_model_path, 'wb').write(tflite_model)
# input('Finished Saving TFlite Model ... press to continue')

class PyModel(nn.Module):
    def __init__(self):

        super(PyModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(3, 64, (4,4), 2, 1),
            nn.InstanceNorm2d(64, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inp):
        out = self.conv1(inp)
        return out

def instance_norm(x):
    import tensorflow as tf
    layer = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-5, trainable=False)
    return layer

def conv_block(inp, outch, ksz, st, padding, bias=True, activ=None, norm=None, name=None):
    net = kl.ZeroPadding2D(padding=padding)(inp)
    net = kl.Conv2D(outch, ksz, strides=st, padding='valid', use_bias=bias, name=name)(net)

    if norm == 'instnorm':
        net = kl.Lambda(instance_norm)(net)

    if activ == 'leaky':
        net = kl.LeakyReLU(0.2)(net)
    elif activ == 'relu':
        net = kl.ReLU()(net)
    elif activ == 'tanh':
        net = kl.Activation('tanh')(net)
    return net

def convT_block(inp, outch, ksz, st, padding='same', bias=True, activ=None, norm=None, name=None):
    if padding != 'same' or padding != 'valid':
        net = kl.Conv2DTranspose(outch, ksz, strides=st, padding=padding, use_bias=bias, name=name)(inp)
    else:
        inp = kl.ZeroPadding2D(padding=(1,1))(inp)
        net = kl.Conv2DTranspose(outch, ksz, strides=st, padding='valid', use_bias=bias, name=name)(inp)

    if norm == 'instnorm':
        net = kl.Lambda(instance_norm)(net)
    if activ == 'leaky':
        net = kl.LeakyReLU(0.2)(net)
    elif activ == 'relu':
        net = kl.ReLU()(net)
    elif activ == 'tanh':
        net = kl.Activation('tanh')(net)
    return net

def res_block(inp, outch, depth=2, activ=None, norm=None, name=None):
    net = inp
    for i in range(depth):
        if (i == depth - 1):
            # net = kl.Add()([net, inp])
            net = conv_block(net, outch, (3,3), (1,1), (1,1), norm=norm, name=name + '_' + str(i))
            net = kl.Add()([net, inp])
            if activ =='leaky':
                net = kl.LeakyReLU(0.2)(net)
        else:
            net = conv_block(net, outch, (3,3), (1,1), (1,1), activ, norm, name=name + '_' + str(i))
    return net

def res_block_decomp(inp, ori,tgt, depth=2, activ=None, norm=None, name=None):
    net = inp
    j = 0
    for i in range(depth):
        if (i == depth - 1):
            # net = kl.Add()([net, inp])
            net = conv_block(net, tgt, (1,1), (1,1), (0,0), bias=False, name=name + '_' + str(j+0))
            net = conv_block(net, tgt, (3,3), (1,1), (1,1), bias=False, name=name + '_' + str(j+1))
            net = conv_block(net, ori, (1,1), (1,1), (0,0), norm=norm, name=name + '_' + str(j+2))
            net = kl.Add()([net, inp])
            if activ =='leaky':
                net = kl.LeakyReLU(0.2)(net)
        else:
            net = conv_block(net, tgt, (1,1), (1,1), (0,0), bias=False, name=name + '_' + str(j+0))
            net = conv_block(net, tgt, (3,3), (1,1), (1,1), bias=False, name=name + '_' + str(j+1))
            net = conv_block(net, ori, (1,1), (1,1), (0,0), activ=activ, norm=norm, name=name + '_' + str(j+2))
            # net = conv_block(net, outch, (3,3), (1,1), (1,1), activ, norm, name=name + '_' + str(i))
        j += 3
    return net

def KModel(inp_shape, activ='leaky', norm='instnorm'):
    x = kl.Input(inp_shape)
    xs = kl.AveragePooling2D((2,2))(x)
    med = conv_block(x, 64, (3,3), (1,1), (1,1), activ=activ, norm=norm, name='in_med0')
    med = conv_block(med, 128, (3,3), (1,1), (1,1), activ, norm, name='in_med1')
    med = conv_block(med, 128, (4,4), (2,2), (1,1), name='in_med2')

    small = conv_block(xs, 64, (3,3), (1,1), (1,1), activ, norm, name='in_small0')
    small = conv_block(small, 128, (3,3), (2,2), (1,1), activ, norm, name='in_small1')
    small_add = kl.Add(name='small_add')([small, med])
    small = conv_block(small, 128, (3,3), (1,1), (1,1), activ, norm, name='in_small2')

    enc = conv_block(small, 256, (3,3), (2,2), (1,1), activ, norm, name='conv3')
    enc = conv_block(enc, 512, (3,3), (2,2), (1,1), activ, norm, name='conv4')
    enc = conv_block(enc, 1024, (3,3), (2,2), (1,1), activ, norm, name='conv5')

    encr = res_block(enc, 1024, 2, activ, norm, name='resblock0')
    encr = res_block(encr, 1024, 2, activ, norm, name='resblock1')
    encr = res_block(encr, 1024, 2, activ, norm,name='resblock2')
    encr = res_block(encr, 1024, 2, activ, norm, name='resblock3')
    encr = res_block(encr, 1024, 2, activ, norm, name='resblock4')
    encr = res_block(encr, 1024, 2, activ, norm, name='resblock5')

    dec = convT_block(encr, 512, (4,4), (2,2), 'same', activ, norm, name='dconv1')
    dec = convT_block(dec, 256, (4,4), (2,2), 'same', activ, norm, name='dconv2')
    dec = convT_block(dec, 128, (4,4), (2,2), 'same', activ, norm, name='dconv3')
    dec_cat1 = kl.Concatenate(axis=-1)([dec,small])
    dec = convT_block(dec_cat1, 64, (4,4), (2,2), 'same', activ, norm, name='dconvs')
    dec_cat2 = kl.Concatenate(axis=-1)([dec, med])
    dec = convT_block(dec_cat2, 64, (4,4), (2,2), 'same', activ, norm, name='dconvm')
    out = convT_block(dec, 3, (3,3), (1,1), 'same', activ='tanh', name='outm')
    # out = med
    model = Model(inputs=x, outputs=out)
    return model

def transfer_weights(pymodel, kmodel):
    # Store Pytorch model to dictionary
    weights = {}
    for name, module in pymodel._modules.items():
        if name != 'resblock':
            w = module.block[0].weight.data.numpy()
            b = module.block[0].bias.data.numpy()
            weights[name] = [w,b]
        if name == 'resblock':
            for i, bseq in enumerate(module):
                j = 0
                for bl in bseq.block:
                    if isinstance(bl, nn.Conv2d):
                        w = bl.weight.data.numpy()
                        b = bl.bias.data.numpy()
                        nm = name + str(i) + '_' + str(j)
                        weights[nm] = [w,b]
                        j += 1

    for lyr in kmodel.layers:
        if len(lyr.get_weights()) > 0:
            if lyr.name in weights.keys():
                [w, b] = weights[lyr.name]

                lyr.set_weights([w.transpose(2, 3, 1, 0), b])
                # lyr.set_weights([w, b])
            else:
                print("Missing key", lyr.name, "in dictionary")

def KModel_compressed(inp_shape, activ='leaky', norm='instnorm'):
    x = kl.Input(inp_shape)
    xs = kl.AveragePooling2D((2,2))(x)
    med = conv_block(x, 64, (3,3), (1,1), (1,1), activ=activ, name='in_med0')
    med = conv_block(med, 128, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='in_med1')
    med = conv_block(med, 128, (3,3), (2,2), (1,1), norm=norm, name='in_med2')

    small = conv_block(xs, 64, (3,3), (1,1), (1,1), activ=activ, name='in_small0')
    small = conv_block(small, 128, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='in_small1')
    small = conv_block(small, 128, (3,3), (1,1), (1,1), norm=norm, name='in_small2')
    small = kl.Add(name='small_add')([small, med])
    small = kl.LeakyReLU(0.2, name='leaky')(small)

    enc = conv_block(small, 256, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='conv3')
    enc = conv_block(enc, 512, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='conv4')
    enc = conv_block(enc, 1024, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='conv5')

    # encr = res_block(enc, 1024, 2, activ, norm, name='resblock')
    encr = res_block_decomp(enc, 1024, 256, 2, activ, norm, name='resblock')

    dec = convT_block(encr, 512, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconv1')
    dec = convT_block(dec, 256, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconv2')
    dec = convT_block(dec, 128, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconv3')
    dec_cat1 = kl.Concatenate(axis=-1)([dec,small])
    dec = convT_block(dec_cat1, 64, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconvs')
    # dec_cat2 = kl.Concatenate(axis=-1)([dec, med])
    dec = kl.UpSampling2D(size=(2,2), interpolation='bilinear')(dec)
    dec = convT_block(dec, 64, (3,3), (1,1), 'same', activ=activ, norm=norm, name='dconvm')
    out = convT_block(dec, 3, (3,3), (1,1), 'same', activ='tanh', name='outm')
    # out = med
    model = Model(inputs=x, outputs=out)
    return model

def KModel_compressed2(inp_shape, activ='leaky', norm='instnorm'):
    x = kl.Input(inp_shape)
    xs = kl.AveragePooling2D((2,2))(x)
    med = conv_block(x, 64, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='in_med0')
    med = conv_block(med, 128, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='in_med1')
    med = conv_block(med, 128, (3,3), (2,2), (1,1), name='in_med2')

    small = conv_block(xs, 64, (3,3), (1,1), (1,1), activ=activ, norm=norm, name='in_small0')
    small = conv_block(small, 128, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='in_small1')
    small_add = kl.Add(name='small_add')([small, med])
    small = conv_block(small_add, 128, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='in_small2')

    enc = conv_block(small, 256, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='conv3')
    enc = conv_block(enc, 512, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='conv4')
    enc = conv_block(enc, 1024, (3,3), (2,2), (1,1), activ=activ, norm=norm, name='conv5')

    # encr = res_block(enc, 1024, 2, activ, norm, name='resblock')
    encr = res_block_decomp(enc, 1024, 256, 2, activ, norm, name='resblock')

    dec = convT_block(encr, 512, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconv1')
    dec = convT_block(dec, 256, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconv2')
    dec = convT_block(dec, 128, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconv3')
    dec_cat1 = kl.Concatenate(axis=-1)([dec,small])
    dec = convT_block(dec_cat1, 64, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconvs')
    dec_cat2 = kl.Concatenate(axis=-1)([dec, med])
    dec = convT_block(dec_cat2, 64, (4,4), (2,2), 'same', activ=activ, norm=norm, name='dconvm')
    out = convT_block(dec, 3, (3,3), (1,1), 'same', activ='tanh', name='outm')
    # out = med
    model = Model(inputs=x, outputs=out)
    return model

def transfer_weights_compressed(pymodel, kmodel, path=opt.model_path):
    # Store Pytorch model to dictionary
    weights = {}
    for name, module in pymodel._modules.items():
        if name != 'resblock' and name !='activ':
            print('NAME:',name)
            w = module.block[0].weight.data.numpy()
            b = module.block[0].bias.data.numpy()
            weights[name] = [w,b]
        if name == 'resblock':
            print('NAME:',name)
            j = 0
            for c, bl in enumerate(module.block):
                if isinstance(bl, nn.Conv2d):
                    w = bl.weight.data.numpy()
                    if hasattr(bl.bias,'data'):
                        b = bl.bias.data.numpy()
                        nm = name + '_' + str(j)
                        weights[nm] = [w,b]
                    else:
                        nm = name + '_' + str(j)
                        weights[nm] = [w]
                    j += 1

    for lyr in kmodel.layers:
        if len(lyr.get_weights()) > 0:
            if lyr.name in weights.keys():
                # [w, b] = weights[lyr.name]
                if len(weights[lyr.name]) > 1:
                    [w, b] = weights[lyr.name]
                    lyr.set_weights([w.transpose(2, 3, 1, 0), b])
                else:
                    print(lyr.name)
                    [w] = weights[lyr.name]
                    lyr.set_weights([w.transpose(2, 3, 1, 0)])
                    # Correct conv operation with transposed weight in pytorch
                    # lyr.set_weights([w.transpose(2, 3, 0, 1)])
            else:
                print("Missing key", lyr.name, "in dictionary")

# Initliaze input image
image_path = '../img/img_out.jpg'
# inp = torch.rand(1, 3, 256, 512)
inp = utl.read_image_to_tensor(image_path, scale=(0.5,0.5), normalize=True)
print(inp.size())
inp_np = inp.numpy().transpose(0,2,3,1)
inp_shape = inp_np.shape

# Define and load Pytorch Model
# py_model_path = os.path.join(opt.model_path, 'model_190712/model_n_medium_30000.pt')
# py_model_path = os.path.join(opt.model_path, 'compressed/ts_decompose_256_17000.pt')
py_model_path = os.path.join(opt.model_path, 'ts_decompose_arch256_7000.pt')
py_model = torch.load(py_model_path)
pmodel = m3.GTestMobileStudent()
pmodel.decompose_layer() # Decompose the tensor
# pmodel.load_state_dict(py_model['Generator'], strict=False)
pmodel.load_state_dict(py_model, strict=True)
input('Finished loading Pytorch model ... press to continue')

# Define Keras Model
model_path = os.path.join(opt.model_path, 'model_comp_decomp.h5')
tflite_model_path = os.path.join(opt.model_path, 'model_comp_decomp.tflite')
# kmodel = KModel(inp_shape[1:])
kmodel = KModel_compressed(inp_shape[1:])
kmodel.summary()
kmodel.compile(tf.keras.optimizers.Adam(0.0002), loss=tf.keras.losses.MSE)
kmodel.save(model_path)
transfer_weights_compressed(pmodel, kmodel)
input('Finished Transfering Pytorch model to Keras ... press to continue')
tf.keras.models.save_model(kmodel, model_path)
input('Finished Saving Keras Model ... press to continue')

# Test save and load model correctly
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
# converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open(tflite_model_path, 'wb').write(tflite_model)
input('Finished Saving TFlite Model ... press to continue')
kmodel = tf.keras.models.load_model(model_path, compile=True)

# Compare result
_, pout = pmodel(inp)
pout = pout.detach().numpy().transpose(0,2,3,1)
kout = kmodel.predict(inp_np, batch_size=1)
#
# # Test TFLite inference
tflint = tf.lite.Interpreter(tflite_model_path)
tflint.allocate_tensors()
input_details = tflint.get_input_details()
output_details = tflint.get_output_details()
print(input_details)
input_shape = input_details[0]['shape']
tflint.set_tensor(input_details[0]['index'],inp_np)
tflint.invoke()

tfout = tflint.get_tensor(output_details[0]['index'])
#
print(pout.shape, tfout.shape)
print(np.mean(np.abs(pout - tfout)))

out_comp = np.hstack((utl.postproc_img(pout), utl.postproc_img(tfout)))
plt.imshow(out_comp)
plt.show()




