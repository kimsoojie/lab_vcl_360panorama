import os
import sys
from util.opt import Options
import util.utilities as utl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import model.models as m
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import *

opt  = Options(sys.argv[0])



def instance_norm(x):
    import tensorflow as tf
    layer = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-5, trainable=False)
    return layer

def strided_slice0(x):
    import tensorflow as tf
    out = tf.strided_slice(x, [0,0,0,0],[1,256,128,3],[1,1,1,1])
    return out

def strided_slice1(x):
    import tensorflow as tf
    out = tf.strided_slice(x, [0,0,128,0],[1,256,256,3],[1,1,1,1])
    return out

def strided_slice2(x):
    import tensorflow as tf
    out = tf.strided_slice(x, [0,0,256,0],[1,256,384,3],[1,1,1,1])
    return out

def strided_slice3(x):
    import tensorflow as tf
    out = tf.strided_slice(x, [0,0,384,0],[1,256,512,3],[1,1,1,1])
    return out


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

def kFOVmodel(inp=[128,512,3]):
    x = kl.Input(inp)
    activ = 'relu'
    norm = 'instnorm'
    # x0 = x[:,:,0:128,:]
    x0 = kl.Lambda(strided_slice0)(x)
    print(x0.shape)
    # x1 = x[:,:,128:256,:]
    x1 = kl.Lambda(strided_slice1)(x)
    print(x1.shape)
    # x2 = x[:,:,256:384,:]
    x2 = kl.Lambda(strided_slice2)(x)
    # x3 = x[:,:,384:512,:]
    x3 = kl.Lambda(strided_slice3)(x)

    conv00 = conv_block(x0, 64, (3,3), (1,1), (1,1), activ=activ, name='conv0_0_0')
    conv00 = conv_block(conv00, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_0_1')
    conv00 = conv_block(conv00, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_0_2')

    conv01 = conv_block(x1, 64, (3,3), (1,1), (1,1), activ=activ, name='conv0_1_0')
    conv01 = conv_block(conv01, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_1_1')
    conv01 = conv_block(conv01, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_1_2')

    conv02 = conv_block(x2, 64, (3,3), (1,1), (1,1), activ=activ, name='conv0_2_0')
    conv02 = conv_block(conv02, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_2_1')
    conv02 = conv_block(conv02, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_2_2')

    conv03 = conv_block(x3, 64, (3,3), (1,1), (1,1), activ=activ, name='conv0_3_0')
    conv03 = conv_block(conv03, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_3_1')
    conv03 = conv_block(conv03, 64, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv0_3_2')


    conv1 = kl.Concatenate(axis=2)([conv00,conv01,conv02,conv03])
    conv1 = conv_block(conv1, 128, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv1_0')
    conv1 = conv_block(conv1, 256, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv1_1')
    conv1 = conv_block(conv1, 256, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv1_2')
    conv1 = conv_block(conv1, 256, (4,4), (2,2), (1,1), activ=activ, norm=norm, name='conv1_3')
    # print(conv1.shape)
    #
    lin = kl.Flatten()(conv1)
    lin = kl.Dense(2048, name='lin0_0')(lin)
    # lin = kl.Dropout(0.0)(lin)
    # lin = kl.ReLU()(lin)
    out = kl.Dense(128, name='lin0_1')(lin)
    out = kl.Activation('sigmoid')(out)
    print(out.shape)

    model = Model(inputs=x,outputs=out)
    return model

def transfer_weights(pymodel, kmodel):
    weights = {}
    for name, module in pymodel._modules.items():
        if name.startswith('conv'):
            for idx, bl in enumerate(module):
                w = bl[0].weight.data.numpy()
                b = bl[0].bias.data.numpy()
                wname = name + '_' + str(idx)
                print(wname)
                weights[wname] = [w,b]
        if name == 'lin0':
            idx = 0
            for bl in module:
                if isinstance(bl,nn.Linear):
                    w = bl.weight.data.numpy()
                    b = bl.bias.data.numpy()
                    wname = name + '_' + str(idx)
                    print(wname, w.shape, b.shape)
                    weights[wname] = [w,b]
                    idx += 1
        if name == 'lin_test':
            w = module.weight.data.numpy()
            b = module.bias.data.numpy()
            weights[name] = [w,b]

    for lyr in kmodel.layers:
        if len(lyr.get_weights()) > 0:
            if lyr.name in weights.keys():
                if lyr.name.startswith('conv'):
                    [w,b] = weights[lyr.name]
                    lyr.set_weights([w.transpose(2,3,1,0),b])
                elif lyr.name.startswith('lin'):
                    [w,b] = weights[lyr.name]
                    lyr.set_weights([np.transpose(w),b])
                else:
                    print('Missing key', lyr.name, 'in dictionary')


class testLinear(nn.Module):
    def __init__(self):
        super(testLinear, self).__init__()
        self.lin_test = nn.Linear(10,10)

    def forward(self, x):
        out = self.lin_test(x)
        return out;

def testKerasLinear(input_shape=[10]):
    x = kl.Input(input_shape)
    out = kl.Dense(10, name='lin_test')(x)
    model = Model(inputs=x, outputs=out)
    return model

pmodel = testLinear()
kmodel = testKerasLinear()
transfer_weights(pmodel, kmodel)
inp_np = np.random.rand(1,10)
inp_np = inp_np.astype(np.float32)
outpytorch = pmodel(torch.from_numpy(inp_np))
outpytorch = outpytorch.detach().numpy()
outkeras = kmodel.predict(inp_np, batch_size=1)
print(np.argmax(outpytorch))
print(np.argmax(outkeras))


input('...')


model_path = os.path.join(opt.model_path, 'model_fov.h5')
tflite_model_path = os.path.join(opt.model_path, 'model_fov.tflite')

image_path = '../img/img_out.jpg'
inp = cv2.imread(image_path)
inp = cv2.resize(inp, (512,128))
print(inp.shape)
inp = np.expand_dims(inp, 0)
inp = inp.astype(np.float32)

weights_path = os.path.join(opt.model_path, 'fov/acc/model_fov2_fov_2000.pt')
pmodel = m.FOVnetwork()
pmodel.load_state_dict(torch.load(weights_path),strict=False)
kmodel = kFOVmodel()

transfer_weights(pmodel, kmodel)
input('....FINISHED TRANSFERRING WEIGHTS....')

# kmodel.compile(tf.keras.optimizers.Adam(0.002), loss=tf.keras.losses.CategoricalCrossentropy)
tf.keras.models.save_model(kmodel, model_path)
out = kmodel.predict(inp, batch_size=1)
inp_torch = torch.from_numpy(inp.transpose(0,3,1,2))
print(inp_torch.shape)
_, outp = pmodel(inp_torch)
outp = outp.detach().numpy()
print(np.argmax(outp))
print(np.argmax(out))
print(np.mean(np.abs(out - outp)))

converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open(tflite_model_path, 'wb').write(tflite_model)


# plt.imshow(out[0,:,:,:])
# plt.show()
