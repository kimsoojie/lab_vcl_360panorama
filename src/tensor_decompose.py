import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, partial_tucker
from tensorly.tenalg import multi_mode_dot
import scipy
from scipy.io import loadmat
import os, sys
from util.opt import Options
import model.models3 as m3
from util.base import *

opt = Options()


# net = m3.GTest()
# print(net)
# input('....')
# net.decompose_layer()
# print(net)
# input('....')
def make_conv_module(kernel, ranks):
    w = kernel.weight
    cr, [first, last] = partial_tucker(w, modes=[0,1], ranks=ranks)
    print('Core', cr.shape)
    print('last', last.shape)
    source = nn.Conv2d(first.shape[1], first.shape[0], 1, 1, 0, bias=False)
    core = nn.ConvTranspose2d(cr.shape[0], cr.shape[1], kernel.kernel_size,
                     kernel.stride, kernel.padding, bias=False)
    target = nn.Conv2d(last.shape[1], last.shape[0], 1, 1, 0, bias=True)
    target.bias.data = kernel.bias.data
    print(cr.size())
    print(first.size())
    print(last.size())

    srcw = torch.transpose(first,1,0).unsqueeze(-1).unsqueeze(-1)
    print('src_transpose',srcw.size())
    print('src',first.size())
    source.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    core.weight.data = cr
    target.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    modules = [source, core, target]
    return nn.Sequential(*modules)


net_meta = {'conv3':[128,128,3,3],
            'conv4':[512,128,3,3]}

dummy = torch.randn(1,1024,10,10)
conv = nn.Conv2d(1024,1024,3,2,1)
conv1 = nn.ConvTranspose2d(1024,256, 1, 1, 0, bias=False)
conv2 = nn.Conv2d(256,256, 3, 2, 1, bias=False)
conv3 = nn.Conv2d(256,1024, 1, 1, 0, bias=False)

if hasattr(conv1.bias,'data'):
    print(conv1.bias.data)
    print('Have')
else:
    print('Dont have')
# core, [tgt, src] = partial_tucker(conv.weight, modes=[0,1], ranks=[256,256])
# print('core', core.shape)
# print('tgt', tgt.shape)
# print('src', src.shape)
# convt = nn.Conv2d(src.shape[1],src.shape[0],1,1,0, bias=False)
# print('conv1_w_shape', conv1.weight.data.size())
# conv1.weight.data = torch.transpose(src,1,0).unsqueeze(-1).unsqueeze(-1)
# conv1.weight.data = src.unsqueeze(-1).unsqueeze(-1)

out1 = conv1(dummy)
print(out1.size())
# net_model = m3.GTest()
# conv3w = net_model.dconv1.block[0]
# out = conv3w(dummy)
# conv3c = make_conv_module(conv3w, [512,512])
# print(conv3c)
# outc = conv3c(dummy)
input('..')






def make_conv_module(in_kernel, tensor_red, ksz, st, pad, activ=None, norm=None):
    conv = in_kernel.weight.data.numpy()
    # bias = in_kernel.bias.data.numpy()
    core, factors = tucker(conv, rank=tensor_red)
    source = np.expand_dims(factors[1],-1)
    source = np.expand_dims(source,-1)

    target = np.expand_dims(factors[0],-1)
    target = np.expand_dims(target, -1)

    w_module = {'core':core, 'source':source, 'target':target}
    for k,v in w_module.items():
        print(k, v.shape)
    return CompBlock(w_module, ksz, st, pad, activ=activ, norm=norm)

class CompBlock(nn.Module):
    def __init__(self, w_module, ksz, st, pad, activ=None, norm=None):
        super(CompBlock, self).__init__()
        mcor = w_module['core'].shape
        msrc = w_module['source'].shape
        mtgt = w_module['target'].shape
        modules = []
        modules.append(nn.Conv2d(msrc[1], msrc[0], 1, 1, 0, bias=False)) # Source
        modules.append(nn.Conv2d(mcor[1], mcor[0], ksz, st, pad, bias=False)) # Core
        modules.append(nn.Conv2d(mtgt[1], mtgt[0], 1, 1, 0, bias=False)) # Target
        print(modules[-1])

        if norm == 'batchnorm':
            modules.append(nn.BatchNorm2d(mtgt[0], affine=True))
        elif norm == 'instnorm':
            modules.append(nn.InstanceNorm2d(mtgt[0], affine=False))

        if activ == 'relu':
            modules.append(nn.ReLU(inplace=True))
        elif activ == 'leaky':
            modules.append(nn.LeakyReLU(0.2, inplace=True))
        elif activ == 'sigmoid':
            modules.append(nn.Sigmoid())
        elif activ == 'tanh':
            modules.append(nn.Tanh())

        self.block = nn.Sequential(*modules)
        self.block[0].weight.data = torch.from_numpy(w_module['source'])
        self.block[1].weight.data = torch.from_numpy(w_module['core'])
        self.block[2].weight.data = torch.from_numpy(np.ascontiguousarray(w_module['target']))
        # self.block[2].bias.data = torch.from_numpy(w_module['bias'])

    def forward(self, x):
        out = self.block(x)
        return out

class SimpleConv(nn.Module):
    def __init__(self, inf, outf, ksz, st, pad, bias):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(inf, outf, ksz, st, pad, bias=bias)

    def forward(self, x):
        return self.conv(x)

class SimpleConv2(nn.Module):
    def __init__(self, conv_w):
        super(SimpleConv2, self).__init__()
        self.conv2 = SimpleConv(128,128,3,2,1, bias=False)
        self.conv2.conv.weight.data = torch.from_numpy(conv_w)
        core, factors = partial_tucker(self.conv2.conv.weight.data.numpy(), modes=[0,1], ranks=[256,128,3,3])
        core2 = tl.tenalg.multi_mode_dot(self.conv2.conv.weight.data.numpy(),
                                         [factors[0],factors[1]],modes=[0,1],transpose=True)

        print(core.shape)
        src = np.expand_dims(factors[1],-1)
        src = np.expand_dims(src,-1)
        tgt = np.expand_dims(factors[0],-1)
        tgt = np.expand_dims(tgt,-1)


        self.core = nn.Conv2d(128,128,3,2,1, bias=False)
        self.core.weight.data = torch.from_numpy(core)
        self.src = nn.Conv2d(128,128,1,1,0, bias=False)
        self.src.weight.data = torch.from_numpy(src)
        self.tgt = nn.Conv2d(128,128,1,1,0, bias=False)
        self.tgt.weight.data = torch.from_numpy(tgt)
        # conv_est = tl.tenalg.multi_mode_dot(core, factors)
        # self.conv2_ = nn.Conv2d(128,128,3,2,1, bias=False)
        # self.conv2_.weight.data = torch.from_numpy(conv_est)

    def forward(self, x):
        out = self.src(x)
        out = self.core(out)
        out = self.tgt(out)
        return out

conv_simple = SimpleConv(128,128,3,2,1, bias=False)
print(conv_simple)

K = torch.randn(1,128,100,100)
CompBlock3 = SimpleConv2(conv_simple.conv.weight.data.numpy())
# CompBlock3 = make_conv_module(conv_simple.conv, [256,128,3,3], 3, 2, 1)
Block3 = conv_simple

print(Block3)
print(CompBlock3)

out = CompBlock3(K)
out_ = Block3(K)

print(torch.mean(torch.abs(out - out_)))


