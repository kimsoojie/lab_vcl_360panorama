import os
import sys
import math
import time
import random

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import segmentation as ski

import torch
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tensorboardX import SummaryWriter

def read_img_to_tensor(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def write_output_single(folder_in, im_name, folder_out, model_path, net_type):
    in_img = read_img_to_tensor(folder_in + '/' + im_name)
    in_img = torch.unsqueeze(in_img, 0)