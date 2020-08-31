from __future__ import print_function
import numpy as np
import pickle
from sklearn import datasets, linear_model
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pickle
import numpy as np
from pprint import pprint
import struct

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

# import pandas as pd

import io, gzip, requests
import scipy.io
import scipy.io as sio

def get_dataset_params(model, dataset):
    
    if model == 'MLP':
        flatten = True
    else:
        flatten = False
    
    if (dataset == 'svhn' or dataset == 'cifar10' or dataset == 'cifar100'):
        image_aug = True
    else:
        image_aug = False
        
    epochs = 100
    if model == 'MLP':
        epochs = 100
    elif model == 'ResNet18' or model == 'VGG18' or model == 'ResNet101':
        if dataset == 'mnist' or dataset == 'fashion' or dataset == 'svhn':
            epochs = 40
    if dataset == 'iris':
        epochs = 20000
        
    if dataset == 'random':
        epochs = 40
        if model != 'MLP':
            epochs = 40
    
    if dataset == 'imagenet32':
        epochs = 50
        if model == "MLP":
            epochs = 3000
        
    num_classes = 10
    if dataset == 'mnist' or dataset == 'fashion':
        img_dim = 28
        input_shape = 28*28
        channels = 1
        input_padding = 2
    elif dataset == 'iris':
        img_dim = 4
        input_shape = 4
        channels = 1
        input_padding = 2
        num_classes = 3
    else:
        img_dim = 32
        input_shape = 32*32*3
        channels = 3
        input_padding = 0
    
    if dataset == 'cifar100':
        num_classes = 100
    
    if dataset == 'imagenet32':
        num_classes = 1000
    
    if dataset == "iris":
        flatten = False

    return flatten, image_aug, input_shape, channels, input_padding, num_classes, epochs, img_dim


import math
def get_entropy(x):
    hist = np.histogram(x, bins=1000, range = (0,1),density=True)
    ent=0
    for i in hist[0]:
        if i!=0:
            ent -= i * np.log2(abs(i))
    return ent
