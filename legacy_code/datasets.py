from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd

import torchvision
from torchvision import datasets, transforms

import struct
from copy import deepcopy
from time import time, sleep
import gc

import io, gzip#, requests
import scipy.io
import scipy.io as sio

import matplotlib.pyplot as plt
import os

import pickle

#from sklearn.preprocessing import normalize
gpu_boole = torch.cuda.is_available()

def get_loaders(dataset, BS=128, N2 = 0, N_sub = 0, BS2 = 20, image_aug = False, ablate = False, entropy_bool = False, shuffle = True, shuffle_perc = 0, entropy_boole = True):
    
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    
    if image_aug:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Lambda(lambda x: add_noise(x))
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Lambda(lambda x: add_noise(x))
            ])

    
    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        train_set_bap = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset_bap = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

        if shuffle_perc > 0:
            N = train_set.train_labels.size()[0]
            K = int(round(N*shuffle_perc))
            rand_inds = torch.randperm(K)
            train_set.train_labels = torch.cat([train_set.train_labels[rand_inds],train_set.train_labels[K:]])



    elif dataset == 'fashion':
        train_set = fashion(root='./data_fashion', 
                            train=True, 
                            transform=transform_train,
                            download=True
                           )

        test_dataset = fashion(root='./data_fashion', 
                                    train=False, 
                                    transform=transform_test,
                                   )
        train_set_bap = fashion(root='./data_fashion', train=True, download=True, transform=transform_train)
        test_dataset_bap = fashion(root='./data_fashion', train=False, download=True, transform=transform_test)

        if shuffle_perc > 0:
            N = train_set.train_labels.size()[0]
            K = int(round(N*shuffle_perc))
            rand_inds = torch.randperm(K)
            train_set.train_labels = torch.cat([train_set.train_labels[rand_inds],train_set.train_labels[K:]])

    
    elif dataset == 'svhn':
        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        train_set_bap = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_dataset_bap = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

        if shuffle_perc > 0:
            N = train_set.labels.size
            K = int(round(N*shuffle_perc))
            rand_inds = torch.randperm(K)
            train_set.labels = torch.cat([torch.Tensor(train_set.labels[rand_inds]),torch.Tensor(train_set.labels[K:])]).cpu().data.numpy()

        
    elif dataset == 'iris':
        
        # load IRIS dataset
        temp = pd.read_csv('./data/iris.csv')
        
        # transform species to numerics
        temp.loc[temp.species=='Iris-setosa', 'species'] = 0
        temp.loc[temp.species=='Iris-versicolor', 'species'] = 1
        temp.loc[temp.species=='Iris-virginica', 'species'] = 2
        
        
        train_X, test_X, train_y, test_y = train_test_split(temp[temp.columns[0:4]].values,
                                                            temp.species.values, test_size=0.2)        
        train_X, test_X = train_X / train_X.max(), test_X / test_X.max()
        
        # wrap up with Variable in pytorch
        xtrain = Variable(torch.Tensor(train_X).float())
        xtest = Variable(torch.Tensor(test_X).float())
        ytrain = Variable(torch.Tensor(train_y).long())
        ytest = Variable(torch.Tensor(test_y).long())        

        train = torch.utils.data.TensorDataset(xtrain, ytrain)
        test = torch.utils.data.TensorDataset(xtest, ytest)

        train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)
        
        if N2 > 0:
            train_bap = torch.utils.data.TensorDataset(xtrain[:N2], ytrain[:N2])
            test_bap = torch.utils.data.TensorDataset(xtest[:N2], ytest[:N2])
        else:
            train_bap = torch.utils.data.TensorDataset(xtrain, ytrain)
            test_bap = torch.utils.data.TensorDataset(xtest, ytest)            
        
        train_loader_bap = torch.utils.data.DataLoader(train_bap, batch_size=BS2, shuffle=False)
        test_loader_bap = torch.utils.data.DataLoader(test_bap, batch_size=BS2, shuffle=False)
        
        if N2 > 0:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')[:N2]
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')[:N2]
        else:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')
     

        return train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test
            
    elif dataset == 'random':
        
        xtrain = Variable(torch.Tensor(np.load('./data/xtrain_random.npy')).float())
        xtest = Variable(torch.Tensor(np.load('./data/xtest_random.npy')).float())
        ytrain = Variable(torch.Tensor(np.load('./data/ytrain_random.npy')).long())
        ytest = Variable(torch.Tensor(np.load('./data/ytest_random.npy')).long())        

        train = torch.utils.data.TensorDataset(xtrain, ytrain)
        test = torch.utils.data.TensorDataset(xtest, ytest)

        train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)
        
        if N2 > 0:
            train_bap = torch.utils.data.TensorDataset(xtrain[:N2], ytrain[:N2])
            test_bap = torch.utils.data.TensorDataset(xtest[:N2], ytest[:N2])
        else:
            train_bap = torch.utils.data.TensorDataset(xtrain, ytrain)
            test_bap = torch.utils.data.TensorDataset(xtest, ytest)            
        
        train_loader_bap = torch.utils.data.DataLoader(train_bap, batch_size=BS2, shuffle=False)
        test_loader_bap = torch.utils.data.DataLoader(test_bap, batch_size=BS2, shuffle=False)
        
        if N2 > 0:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')[:N2]
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')[:N2]
        else:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')
        
        
        return train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test
        
    elif dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_set_bap = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset_bap = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        if shuffle_perc > 0:            
            N = np.array(train_set.train_labels).size
            K = int(round(N*shuffle_perc))
            rand_inds = torch.randperm(K)
            train_set.train_labels = list(torch.cat([torch.Tensor(np.array(train_set.train_labels)[rand_inds]),torch.Tensor(np.array(train_set.train_labels[K:]))]).cpu().data.numpy().astype(np.int64))


    elif dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        train_set_bap = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset_bap = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

        if shuffle_perc > 0:            
            N = np.array(train_set.train_labels).size
            K = int(round(N*shuffle_perc))
            rand_inds = torch.randperm(K)
            train_set.train_labels = list(torch.cat([torch.Tensor(np.array(train_set.train_labels)[rand_inds]),torch.Tensor(np.array(train_set.train_labels[K:]))]).cpu().data.numpy().astype(np.int64))
    
    
    elif dataset == 'imagenet32':
        xtrain, ytrain = imagenet_train()
        xtest, ytest = imagenet_test()

        xtrain = torch.Tensor(xtrain).float()
        xtest = torch.Tensor(xtest).float()
        ytrain = torch.Tensor(ytrain).long()
        ytest = torch.Tensor(ytest).long()

        train = torch.utils.data.TensorDataset(xtrain, ytrain)
        test = torch.utils.data.TensorDataset(xtest, ytest)

        train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)
        
        if N2 > 0:
            train_bap = torch.utils.data.TensorDataset(xtrain[:N2], ytrain[:N2])
            test_bap = torch.utils.data.TensorDataset(xtest[:N2], ytest[:N2])
        else:
            train_bap = torch.utils.data.TensorDataset(xtrain, ytrain)
            test_bap = torch.utils.data.TensorDataset(xtest, ytest)            
        
        train_loader_bap = torch.utils.data.DataLoader(train_bap, batch_size=BS2, shuffle=False)
        test_loader_bap = torch.utils.data.DataLoader(test_bap, batch_size=BS2, shuffle=False)
        
        if N2 > 0:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')[:N2]
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')[:N2]
        else:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')
     
        # entropy = 0
        # entropy_test = 0

        return train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test
        # return train_loader, test_loader, train_loader_bap, test_loader_bap
            
    
    
    if N_sub>0:
        train_set.train_data = train_set.train_data[:N_sub]    
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BS, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BS, shuffle=False)
    
    if N2 > 0:
        train_set_bap.train_data = train_set_bap.train_data[:N2]
        test_dataset_bap.test_data = test_dataset_bap.test_data[:N2]
    
    train_loader_bap = torch.utils.data.DataLoader(train_set_bap, batch_size = BS2, shuffle=False)
    test_loader_bap = torch.utils.data.DataLoader(test_dataset_bap, batch_size = BS2, shuffle=False)
    
    if ablate:
        train_loader_bap = torch.utils.data.DataLoader(train_set_bap, batch_size = 10000, shuffle=False)
        test_loader_bap = torch.utils.data.DataLoader(test_dataset_bap, batch_size = 10000, shuffle=False)
    
    ##entropy:
    if N2 > 0:
        entropy = np.load('./data/entropy_'+dataset+'_train.npy')[:N2]
        entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')[:N2]
    else:
        entropy = np.load('./data/entropy_'+dataset+'_train.npy')
        entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')
        
    if entropy_bool:
        if N_sub > 0:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')[:N_sub]
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')[:N_sub]
        else:
            entropy = np.load('./data/entropy_'+dataset+'_train.npy')
            entropy_test = np.load('./data/entropy_'+dataset+'_test.npy')
        etrain = torch.from_numpy(entropy.flatten()).float()
        etest = torch.from_numpy(entropy_test.flatten()).float()
        train_loader.dataset.train_labels = entropy
        test_loader.dataset.test_labels = entropy_test
        return train_loader, test_loader
        
    
    return train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test

##Loading for fashion-mnist:
    
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs


class fashion(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

def imagenet_train(data_folder = './data/imagenet32_train', img_size=32):
    
    xbatches = []
    ybatches = []
    

    for idx in range(10):

        data_file = os.path.join(data_folder, 'train_data_batch_')
    
        d = unpickle(data_file + str(idx+1))
        x = d['data']
        y = d['labels']
        
        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]
    
        img_size2 = img_size * img_size
    
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        x = x.astype(np.float)
        x /= 255
    
        xbatches.append(x)
        ybatches.append(y)        
        
    xtrain = np.concatenate(xbatches)
    ytrain = np.concatenate(ybatches)
    
    print(xtrain.shape)

    return xtrain, ytrain

def imagenet_test(data_folder = './data/', img_size=32):
    
    data_file = os.path.join(data_folder, 'val_data')

    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    x = x.astype(np.float)
    x /= 255

    return x, y


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)


def get_random_batch(loader,N,BS):
    batch_num = np.random.randint(0,int(np.floor(N//BS)))
    for i, (x,y) in enumerate(loader):
        if i!= batch_num:
            continue
        else:
            return x, y