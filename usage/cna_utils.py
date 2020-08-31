## Some general imports needed:
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

## Area and distance-related imports for CNA-Margin:
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff, pdist, cdist

gpu_boole = torch.cuda.is_available()

def get_entropy(x):
  hist = np.histogram(x, bins=1000, range = (0,1),density=True)
  data = hist[0]
  ent=0
  for i in hist[0]:
      if i!=0:
          ent -= i * np.log2(abs(i))
  return ent

def get_entropy_batch_np(x):
  entropy = []
  for i in range(x.shape[0]):
    entropy.append(get_entropy(x[i]))
  entropy = np.array(entropy)
  return entropy

def get_slopes(x, net):

  activations = []
  def get_activation_sums():
      def hook(model, input, output):
          activations.append(output.detach().view(output.shape[0],-1).sum(dim=1))
      return hook 

  hooks = []
  def register_hooks(module):    
    if any([isinstance(module, nn.Linear), isinstance(module, nn.Conv2d)]):
      handle = module.register_forward_hook(get_activation_sums())
      hooks.append(handle)
    if hasattr(module, 'children'):
      children = list(module.children())
      for i in range(len(children)):
        register_hooks(children[i])  

  register_hooks(net)
  if len(hooks) > 1:
    hooks[-1].remove()

  output = net(x)

  acts = np.array([a.cpu().data.numpy() for a in activations])

  #De-registering hooks:
  for hook in hooks:
    hook.remove()

  def threshold(M):
      Mabs = np.abs(M)
      M[Mabs<0.0000001] = 0
      return M
  
  C = np.array([np.ones(len(acts)),np.arange(1,len(acts)+1)]).transpose()
  Cf = np.linalg.inv((C.T).dot(C)).dot(C.T)
  Cf = threshold(Cf)
  Cf = Cf[1,:]

  S = 0
  for j in range(len(Cf)):
      S += acts[j]*Cf[j]
  
  return S

def get_CNA(x, net):
  entropy = get_entropy_batch_np(x.cpu().data.numpy())
  slopes = get_slopes(x, net)
  return np.corrcoef(entropy,slopes)[0,1]


def get_slopes_all(data_loader, net, flatten=False):    
  slopes = []
  for x, y in data_loader:
      if flatten:
          x = x.view(x.shape[0],-1)
      if gpu_boole:
          x = x.cuda()
      slopes = slopes + list(get_slopes(x, net))            
  slopes = np.array(slopes)
  return slopes

def get_entropy_all(data_loader):
  entropy = []
  for x, y in data_loader:
      entropy = entropy + list(get_entropy_batch_np(x.cpu().data.numpy()))
  entropy = np.array(entropy)
  return entropy

def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_normalized_slope_entropy_curves(net, train_loader, test_loader, flatten=True):
  #Getting slopes and entropy for train and test sets:
  slopes_train = get_slopes_all(train_loader, net, flatten = flatten)
  slopes_test = get_slopes_all(test_loader, net, flatten = flatten)
  entropy_train = get_entropy_all(train_loader)
  entropy_test = get_entropy_all(test_loader)

  #Indices of sorted entropy:
  ent_ind_train = np.argsort(entropy_train)
  ent_ind_test = np.argsort(entropy_test)

  #Smoothing slope and entropy curves:
  K = 25
  slopes_train_smooth, entropy_train_smooth = running_mean(slopes_train[ent_ind_train],K), running_mean(entropy_train[ent_ind_train],K)
  slopes_test_smooth, entropy_test_smooth = running_mean(slopes_test[ent_ind_test],K), running_mean(entropy_test[ent_ind_test],K)

  #Min-max normalization of slopes and entropy w.r.t. training and test sets:
  slopes_min, slopes_max = np.min([slopes_train_smooth.min(),slopes_test_smooth.min()]), np.max([slopes_train_smooth.max(),slopes_test_smooth.max()])
  entropy_min, entropy_max = np.min([entropy_train_smooth.min(),entropy_test_smooth.min()]), np.max([entropy_train_smooth.max(),entropy_test_smooth.max()])  
  slopes_min = np.min([slopes_train_smooth.min(),slopes_test_smooth.min()])
  slopes_train_smooth -= slopes_min
  slopes_test_smooth -= slopes_min
  slopes_max = np.max([slopes_train_smooth.max(),slopes_test_smooth.max()])
  slopes_train_smooth /= slopes_max
  slopes_test_smooth /= slopes_max
  
  entropy_min = np.min([entropy_train_smooth.min(),entropy_test_smooth.min()])
  entropy_train_smooth -= entropy_min
  entropy_test_smooth -= entropy_min
  entropy_max = np.max([entropy_train_smooth.max(),entropy_test_smooth.max()])
  entropy_train_smooth /= entropy_max
  entropy_test_smooth /= entropy_max

  return [slopes_train_smooth, entropy_train_smooth, slopes_test_smooth, entropy_test_smooth]

def curve_est(x, y, x2, y2, points=15): #Function for linear interpolation of curves    
    f = interp1d(x, y)
    f2 = interp1d(x2,y2)
        
    xnew = np.linspace ( min(x), max(x), num = points) 
    xnew2 = np.linspace ( min(x2), max(x2), num = points) 
    
    ynew = f(xnew) 
    ynew2 = f2(xnew2) 
    
    return xnew, ynew, xnew2, ynew2

#CNA-Area. Area between inscribed polygon of slope-entropy train-test curves:
def CNA_A(train_loader, test_loader, net, flatten=True):

  #Estimated, smoothed coordinates of slope-entropy curve:
  xnew, ynew, xnew2, ynew2 = curve_est(*get_normalized_slope_entropy_curves(net, train_loader, test_loader, flatten=True), points = 100)

  #Getting maximum circumscribed polygon:
  array1 = np.array([xnew, ynew]).T
  array2 = np.array([xnew2, ynew2]).T
  array2 = np.flip(array2,axis=0)
  polygon_points = np.concatenate((array1,array2,array1[0].reshape([1,2])),axis=0)
  
  polygon = Polygon(polygon_points)
  area = polygon.area
  
  return area

#Margin calculation. Getting the maximum interclass decision boundary of the network:
def margin_calc(data_loader, net, flatten=True):
    margin = torch.Tensor([])
    if gpu_boole:
        margin = margin.cuda()
    for i, (x, y) in enumerate(data_loader):
        if gpu_boole:
            x, y = x.cuda(), y.cuda()
        if flatten:
            x = x.view(x.shape[0], -1)
        output_m = net(x)
        for k in range(y.size(0)):
            output_m[k, y[k]] = output_m[k,:].min()
            margin = torch.cat((margin, output_m[:, y].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
    margin = margin.cpu().data.numpy()
    val_margin = np.percentile( margin, 5 )    
    return val_margin

#CNA-Margin. Generalization gap metric based on CNA principles. 
#State-of-the-art for generalization gap prediction (as of 2019):
def CNA_M(train_loader, test_loader, net, flatten=True):
  cna_a = CNA_A(train_loader, test_loader, net, flatten=flatten)
  tr_margin = margin_calc(train_loader, net, flatten=flatten)
  cna_m = cna_a * tr_margin
  return cna_m

def CNA_all(data_loader, net, flatten=True):
    slopes = get_slopes_all(data_loader, net, flatten)    
    ents = get_entropy_all(data_loader, net, flatten)
    return np.corrcoef(slopes, ents)[0,1]

def CNA_M_batch(net, x, y, test_loader, flatten=True):
  for x_test, y_test in test_loader:
    x_test, y_test = x_test, y_test
    break

  if gpu_boole:
    x_test, y_test = x_test.cuda(), y_test.cuda()
  if flatten:
    x_test = x_test.view(x_test.shape[0],-1)

  slopes_train, slopes_test = get_slopes(x, net), get_slopes(x_test, net)
  entropy_train, entropy_test = get_entropy_batch_np(x.cpu().data.numpy()), get_entropy_batch_np(x_test.cpu().data.numpy())

  ##Normalized curves:

  #Indices of sorted entropy:
  ent_ind_train = np.argsort(entropy_train)
  ent_ind_test = np.argsort(entropy_test)

  #Smoothing slope and entropy curves:
  K = 25
  slopes_train_smooth, entropy_train_smooth = running_mean(slopes_train[ent_ind_train],K), running_mean(entropy_train[ent_ind_train],K)
  slopes_test_smooth, entropy_test_smooth = running_mean(slopes_test[ent_ind_test],K), running_mean(entropy_test[ent_ind_test],K)

  #Min-max normalization of slopes and entropy w.r.t. training and test sets:
  slopes_min, slopes_max = np.min([slopes_train_smooth.min(),slopes_test_smooth.min()]), np.max([slopes_train_smooth.max(),slopes_test_smooth.max()])
  entropy_min, entropy_max = np.min([entropy_train_smooth.min(),entropy_test_smooth.min()]), np.max([entropy_train_smooth.max(),entropy_test_smooth.max()])  
  slopes_min = np.min([slopes_train_smooth.min(),slopes_test_smooth.min()])
  slopes_train_smooth -= slopes_min
  slopes_test_smooth -= slopes_min
  slopes_max = np.max([slopes_train_smooth.max(),slopes_test_smooth.max()])
  slopes_train_smooth /= slopes_max
  slopes_test_smooth /= slopes_max
  
  entropy_min = np.min([entropy_train_smooth.min(),entropy_test_smooth.min()])
  entropy_train_smooth -= entropy_min
  entropy_test_smooth -= entropy_min
  entropy_max = np.max([entropy_train_smooth.max(),entropy_test_smooth.max()])
  entropy_train_smooth /= entropy_max
  entropy_test_smooth /= entropy_max

  ## Interpolated Curves:
  x1, y1, x2, y2 = slopes_train_smooth, entropy_train_smooth, slopes_test_smooth, entropy_test_smooth
  points = 15

  f = interp1d(x1, y1)
  f2 = interp1d(x2, y2)
      
  xnew = np.linspace ( min(x1), max(x1), num = points) 
  xnew2 = np.linspace ( min(x2), max(x2), num = points) 
  
  ynew = f(xnew) 
  ynew2 = f2(xnew2)

  ## CNA-Area final calculation:

  #Getting maximum circumscribed polygon:
  array1 = np.array([xnew, ynew]).T
  array2 = np.array([xnew2, ynew2]).T
  array2 = np.flip(array2,axis=0)
  polygon_points = np.concatenate((array1,array2,array1[0].reshape([1,2])),axis=0)
  
  polygon = Polygon(polygon_points)
  area = polygon.area

  ## CNA-Margin:

  ## Margin for batch:
  if gpu_boole:
    x, y = x.cuda(), y.cuda()
  if flatten:
    x = x.view(x.shape[0], -1)
    output_m = net(x)
    for k in range(y.size(0)):
      output_m[k, y[k]] = output_m[k,:].min()
      margin = output_m[:, y].diag() - output_m[:, output_m.max(1)[1]].diag()
  margin = margin.cpu().data.numpy()
  tr_margin = np.percentile( margin, 5 )
  
  cna_m = area * tr_margin

  return cna_m

def get_dataset_loaders(dataset_name, batch_size = 128, transform = False):
    if not transform:
        transform = transforms.ToTensor()
        
    elif transform:
        transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
            std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
            ),
        ])
    
    if dataset_name == 'mnist':
        train_set = dataset.MNIST(root ='./data', transform=transform, train=True, download=True)
        test_set = dataset.MNIST(root ='./data', transform=transform, train=False, download=True)
    elif dataset_name == 'fmnist':
        train_set = dataset.FashionMNIST(root ='./data', transform=transform, train=True, download=True)
        test_set = dataset.FashionMNIST(root ='./data', transform=transform, train=False, download=True)        
    elif dataset_name == 'cifar10':
        train_set = dataset.CIFAR10(root ='./data', transform=transform, train=True, download=True)
        test_set = dataset.CIFAR10(root ='./data', transform=transform, train=False, download=True)
    elif dataset_name == 'cifar100':
        train_set = dataset.CIFAR100(root ='./data', transform=transform, train=True, download=True)
        test_set = dataset.CIFAR100(root ='./data', transform=transform, train=False, download=True)
    elif dataset_name == 'svhn':
        train_set = dataset.SVHN('.', download=True, split='train', transform=transform)        
        test_set = dataset.SVHN('.', download=True, split='test', transform=transform)
    elif dataset_name == 'imagenet32':
        xtrain, ytrain = imagenet_train()
        xtest, ytest = imagenet_test()

        xtrain = torch.Tensor(xtrain).float()
        xtest = torch.Tensor(xtest).float()
        ytrain = torch.Tensor(ytrain).long()
        ytest = torch.Tensor(ytest).long()

        train_set = torch.utils.data.TensorDataset(xtrain, ytrain)
        test_set = torch.utils.data.TensorDataset(xtest, ytest)        
    
    elif dataset == 'random':        
        xtrain = torch.Tensor(np.load('./data/xtrain_random.npy')).float()
        xtest = torch.Tensor(np.load('./data/xtest_random.npy')).float()
        ytrain = torch.Tensor(np.load('./data/ytrain_random.npy')).long()
        ytest = torch.Tensor(np.load('./data/ytest_random.npy')).long()      
        train_set = torch.utils.data.TensorDataset(xtrain, ytrain)
        test_set = torch.utils.data.TensorDataset(xtest, ytest)
        
        
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size = batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader        

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import pickle

## Functions for ImageNet32.
## Needs to be downloaded and placed in a local folder.
## URL: https://patrykchrabaszcz.github.io/Imagenet32/

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


## Add net definitions here.
## To load your model from `get_cna.py`, your model definition needs to be in the namespace.

class Net(nn.Module):
  def __init__(self, input_size, width, num_classes):
    super(Net, self).__init__()

    ##feedfoward layers:
    self.ff1 = nn.Linear(input_size, width) #input

    self.ff2 = nn.Linear(width, width) #hidden layers
    self.ff3 = nn.Linear(width, width)

    self.ff_out = nn.Linear(width, num_classes) #logit layer     

    ##activations:
    self.relu = nn.ReLU()

    #other activations:
    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

    ## Some normalization functions.
    ## Feel free to add them into the forward pass if you like.
    
    #dropout:
    self.do = nn.Dropout()

    #batch-normalization:
    self.bn1 = nn.BatchNorm1d(width)
    self.bn2 = nn.BatchNorm1d(width)
    self.bn3 = nn.BatchNorm1d(width)

                
  def forward(self, input_data):
    out = self.relu(self.ff1(input_data)) 
    out = self.relu(self.ff2(out)) 
    out = self.relu(self.ff3(out))
    out = self.ff_out(out)
    return out #returns class probabilities for each image