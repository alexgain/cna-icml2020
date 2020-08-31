import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

import struct
from copy import deepcopy
from time import time, sleep
import gc
import copy

# from sklearn.preprocessing import normalize
gpu_boole = torch.cuda.is_available()

def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def np_cov(m, rowvar=False):
    # Handles complex arrays too
    m = m.cpu().data.numpy()
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError('m has more than 2 dimensions')
    dtype = np.result_type(m, np.float64)
    m = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and m.shape[0] != 1:
        m = m.T
    if m.shape[0] == 0:
        return np.array([], dtype=dtype).reshape(0, 0)

    # Determine the normalization
    fact = m.shape[1] - 1
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    m -= np.mean(m, axis=1, keepdims=1)
    c = np.dot(m, m.T.conj())
    c *= np.true_divide(1, fact)
    c = c.squeeze()
    c = torch.from_numpy(c)
    if gpu_boole:
        c = c.cuda()
    return c


class MLP(nn.Module):
    def __init__(self, input_size, width=500, num_classes=10):
        super(MLP, self).__init__()

        ##feedfoward layers:

        bias_ind = True

        self.ff1 = nn.Linear(input_size, width, bias = bias_ind) #input

        self.ff2 = nn.Linear(width, width, bias = bias_ind) #hidden layers
        self.ff3 = nn.Linear(width, width, bias = bias_ind)
        self.ff4 = nn.Linear(width, width, bias = bias_ind)
        self.ff5 = nn.Linear(width, width, bias = bias_ind)

##        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        
        ##BN:
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        self.bn4 = nn.BatchNorm1d(width)
        self.bn5 = nn.BatchNorm1d(width)
        
    def forward(self, input_data):

        ##forward pass computation:
        
        out = self.relu(self.ff1(input_data)) #input

        out = self.relu(self.ff2(out)) #hidden layers
        out = self.relu(self.ff3(out))
        out = self.relu(self.ff4(out))
        out = self.relu(self.ff5(out))

#        out = self.relu(self.bn1(self.ff1(input_data)))
#
#        out = self.relu(self.bn2(self.ff2(out)))
#        out = self.relu(self.bn3(self.ff3(out)))
#        out = self.relu(self.bn4(self.ff4(out)))
#        out = self.relu(self.bn5(self.ff5(out)))

        out = self.ff_out(out)
##        out = self.sm(self.ff_out(out))
##        out = F.log_softmax(self.ff_out(out), dim=1)

        return out #returns class probabilities for each image

    def beta(self, x):

        acts = []
        
        out = self.ff1(x)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
        out = self.relu(out)
        # acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.ff2(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
        out = self.relu(out)
        # acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.ff3(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
        out = self.relu(out)
        # acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.ff4(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
        out = self.relu(out)
        # acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.ff5(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
        out = self.relu(out)
        # acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.ff_out(out)
        # out = self.sm(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
        
    ##        out = self.relu(self.ff5(out))
    ##        acts.append(out.mean(dim=1))
    ##        out = self.relu(self.ff_out(out))
    ##        acts.append(out.mean(dim=1))

        acts = np.array(acts)
        # acts /= acts.max()
        
        #calculating beta:
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

    def beta_torch(self, x):

        acts = []
        
        out = self.ff1(x)
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur))
        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur))

        out = self.ff2(out)
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur))
        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur))

        out = self.ff3(out)
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur))
        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur))

        out = self.ff4(out)
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur))
        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur))

        out = self.ff5(out)
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur))
        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur))

        out = self.ff_out(out)
        # out = self.sm(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur))
        
    ##        out = self.relu(self.ff5(out))
    ##        acts.append(out.mean(dim=1))
    ##        out = self.relu(self.ff_out(out))
    ##        acts.append(out.mean(dim=1))

        # acts = np.array(acts)
        # acts /= acts.max()
        
        #calculating beta:
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


    def get_activations(self,x):
        
        acts = []
        
        out = self.ff1(x) #input
        out = self.relu(out)
        acts.append(copy.copy(out.cpu().data.numpy()))
        
        out = self.ff2(out) #hidden layers
        out = self.relu(out)
        acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff3(out)
        out = self.relu(out)
        acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff4(out)
        out = self.relu(out)
        acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff5(out)
        out = self.relu(out)
        acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff_out(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))

        return np.array(acts)        

    def get_preactivations(self,x):
        
        acts = []
        
        out = self.ff1(x) #input
        acts.append(copy.copy(out.cpu().data.numpy()))
        out = self.relu(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))
        
        out = self.ff2(out) #hidden layers
        acts.append(copy.copy(out.cpu().data.numpy()))
        out = self.relu(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff3(out)
        acts.append(copy.copy(out.cpu().data.numpy()))
        out = self.relu(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff4(out)
        acts.append(copy.copy(out.cpu().data.numpy()))
        out = self.relu(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff5(out)
        acts.append(copy.copy(out.cpu().data.numpy()))
        out = self.relu(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))

        out = self.ff_out(out)
        # acts.append(copy.copy(out.cpu().data.numpy()))

        return np.array(acts)        


    def beta_error(self,x,errors):
        return self.beta(x)*(Variable(torch.Tensor(deepcopy(errors.cpu().data.numpy()))).view(-1,1))
    
    def forward_error(self,x,errors):
        return self.forward(x)*(Variable(torch.Tensor(deepcopy(errors.cpu().data.numpy()))).view(-1,1))

    def corr(self, input_data, entropy):
        slopes = self.beta(input_data)
        vx = slopes - slopes.mean()
        vy = entropy - entropy.mean()

        num = torch.sum(vx * vy)
        den = torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))

        return num / den


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def beta(self, x):
 
        acts = []

        out = self.bn1(self.conv1(x))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = F.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn2(self.conv2(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
   
        out += self.shortcut(x)
        # acts.append(out.view(out.shape[0],-1).sum(dim=1))

        out = F.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        
        return acts


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def beta(self, x):
 
        acts = []

        out = self.bn1(self.conv1(x))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = F.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())
 
        out = self.bn2(self.conv2(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = F.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn3(self.conv3(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

   
        out += self.shortcut(x)
        # acts.append(out.view(out.shape[0],-1).sum(dim=1))

        out = F.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        
        
        return acts


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels = 3, input_padding = 0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding= 3 + input_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.sm = nn.Softmax()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def beta(self, x):
        acts = []
        out = self.bn1(self.conv1(x))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = F.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())
        
        out2 = out.clone()
        layer1_children = list(self.layer1.children())        
        for k in range(len(layer1_children)): 
            # try:
            acts = acts + layer1_children[k].beta(out2.clone())
            # except:
            #     acts = acts + layer1_children[k].module.beta(out2.clone())
                
            out2 = layer1_children[k](out2)
        del layer1_children
        out = self.layer1(out)

        out2 = out.clone()
        layer2_children = list(self.layer2.children())        
        for k in range(len(layer2_children)):            
            # try:
            acts = acts + layer2_children[k].beta(out2.clone())
            # except:
            #     acts = acts + layer2_children[k].module.beta(out2.clone())
            out2 = layer2_children[k](out2)
        del layer2_children
        out = self.layer2(out)

        out2 = out.clone()
        layer3_children = list(self.layer3.children())        
        for k in range(len(layer3_children)):     
            # try:
            acts = acts + layer3_children[k].beta(out2.clone())
            # except:
            #     acts = acts + layer3_children[k].module.beta(out2.clone())                
            out2 = layer3_children[k](out2)
        del layer3_children
        out = self.layer3(out)

        out2 = out.clone()
        layer4_children = list(self.layer4.children())        
        for k in range(len(layer4_children)):
            # try:
            acts = acts + layer4_children[k].beta(out2.clone())
            # except:
            #     acts = acts + layer4_children[k].module.beta(out2.clone())
                
            out2 = layer4_children[k](out2)
        del layer4_children
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.sm(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        acts = np.array(acts)
        # acts /= acts.max()
                
        #calculating beta:
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

    def beta2(self, x):
        
        acts = []
        
        out = self.bn1(self.conv1(x))
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
   
        out = F.relu(out)
        out = self.layer1(out)
        acts.append((out.view(out.shape[0],-1).mean(dim=1))/(out.view(out.shape[0],-1).std(dim=1)))   
        out = F.relu(out)
        out = self.layer2(out)
        acts.append((out.view(out.shape[0],-1).mean(dim=1))/(out.view(out.shape[0],-1).std(dim=1)))   
        out = F.relu(out)
        out = self.layer3(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
   
        out = F.relu(out)
        out = self.layer4(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
   
        out = F.relu(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        acts.append((out.view(out.shape[0],-1).mean(dim=1)))
        # acts.append(out.view(out.shape[0],-1).sum(dim=1))
   
                
        #calculating beta:
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

        def corr(self, input_data, entropy):
            slopes = self.beta(input_data)
            vx = slopes - slopes.mean()
            vy = entropy - entropy.mean()
    
            num = torch.sum(vx * vy)
            den = torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))
    
            return num / den


def ResNet18(num_classes,channels,input_padding):
    return ResNet(BasicBlock, [2,2,2,2],num_classes,channels,input_padding)

def ResNet34(num_classes,channels,input_padding):
    return ResNet(BasicBlock, [3,4,6,3],num_classes,channels,input_padding)

def ResNet50(num_classes,channels,input_padding):
    return ResNet(Bottleneck, [3,4,6,3],num_classes,channels,input_padding)

def ResNet101(num_classes,channels,input_padding):
    return ResNet(Bottleneck, [3,4,23,3],num_classes,channels,input_padding)

def ResNet152(num_classes,channels,input_padding):
    return ResNet(Bottleneck, [3,8,36,3],num_classes,channels,input_padding)


class VGG18(nn.Module):
    def __init__(self, num_classes, channels = 3, input_padding = 0):
        super(VGG18, self).__init__()

        ##conv and mp layers:
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding = 3 + input_padding)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding = 1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding = 1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding = 1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding = 1)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding = 1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding = 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ap = nn.AvgPool2d(kernel_size=1, stride=1)

        ##dense layers:
        self.dense1 = nn.Linear(512, num_classes)
        
        ##activations:
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        self.do1 = nn.Dropout2d(p=0.4)
        self.do2 = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)

        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)

        self.bn11 = nn.BatchNorm2d(512)
        self.bn12 = nn.BatchNorm2d(512)
        self.bn13 = nn.BatchNorm2d(512)

        self.sm = nn.Softmax()
        
    def forward(self, input_data):

        out = self.relu(self.bn1(self.conv1(input_data)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.mp(out)

        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.mp(out)

        out = self.relu(self.bn5(self.conv5(out)))
        out = self.relu(self.bn6(self.conv6(out)))
        out = self.relu(self.bn7(self.conv7(out)))
        out = self.mp(out)

        out = self.relu(self.bn8(self.conv8(out)))
        out = self.relu(self.bn9(self.conv9(out)))
        out = self.relu(self.bn10(self.conv10(out)))
        out = self.mp(out)

        out = self.relu(self.bn11(self.conv11(out)))
        out = self.relu(self.bn12(self.conv12(out)))
        out = self.relu(self.bn13(self.conv13(out)))
        out = self.mp(out)
        out = self.ap(out)
        
        out = out.view(out.size(0), -1)
        out = self.dense1(out)

        return out 

    def beta(self, x):
        
        acts = []        

        out = self.bn1(self.conv1(x))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)        
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn2(self.conv2(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn3(self.conv3(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn4(self.conv4(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())
        
        out = self.mp(out)

        out = self.bn5(self.conv5(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn6(self.conv6(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn7(self.conv7(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())
        
        out = self.mp(out)

        out = self.bn8(self.conv8(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn9(self.conv9(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn10(self.conv10(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())
        
        out = self.mp(out)

        out = self.bn11(self.conv11(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn12(self.conv12(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.bn13(self.conv13(out))
        acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())


        out = self.relu(out)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        # acts.append(copy.copy(acts_cur).cpu().data.numpy())

        out = self.mp(out)
        out = self.ap(out)
        
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        # out = self.sm(out)
        acts_cur = out.view(out.shape[0],-1).sum(dim=1)
        # acts_cur = out.mean(dim=1).view(out.shape[0],-1).sum(dim=1)
        acts.append(copy.copy(acts_cur).cpu().data.numpy())
           
        acts = np.array(acts)
        # acts /= acts.max()
                
        #calculating beta:
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
    
    def get_activations(self, x):

        acts = []        

        out = self.bn1(self.conv1(x))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn2(self.conv2(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn3(self.conv3(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn4(self.conv4(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn5(self.conv5(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn6(self.conv6(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn7(self.conv7(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn8(self.conv8(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn9(self.conv9(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn10(self.conv10(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn11(self.conv11(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn12(self.conv12(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn13(self.conv13(out))
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.mp(out)
        out = self.ap(out)
        
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        # out = self.sm(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())
        
        return acts        

    def get_preactivations(self, x):

        acts = []        

        out = self.bn1(self.conv1(x))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn2(self.conv2(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn3(self.conv3(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn4(self.conv4(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn5(self.conv5(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn6(self.conv6(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn7(self.conv7(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn8(self.conv8(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn9(self.conv9(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn10(self.conv10(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        
        out = self.mp(out)

        out = self.bn11(self.conv11(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn12(self.conv12(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())

        out = self.bn13(self.conv13(out))
        acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.relu(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())


        out = self.mp(out)
        out = self.ap(out)
        
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        # out = self.sm(out)
        # acts.append(out.view(out.shape[0],-1).cpu().data.numpy())
        
        return acts        

        
    def corr(self, input_data, entropy):
        slopes = self.beta(input_data)
        vx = slopes - slopes.mean()
        vy = entropy - entropy.mean()

        num = torch.sum(vx * vy)
        den = torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))

        return num / den

def conv3x3(in_planes, out_planes, stride=1, input_padding=0):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1 + input_padding, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, channels = 3, input_padding = 0):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(channels,nStages[0],input_padding= 1 + input_padding)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        num_blocks = int(num_blocks)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class MLP_Ablate(nn.Module):
    def __init__(self, input_size, width, num_classes):
        super(MLP_Ablate, self).__init__()

        ##feedfoward layers:

        bias_ind = True

        self.ff1 = nn.Linear(input_size, width, bias = bias_ind) #input

        self.ff2 = nn.Linear(width, width, bias = bias_ind) #hidden layers
        self.ff3 = nn.Linear(width, width, bias = bias_ind)
        self.ff4 = nn.Linear(width, width, bias = bias_ind)
        self.ff5 = nn.Linear(width, width, bias = bias_ind)

##        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        self.ff_out = nn.Linear(width, 10, bias = bias_ind) #output     
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        
    def forward(self, input_data):

        ##forward pass computation:
        
        out = self.relu(self.ff1(input_data)) #input

        out = self.relu(self.ff2(out)) #hidden layers
        out = self.relu(self.ff3(out))
        out = self.relu(self.ff4(out))
        out = self.relu(self.ff5(out))

        out = self.ff_out(out)
##        out = self.sm(self.ff_out(out))
##        out = F.log_softmax(self.ff_out(out), dim=1)

        return out #returns class probabilities for each image


    def get_rand_neurons(self, input_data, p=0.1):

        neur_count = 0

        out = self.relu(self.ff1(input_data))
        neur_count += int(np.prod(out.shape[1:]))
        out = self.relu(self.ff2(out))
        neur_count += int(np.prod(out.shape[1:]))
        out = self.relu(self.ff3(out))
        neur_count += int(np.prod(out.shape[1:]))
        out = self.relu(self.ff4(out))
        neur_count += int(np.prod(out.shape[1:]))
        out = self.relu(self.ff5(out))
        neur_count += int(np.prod(out.shape[1:]))

        out = self.ff_out(out)
        neur_count += int(np.prod(out.shape[1:]))
        
##        neurons = Variable(torch.LongTensor(round(neur_count*p)).random_(0, neur_count))
##        neurons = Variable(torch.LongTensor(round(neur_count*p)).random_(0, round(200)))
        # neurons = Variable(torch.LongTensor(round(neur_count*p)).random_(0, 1200))
 
        neurons = np.random.randint(0,neur_count, size = int(np.round(neur_count*p)))
        
        neurons = Variable(torch.LongTensor(neurons))       
        if gpu_boole:
            neurons = neurons.cuda()

##        print("Rand neurons plot:")
##        plt.hist(neurons.data.numpy())
##        plt.show()


        return neurons
        

    def forward_ablate(self,input_data,k_neurons):

        BS_cur = input_data.shape[0]
        neur_count = 0
        
        out = self.relu(self.ff1(input_data))
        
        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        out = self.relu(self.ff2(out))

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        out = self.relu(self.ff3(out))

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        out = self.relu(self.ff4(out))

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        out = self.relu(self.ff5(out))

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        out = self.ff_out(out)

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        return out

    def forward_bap_ablate(self,input_data,input_data_bap,entropy,p=0.1):

        neurons_abl = self.get_top_neurons_bap(input_data_bap,entropy,p)

        return self.forward_ablate(input_data,neurons_abl)

    def forward_rand_ablate(self, input_data, p = 0.1):

        neurons_abl = self.get_rand_neurons(input_data,p)

        return self.forward_ablate(input_data,neurons_abl)

    def beta_ablate(self,input_data,k_neurons):

        acts = []

        BS_cur = input_data.shape[0]
        neur_count = 0
        
        out = self.ff1(input_data)
        
        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff2(out)

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff3(out)

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff4(out)

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff5(out)

        mask = torch.ones(1,*list(out.shape[1:]))
        ind_start = neur_count
        ind_end = int(neur_count + np.prod(out.shape[1:]))
        neur_count += int(np.prod(out.shape[1:]))
        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
        if len(k_neurons_sub) != 0:
            k_neurons_sub -= ind_start
            k_neurons_sub = k_neurons_sub.long()
            mask_neg = torch.ones(ind_end - ind_start)
            mask_neg[k_neurons_sub.cpu().data] = 0
            mask *= mask_neg.view(mask.shape)
            if gpu_boole:
                mask = mask.cuda()
            out = out * Variable(mask).float()

        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff_out(out)

##        mask = torch.ones(1,*list(out.shape[1:]))
##        ind_start = neur_count
##        ind_end = int(neur_count + np.prod(out.shape[1:]))
##        neur_count += int(np.prod(out.shape[1:]))
##        k_neurons_sub = k_neurons[(k_neurons >= ind_start) & (k_neurons<ind_end)]
##        if len(k_neurons_sub) != 0:
##            k_neurons_sub -= ind_start
##            k_neurons_sub = k_neurons_sub.long()
##            mask_neg = torch.ones(ind_end - ind_start)
##            mask_neg[k_neurons_sub.cpu().data] = 0
##            mask *= mask_neg.view(mask.shape)
##            if gpu_boole:
##                mask = mask.cuda()
##            out = out * Variable(mask).float()

        acts.append(out.view(out.shape[0],-1).sum(dim=1))

        #calculating beta:
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

    def corr_ablate(self, input_data, entropy, p = 0.1):

        neurons_abl = self.get_rand_neurons(input_data,p)

        slopes = self.beta_ablate(input_data, neurons_abl)
        vx = slopes - slopes.mean()
        vy = entropy - entropy.mean()

        num = torch.sum(vx * vy)
        den = torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))

        return num / den

    def beta(self, x):

        acts = []
        
        out = self.ff1(x) #input
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff2(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff3(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff4(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff5(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        out = self.relu(out)

        out = self.ff_out(out)
        acts.append(out.view(out.shape[0],-1).sum(dim=1))
        
    ##        out = self.relu(self.ff5(out))
    ##        acts.append(out.mean(dim=1))
    ##        out = self.relu(self.ff_out(out))
    ##        acts.append(out.mean(dim=1))

        #calculating beta:
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

    def corr(self, input_data, entropy):
        slopes = self.beta(input_data)
        vx = slopes - slopes.mean()
        vy = entropy - entropy.mean()

        num = torch.sum(vx * vy)
        den = torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2))

        return num / den


#####
##Pre-trained ImageNet ResNet
####
        
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockIN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockIN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckIN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckIN, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetIN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetIN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetIN(BasicBlockIN, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetIN(BasicBlockIN, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetIN(BottleneckIN, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetIN(BottleneckIN, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetIN(BottleneckIN, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


