import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

from torchvision import datasets, transforms

import struct
from copy import deepcopy
from time import time, sleep
import gc

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

np.random.seed(1)

gpu_boole = torch.cuda.is_available()

###                               ###
### Data import and preprocessing ###
###                               ###

N = 60000
##N = 1000
BS = 128
N2 = 500

transform_data = transforms.ToTensor()

train_set = datasets.MNIST('./data', train=True, download=True,
                   transform=transform_data)

train_set.train_data = train_set.train_data[:N]

##adding noise:
##noise_level = 0
##train_set.train_data = train_set.train_data.float()
##train_set.train_data = train_set.train_data + noise_level*torch.abs(torch.randn(*train_set.train_data.shape))
##train_set.train_data = train_set.train_data / train_set.train_data.max()

train_loader = torch.utils.data.DataLoader(train_set, batch_size = BS, shuffle=True)

train_loader_bap = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transform_data),
    batch_size=N2, shuffle=False)

test_dataset = datasets.MNIST(root='./data', 
                            train=False, 
                            transform=transform_data,
                           )
##adding noise to test:
##test_dataset.test_data = test_dataset.test_data.float()
##test_dataset.test_data = test_dataset.test_data + noise_level*torch.abs(torch.randn(*test_dataset.test_data.shape))
##test_dataset.test_data = test_dataset.test_data / test_dataset.test_data.max()

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=N2, shuffle=False)

test_loader_bap = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=N2, shuffle=False)


##adding noise to test:
##test_dataset.test_data = test_dataset.test_data.float()
##test_dataset.test_data = test_dataset.test_data + noise_level*torch.abs(torch.randn(*test_dataset.test_data.shape))
##test_dataset.test_data = test_dataset.test_data / test_dataset.test_data.max()

##test_loader = torch.utils.data.DataLoader(
##    test_dataset,
##    batch_size=10000, shuffle=False)
##
##test_loader_bap = torch.utils.data.DataLoader(
##    test_dataset,
##    batch_size=N2, shuffle=False)

##test_loader = torch.utils.data.DataLoader(
##    datasets.MNIST('./data', train=False, transform=transforms.Compose([
##                       transforms.ToTensor(),
##                       transforms.Normalize((0.1307,), (0.3081,))
##                   ])),
##    batch_size=10000, shuffle=False)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
##ytrain = to_categorical(ytrain, 3)
##ytest = to_categorical(ytest, 3)

entropy = np.load('./data/entropy_mnist_train.npy')[:N2]
entropy_test = np.load('./data/entropy_mnist_test.npy')[:N2]
##np.random.shuffle(entropy)

###                      ###
### Define torch network ###
###                      ###

class Net(nn.Module):
    def __init__(self, input_size, width, num_classes):
        super(Net, self).__init__()

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
        neurons = Variable(torch.LongTensor(round(neur_count*p)).random_(0, 1200))

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

        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
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

        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
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

        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
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

        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
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

        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
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

        out = self.sm(out)
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))

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
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
        out = self.relu(out)

        out = self.ff2(out)
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
        out = self.relu(out)

        out = self.ff3(out)
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
        out = self.relu(out)

        out = self.ff4(out)
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
        out = self.relu(out)

        out = self.ff5(out)
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
        out = self.relu(out)

        out = self.ff_out(out)
        out = self.sm(out)
        acts.append(out.sum(dim=1)*(int(np.prod(list(out.shape[1:])))))
        
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


###hyper-parameters:
input_size = 28*28
width = 750
num_classes = 10

###defining network:        
my_net = Net(input_size, width, num_classes)
if gpu_boole:
    my_net = my_net.cuda()

###                       ###
### Loss and optimization ###
###                       ###

LR = 0.01
##loss_metric = nn.MSELoss()
loss_metric = nn.CrossEntropyLoss()
##loss_metric = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
##optimizer = torch.optim.RMSprop(my_net.parameters(), lr = 0.00001)
##optimizer = torch.optim.RMSprop(my_net.parameters(), lr = 0.00001, momentum = 0.8)
##optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)

###                           ###
### Alternative BAP calc func ###
###                           ###

def CO_calc(model,x,y):
##    slopes = []
##    for i in range(x.shape[0]//100):
##        slopes = slopes + list(model.beta(x[i*100:(i+1)*100].cuda()).cpu().data.numpy())
    slopes = model.beta(x.cuda()).cpu().data.numpy()
    slopes = np.array(slopes)
    return np.corrcoef(slopes,y[:slopes.shape[0]])[0,1]


###          ###
### Training ###
###          ###

#Some more hyper-params and initializations:
epochs = 0

##BS = 32

##train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
##test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)

##printing train statistics:
def train_acc(verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in train_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long() 
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        loss_sum += loss_metric(outputs,Variable(labels))
        
    if verbose:
        print('Accuracy of the network on the train images: %f %%' % (100 * correct / total))

    return 100.0 * correct / total, loss_sum.cpu().data.numpy().item()
    
def test_acc(verbose = 1):
    # Test the Model
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in test_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        loss_sum += loss_metric(outputs,Variable(labels))

    if verbose:
        print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))

    return 100.0 * correct / total, loss_sum.cpu().data.numpy().item()
    
def bap_val(verbose = 1):
    for images, labels in train_loader_bap:
        images = Variable(images.view(-1, 28*28))
        if gpu_boole:
            images = images.cuda()
        bap = my_net.corr(images,entropy)
        break
    if verbose:
        print('BAP value:',bap.cpu().data.numpy().item())
    return bap.cpu().data.numpy().item()

def bap_val_test(verbose = 1):
    for images, labels in test_loader_bap:
        images = Variable(images.view(-1, 28*28))
        if gpu_boole:
            images = images.cuda()
        bap = my_net.corr(images,entropy_test)
        break
    if verbose:
        print('BAP value:',bap.cpu().data.numpy().item())
    return bap.cpu().data.numpy().item()

    
entropy = torch.Tensor(entropy)
if gpu_boole:
    entropy = entropy.cuda()
entropy = Variable(entropy)

entropy_test = torch.Tensor(entropy_test)
if gpu_boole:
    entropy_test = entropy_test.cuda()
entropy_test = Variable(entropy_test)

bap_train_max = -1
bap_test_max = -1

bap_train_store = []
bap_test_store = []
loss_train_store = []
loss_test_store = []
train_perc_store = []
test_perc_store = []
##bap_train_sum = 0
##bap_test_sum = 0
##bap_train_abs_sum = 0
##bap_test_abs_sum = 0
##num_of_batches = 0

def bap_abl_val_test(p=0.1, verbose = 0):

    correct = 0
    total = 0
    for images, labels in test_loader_bap:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        break
    
    bap = my_net.corr_ablate(images,entropy_test,float(p))
    bap_cpu = bap.cpu().data.numpy().item()

    if verbose:
        print('Ablation accuracy (Random) of the network on the 10000 test images: %f %%' % bap_cpu)
    
    return bap_cpu

bap_abl_val_test(verbose=1)


train_perc, loss_train = train_acc()
test_perc, loss_test = test_acc()
bap_train = bap_val()
bap_test = bap_val_test()

bap_train_store.append(bap_train)
bap_test_store.append(bap_train)
loss_train_store.append(loss_train)
loss_test_store.append(loss_test)
train_perc_store.append(train_perc)
test_perc_store.append(test_perc)


###training loop (w/corr):
t1 = time()
for epoch in range(epochs):

    ##time-keeping 1:
    time1 = time()

    for i, (x,y) in enumerate(train_loader):

        ##some pre-processing:
        x = x.view(-1,28*28)
##        y = y.float()
##        y = y.long()
##        y = torch.Tensor(to_categorical(y.long().cpu().numpy(),num_classes)) #MSE

        ##cuda:
        if gpu_boole:
            x = x.cuda()
            y = y.cuda()

        ##data preprocessing for optimization purposes:        
        x = Variable(x)
        y = Variable(y) #MSE 1-d output version


        ###regular BP gradient update:
        optimizer.zero_grad()
        outputs = my_net.forward(x)
        loss = loss_metric(outputs,y)# - 0.1*bap_test
        loss.backward()
                
        ##performing update:
        optimizer.step()
        
        ##printing statistics:
        if (i+1) % np.floor(N/BS) == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, epochs, i+1, N//BS, loss.data[0]))

            train_perc, loss_train = train_acc()
            test_perc, loss_test = test_acc()
            bap_train = bap_val()
            bap_test = bap_val_test()
            if bap_train_max < bap_train:
                bap_train_max = bap_train
            if bap_test_max < bap_test:
                bap_test_max = bap_test

            bap_train_store.append(bap_train)
            bap_test_store.append(bap_test)
            loss_train_store.append(loss_train)
            loss_test_store.append(loss_test)
            train_perc_store.append(train_perc)
            test_perc_store.append(test_perc)
            

    ##time-keeping 2:
    time2 = time()
    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()

t2 = time()
print((t2 - t1)/60,'total minutes elapsed')
             

bap_train_store = np.array(bap_train_store)
bap_test_store = np.array(bap_test_store)
loss_train_store = np.array(loss_train_store)
loss_test_store = np.array(loss_test_store)
train_perc_store = np.array(train_perc_store)
test_perc_store = np.array(test_perc_store)

print()
print("BAP train sum:",bap_train_store[1:].sum())
print("BAP train abs sum:",bap_train_store[1:].sum()+1)
print("BAP train sum / epochs:", bap_train_store[1:].sum() / epochs)
print("BAP train abs sum / epochs:", (bap_train_store[1:].sum() + 1) / epochs)
print()

print("BAP test sum:",bap_test_store[1:].sum())
print("BAP test abs sum:",bap_test_store[1:].sum()+1)
print("BAP test sum / batches:", bap_test_store[1:].sum() / epochs)
print("BAP test abs sum / batches:", (bap_test_store[1:].sum() + 1) / epochs)
print()

print("BAP train max:", bap_train_max)
print("BAP test max:", bap_test_max)
print()

print("Hyperparams:")
print("(LR, epochs, BS)",(LR,epochs,BS))
print()

print("Full arrays:")
print("BAP train:",bap_train_store)
print("BAP test:",bap_test_store)
print("Loss train:",loss_train_store)
print("Loss test:",loss_test_store)
print("Train Acc.:",train_perc_store)
print("Test Acc:",test_perc_store)

np.save('bap_train.npy',bap_train_store)
np.save('bap_test.npy',bap_test_store)
np.save('loss_train.npy',loss_train_store)
np.save('loss_test.npy',loss_test_store)
np.save('train_perc.npy',train_perc_store)
np.save('test_perc.npy',test_perc_store)
np.save('hyperparams_bap_maxes.npy',np.array([LR,epochs,BS,bap_train_max,bap_test_max]))
np.save('bap_train_areas.npy',np.array([bap_train_store[1:].sum(),bap_train_store[1:].sum()+1,bap_train_store[1:].sum() / epochs,bap_train_store[1:].sum() / epochs]))
np.save('bap_test_areas.npy',np.array([bap_test_store[1:].sum(),bap_test_store[1:].sum()+1,bap_test_store[1:].sum() / epochs,bap_test_store[1:].sum() / epochs]))

###                      ###
### Ablation experiments ###
###                      ###

def test_acc_abl(p=0.1, verbose = 0):

    correct = 0
    total = 0
    for images, labels in test_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net.forward_rand_ablate(images,p)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

    if verbose:
        print('Ablation accuracy (Random) of the network on the 10000 test images: %f %%' % (100 * correct / total))
    
    return 100 * correct / total

def bap_abl_val_test(p=0.1, verbose = 0):

    correct = 0
    total = 0
    for images, labels in test_loader_bap:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        break
    
    bap = my_net.corr_ablate(images,entropy_test,float(p))
    bap_cpu = bap.cpu().data.numpy().item()

    if verbose:
        print('BAP test with ablation:', bap_cpu)
    
    return bap_cpu

my_net.load_state_dict(torch.load('./data/mlp_mnist_net.pt'))

p_list = list(np.arange(0,1,0.05))[1:]
bap_abl_store = []
test_acc_store = []
trials = 20

for i in range(len(p_list)):
    
    print(i)
    bap_tot = 0
    test_acc_tot = 0
    
    for trial in range(trials):
        bap_tot += bap_abl_val_test(p_list[i])
        test_acc_tot += test_acc_abl(p_list[i])
        
    bap_tot /= trials
    test_acc_tot /= trials
    
    bap_abl_store.append(bap_tot)
    test_acc_store.append(test_acc_tot)

bap_abl_store = np.array(bap_abl_store)
test_acc_store = np.array(test_acc_store)


















