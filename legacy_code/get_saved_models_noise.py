from datasets import *
from models import *
import sys
sys.path.insert(0, './utils/')
from util import *

##lists to run:
# models = ['VGG18','MLP','ResNet18','ResNet101','WideResNet28-20','WideResNet28-10']
models = ['ResNet18']
# datasets = ['mnist','fashion','svhn','cifar10','cifar100']
datasets = ['cifar100']
# datasets = ['cifar10','cifar100']
metrics = ['bam']
noises = [0.1,0.2,0.4,0.8,1.6]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

all_test_accs = []
all_bam = []

for model in models:
    
    model_test_accs = []
    model_bam = []
    
    if model == 'MLP':
        flatten = True
    else:
        flatten = False
    
    for dataset in datasets:

        for noise in noises:
            
            flatten, image_aug, input_shape, channels, input_padding, num_classes, epochs, img_dim = get_dataset_params(model, dataset)
            
            if model == 'VGG18' or "ResNet18":
                if dataset == 'cifar10' or dataset == 'cifar100':
                    epochs = 50        
            
            if model == 'MLP':
                epochs = 100
            
            N_sub = 0            
            if model == 'ResNet101' or model == 'WideResNet28-20':
                train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 0, image_aug = False, shuffle = False, BS=256)
            else:
                train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 0, image_aug = False, shuffle = False)
            
            if dataset != 'svhn':               
                noise_data = torch.normal(mean=0, std = noise*torch.ones([train_loader.dataset.train_data.shape[0],channels,img_dim,img_dim]))
            else:
                noise_data = torch.normal(mean=0, std = noise*torch.ones([train_loader.dataset.data.shape[0],channels,img_dim,img_dim]))
            
            
            if not os.path.exists('./data_noise'):
                os.makedirs('./data_noise')

            np.save('./data_noise/'+model+'_'+dataset+'_noise_'+str(int(noise*100))+'.npy',noise_data.cpu().data.numpy())

            # if dataset == 'mnist' or dataset == 'fashion':
            #     noise_data = noise_data.unsqueeze(1)
                    
            if model == 'MLP':
                my_net = MLP(input_size = input_shape, num_classes = num_classes)
            elif model == 'VGG18':
                my_net = VGG18(num_classes = num_classes, channels = channels, input_padding = input_padding)
            elif model == 'ResNet18':
                my_net = ResNet18(num_classes = num_classes, channels = channels, input_padding = input_padding)
            elif model == 'ResNet101':
                my_net = ResNet101(num_classes = num_classes, channels = channels, input_padding = input_padding)
            elif model == 'WideResNet28-20':
                my_net = Wide_ResNet(28, 20, 0.3, num_classes = num_classes, channels = channels, input_padding = input_padding)
            elif model == 'WideResNet28-10':
                my_net = Wide_ResNet(28, 10, 0.3, num_classes = num_classes, channels = channels, input_padding = input_padding)
            
            print(torch.cuda.device_count())
            print(torch.cuda.device_count() > 1)
            if torch.cuda.device_count() > 1:
                # my_net = nn.DataParallel(my_net)
                my_net = nn.DataParallel(my_net)
            
            my_net.to(device)
                
            LR = 0.02
            if dataset == "random":
                LR = 0.01
            
            loss_metric = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
            # optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)#, momentum = 0.9)
                    
            # from metrics import *
            def train_acc(verbose = 1, flatten=False, input_shape = 28*28):
                correct = 0
                total = 0
                loss_sum = 0
                for i, (images,labels) in enumerate(train_loader):
                    
                    cur_ind = i*train_loader.batch_size
                    images += noise_data[cur_ind:cur_ind+images.shape[0]]
                    
                    if flatten:
                        images = images.view(-1, input_shape)
                        
                    images=images.to(device)
                    labels=labels.to(device)
                        
                    images = Variable(images)
                    outputs = my_net(images)
                    _, predicted = torch.max(outputs.data, 1)
                ##    labels = torch.max(labels.float(),1)[1]
                ##    predicted = torch.round(outputs.data).view(-1).long() 
                    total += labels.size(0)
                    correct += (predicted.float() == labels.float()).sum()#.cpu()
            
                    loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()
                    
                if verbose:
                    print('Accuracy of the network on the train images: %f %%' % (100.0 * np.float(correct) / np.float(total)))
            
                return 100.0 * np.float(correct) / np.float(total), loss_sum
                
            def test_acc(verbose = 1, flatten=False, input_shape = 28*28):
                # Test the Model
                correct = 0
                total = 0
                loss_sum = 0
                for images, labels in test_loader:
    
                    images=images.to(device)
                    labels=labels.to(device)
    
                    if flatten:
                        images = images.view(-1, input_shape)
                    images = Variable(images)
                    outputs = my_net(images)
                    _, predicted = torch.max(outputs.data, 1)
                ##    labels = torch.max(labels.float(),1)[1]
                ##    predicted = torch.round(outputs.data).view(-1).long()
                    total += labels.size(0)
                    correct += (predicted.float() == labels.float()).sum()#.cpu()
            
                    loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()
            
                if verbose:
                    print('Accuracy of the network on the train images: %f %%' % (100.0 * np.float(correct) / np.float(total)))
            
                return 100.0 * np.float(correct) / np.float(total), loss_sum
                
            def bam_calc(verbose=1, flatten=False, input_shape = 28*28):
                
                slopes = []
            
                for images, labels in train_loader_bam:
                    if flatten:
                        images = Variable(images.view(-1, input_shape), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
            
                    # images = images.no_grad()
                    try:
                        slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
                    except:
                        slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
                        
            
                slopes = np.array(slopes)
            
                bam = np.corrcoef(slopes,entropy)[0,1]
                
                if verbose:
                    print('BAM Train:',bam)
            
                return bam
            
            def bam_calc_test(verbose=1, flatten=False, input_shape = 28*28):
                
                slopes = []
            
                for images, labels in test_loader_bam:
                    if flatten:
                        images = Variable(images.view(-1, input_shape), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
                    
                    # images = images.no_grad()
                    
                    try:
                        slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
                    except:
                        slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
            
                slopes = np.array(slopes)
                
                bam = np.corrcoef(slopes,entropy_test)[0,1]
                
                if verbose:
                    print('BAM Test:',bam)
            
                return bam
                
            train_acc_store = []
            test_acc_store = []
            train_loss_store = []
            test_loss_store = []
            
            ###training loop (w/corr):
            t1 = time()
            for epoch in range(epochs):
            
                ##time-keeping 1:
                time1 = time()
            
                for i, (x,y) in enumerate(train_loader):
                    
                    if model == 'WideResNet28-20' or model == 'WideResNet28-10':
                        print(i)
                                        
                    ##data preprocessing for optimization purposes:        
                    x = Variable(x)
                    y = Variable(y)
                        
                    ##cuda:
                    x=x.to(device)
                    y=y.to(device)

                    cur_ind = i*train_loader.batch_size
                    cur_noise = noise_data[cur_ind:cur_ind+x.shape[0]]
                    cur_noise = cur_noise.to(device)
                    x += cur_noise

                    if flatten:
                        x = x.view(-1,input_shape)
            
                    ###regular BP gradient update:
                    optimizer.zero_grad()
                    outputs = my_net.forward(x)
                    loss = loss_metric(outputs,y)# - 0.1*bam_test
                    loss.backward()
                            
                    ##performing update:
                    optimizer.step()
                    
                    ##printing statistics:
                    if model == 'ResNet101' or model == 'WideResNet28-20':
                        BS = 256
                    else:
                        BS = 128
                    if dataset == 'mnist' or dataset=='fashion':
                        N = 60000
                    if dataset == 'iris':
                        N = 120
                        BS = 120
                    if dataset == 'svhn':
                        N = 73257
                    if dataset == 'cifar10' or dataset=='cifar100' or dataset == 'random':
                        N = 50000
                    
                    if N_sub > 0:
                        N = N_sub
                    
                    if (i+1) % np.floor(N/BS) == 0:
                        print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                               %(epoch+1, epochs, i+1, N//BS, loss.data.item()))
            
                        train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)
                        test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape)
                        # bam_calc()
                        # bam_calc_test()
    
                        train_acc_store.append(train_perc)
                        train_loss_store.append(loss_train)
                        test_acc_store.append(test_perc)
                        test_loss_store.append(loss_test)
                                                
                    save_directory = './results/'+model+'/'+dataset+'/'
                    # if epoch ==1 or epoch % 5 == 0:
                    #     torch.save(my_net.state_dict(), './saved_models/'+model+'_'+dataset+'_epoch'+str(epoch)+'_noise'+str(int(noise*100))+'.state')
                    if epoch == epochs - 2:
                        torch.save(my_net.state_dict(), './saved_models/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'.state')
                        np.save('./results/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'_train_acc'+'.npy',train_perc)
                        np.save('./results/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'_test_acc'+'.npy',test_perc)
                        np.save('./results/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'_train_loss'+'.npy',loss_train)
                        np.save('./results/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'_test_loss'+'.npy',loss_test)
                        np.save('./results/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'_loss_gap'+'.npy',np.abs(loss_train-loss_test))
                        np.save('./results/'+model+'_'+dataset+'_noise'+str(int(noise*100))+'_acc_gap'+'.npy',np.abs(train_perc-test_perc))
                        
            
                ##time-keeping 2:
                time2 = time()
                print('Elapsed time for epoch:',time2 - time1,'s')
                print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
                print()
            
            t2 = time()
            print((t2 - t1)/60,'total minutes elapsed')                     
            
            # train_acc_store = np.array(train_acc_store)
            # train_loss_store = np.array(train_loss_store)
            # test_acc_store = np.array(test_acc_store)
            # test_loss_store = np.array(test_loss_store)
                    
            # if not os.path.exists('./results'):
            #     os.makedirs('./results')
            # if not os.path.exists('./results/'+model):
            #     os.makedirs('./results/'+model)
            # if not os.path.exists('./results/'+model+'/'+dataset):
            #     os.makedirs('./results/'+model+'/'+dataset)
                
            # save_directory = './results/'+model+'/'+dataset+'/'
                                    
            # np.save(save_directory+'train_acc.npy',train_acc_store)
            # np.save(save_directory+'train_loss.npy',train_loss_store)
            # np.save(save_directory+'test_acc.npy',test_acc_store)
            # np.save(save_directory+'test_loss.npy',test_loss_store)
            # np.save(save_directory+'hyperparams.npy',np.array([LR,epochs,BS]))
            # torch.save(my_net.state_dict(), './saved_models/'+model+'_'+dataset+'.state')
            
