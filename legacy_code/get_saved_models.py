from datasets import *
from models import *
import sys
sys.path.insert(0, './utils/')
from util import *

##lists to run:
# models = ['VGG18','MLP','ResNet18','ResNet101','WideResNet28-20','WideResNet28-10']
models = ['MLP']
#datasets = ['mnist','fashion','svhn','cifar10','cifar100']
datasets = ['imagenet32']#,'fashion','svhn']
# datasets = ['random']
metrics = ['bam']

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

        flatten, image_aug, input_shape, channels, input_padding, num_classes, epochs, img_dim = get_dataset_params(model, dataset)
                
        N_sub = 0            
        if model == 'ResNet101' or model == 'WideResNet28-20':
            train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 0, image_aug = image_aug, BS=512)
        else:
            train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 0, image_aug = image_aug, BS=512)
                
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
            for images, labels in train_loader:
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
                correct += (predicted.float() == labels.float()).sum().cpu().data.numpy().item()#.cpu()
        
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
                correct += (predicted.float() == labels.float()).sum().cpu().data.numpy().item()#.cpu()
        
                loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()
        
            if verbose:
                print('Accuracy of the network on the test images: %f %%' % (100.0 * np.float(correct) / np.float(total)))
        
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
        # train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)        
        # test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape)
        
        state_dict = torch.load('./saved_models/'+model+'_'+dataset+'.state')
        my_net.load_state_dict(state_dict)
        
        t1 = time()
        for epoch in range(epochs):
        
            ##time-keeping 1:
            time1 = time()
        
            for i, (x,y) in enumerate(train_loader):
                
                
                if i%10==0:
                    print(i)
                
                if model == 'WideResNet28-20' or model == 'WideResNet28-10':
                    print(i)
                                    
                ##data preprocessing for optimization purposes:        
                x = Variable(x)
                y = Variable(y)

                ##cuda:
                x=x.to(device)
                y=y.to(device)
                if flatten:
                    x = x.view(-1,input_shape)
        
                if dataset=='imagenet':
                    y = y.long()
                            
                ###regular BP gradient update:
                optimizer.zero_grad()
                outputs = my_net.forward(x)
                loss = loss_metric(outputs,y)# - 0.1*bam_test
                loss.backward()
                        
                ##performing update:
                optimizer.step()
                
                ##printing statistics:
                if model == 'ResNet101' or model == 'WideResNet28-20':
                    BS = 512
                else:
                    BS = 512
                if dataset == 'mnist' or dataset=='fashion':
                    N = 60000
                if dataset == 'iris':
                    N = 120
                    BS = 120
                if dataset == 'svhn':
                    N = 73257
                if dataset == 'cifar10' or dataset=='cifar100' or dataset == 'random':
                    N = 50000
                if dataset == 'imagenet32':
                    N = 1281167
                    
                
                if N_sub > 0:
                    N = N_sub
                
                if (i+1) % np.floor(N/BS) == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                           %(epoch+1, epochs, i+1, N//BS, loss.data.item()))
        
                    if model !='ResNet101' or dataset != 'imagenet32':
                        # train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)
                        test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape)
                        # bam_calc()
                        # bam_calc_test()
    
                        # train_acc_store.append(train_perc)
                        # train_loss_store.append(loss_train)
                        test_acc_store.append(test_perc)
                        test_loss_store.append(loss_test)
                    
            if not os.path.exists('./results'):
                os.makedirs('./results')
            if not os.path.exists('./results/'+model):
                os.makedirs('./results/'+model)
            if not os.path.exists('./results/'+model+'/'+dataset):
                os.makedirs('./results/'+model+'/'+dataset)
                    
            save_directory = './results/'+model+'/'+dataset+'/'
            if epoch == 1 or epoch % 20 == 0:
                torch.save(my_net.state_dict(), './saved_models/'+model+'_'+dataset+'_epoch'+str(epoch)+'.state')

            if epoch%10 ==0 and epoch != 0 and model != 'ResNet101':
                train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)        
        
            ##time-keeping 2:
            time2 = time()
            print('Elapsed time for epoch:',time2 - time1,'s')
            print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
            print()

        if model != 'ResNet101':
            train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)        
            test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape)

        t2 = time()        
        print((t2 - t1)/60,'total minutes elapsed')                     
        
        train_acc_store = np.array(train_acc_store)
        train_loss_store = np.array(train_loss_store)
        test_acc_store = np.array(test_acc_store)
        test_loss_store = np.array(test_loss_store)
                
        if not os.path.exists('./results'):
            os.makedirs('./results')
        if not os.path.exists('./results/'+model):
            os.makedirs('./results/'+model)
        if not os.path.exists('./results/'+model+'/'+dataset):
            os.makedirs('./results/'+model+'/'+dataset)
            
        save_directory = './results/'+model+'/'+dataset+'/'
                                
        np.save(save_directory+'train_acc.npy',train_acc_store)
        np.save(save_directory+'train_loss.npy',train_loss_store)
        np.save(save_directory+'test_acc.npy',test_acc_store)
        np.save(save_directory+'test_loss.npy',test_loss_store)
        np.save(save_directory+'hyperparams.npy',np.array([LR,epochs,BS]))
        torch.save(my_net.state_dict(), './saved_models/'+model+'_'+dataset+'.state')
        
