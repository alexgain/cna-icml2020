from datasets import *
from models import *

##lists to run:
models = ['MLP']
datasets = ['mnist']#,'fashion','svhn','cifar10','cifar100']
metrics = ['bam']
saved_models = {'MLP':'./data/mlp_mnist_net.pt'}
ablation_levels = np.arange(0.1,1,0.1)
trials = 20

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
            
        if (dataset == 'cifar10' or dataset == 'cifar100'):
            image_aug = True
        else:
            image_aug = False
        
        N_sub = 0
        train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 10000, image_aug = image_aug, ablate=True)
        
        num_classes = 10
        
        if model == 'MLP':
            epochs = 20
        elif model == 'ResNet18' or model == 'VGG18':
            if dataset == 'mnist' or dataset == 'fashion' or dataset == 'svhn':
                epochs = 10
        
        if dataset == 'mnist' or dataset == 'fashion':
            input_shape = 28*28
            channels = 1
            input_padding = 2
        else:
            input_shape = 32*32*3
            channels = 3
            input_padding = 0

        if dataset == 'cifar100':
            num_classes = 100
                    
        
        if model == 'MLP':
            my_net = MLP_Ablate(width=750,input_size = input_shape, num_classes = num_classes)
        elif model == 'VGG18':
            my_net = VGG18(num_classes = num_classes, channels = channels, input_padding = input_padding)
        elif model == 'ResNet18':
            my_net = ResNet18(num_classes = num_classes, channels = channels, input_padding = input_padding)

        for layer in my_net.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        
        if model in saved_models:
            my_net.load_state_dict(torch.load(saved_models[model]))
        
        if gpu_boole:
            my_net = my_net.cuda()
            
        LR = 0.01
        
        loss_metric = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
        # optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)#, momentum = 0.9)
                
        def train_acc(verbose = 1, flatten=False, input_shape = 28*28):
            correct = 0
            total = 0
            loss_sum = 0
            for images, labels in train_loader:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                if gpu_boole:
                    images, labels = images.cuda(), labels.cuda()
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
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                if gpu_boole:
                    images, labels = images.cuda(), labels.cuda()
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
            
        def bap_val(verbose = 1, flatten=False, input_shape = 28*28):
            for images, labels in train_loader_bap:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                bap = my_net.corr(images,entropy)
                break
            if verbose:
                print('BAP value:',bap.cpu().data.numpy().item())
            return bap.cpu().data.numpy().item()
        
        def bap_val_test(verbose = 1, flatten=False, input_shape = 28*28):
            for images, labels in test_loader_bap:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                bap = my_net.corr(images,entropy_test)
                break
            if verbose:
                print('BAP value:',bap.cpu().data.numpy().item())
            return bap.cpu().data.numpy().item()
        
        def bap_calc(verbose=1, flatten=False, input_shape = 28*28):
            
            slopes = []
        
            for images, labels in train_loader_bap:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                
                slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
        
            slopes = np.array(slopes)
        
            bap = np.corrcoef(slopes,entropy)[0,1]
            
            if verbose:
                print('BAP Train:',bap)
        
            return bap
        
        def bap_calc_test(verbose=1, flatten=False, input_shape = 28*28):
            
            slopes = []
        
            for images, labels in test_loader_bap:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                
                slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
        
            slopes = np.array(slopes)
            
            bap = np.corrcoef(slopes,entropy_test)[0,1]
            
            if verbose:
                print('BAP Test:',bap)
        
            return bap
        
        def test_acc_abl(p=0.1, verbose = 0, flatten = False, input_shape = 28*28):
        
            correct = 0
            total = 0
            for images, labels in test_loader:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = my_net.forward_rand_ablate(images,p)
                _, predicted = torch.max(outputs.data, 1)
            ##    labels = torch.max(labels.float(),1)[1]
            ##    predicted = torch.round(outputs.data).view(-1).long()
                total += labels.size(0)
                correct += (predicted.float() == labels.float()).sum()
        
            if verbose:
                print('Ablation accuracy (Random) of the network on the 10000 test images: %f %%' % (100 * correct / total))
            
            return 100 * correct / total

        def bap_abl_val_test(p=0.1, verbose = 0, flatten=False, input_shape = 28*28):
        
            correct = 0
            total = 0
            for images, labels in test_loader_bap:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                break
            
            entropy_proxy = Variable(torch.Tensor(entropy_test))
            if gpu_boole:
                entropy_proxy = entropy_proxy.cuda()
                        
            bap = my_net.corr_ablate(images,entropy_proxy,float(p))
            bap_cpu = bap.cpu().data.numpy().item()
        
            if verbose:
                print('BAP test with ablation:', bap_cpu)
            
            return bap_cpu

        
        def bap_abl_val_test2(p=0.1, verbose=0, flatten=False, input_shape = 28*28):
            
            slopes = []
        
            for images, labels in test_loader_bap:
                if flatten:
                    images = Variable(images.view(-1, input_shape))
                images = Variable(images)
                if gpu_boole:
                    images = images.cuda()
                
                slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
        
            slopes = Variable(torch.Tensor(np.array(slopes)))
            if gpu_boole:
                slopes = slopes.cuda()
            
            bap = my_net.corr_ablate(images,entropy_test,float(p))
            bap_cpu = bap.cpu().data.numpy().item()
            
            if verbose:
                print('BAP test with ablation:', bap_cpu)
        
            return bap_cpu
    
        bap_abl_store = []
        test_acc_store = []
        
        print('Starting ablation experiments...')
        
        for i in range(ablation_levels.shape[0]):
            
            bap_tot = 0
            test_acc_tot = 0
            
            for trial in range(trials):
                bap_tot += bap_abl_val_test(ablation_levels[i],flatten=flatten,input_shape=input_shape)
                test_acc_tot += test_acc_abl(ablation_levels[i],flatten=flatten,input_shape=input_shape)
                
            bap_tot /= trials
            test_acc_tot /= trials
            
            bap_abl_store.append(bap_tot)
            test_acc_store.append(test_acc_tot)
            all_bam.append(bap_tot)
            all_test_accs.append(test_acc_tot)

            print('Ablation Level:',ablation_levels[i],'%;','Test Acc:',test_acc_tot.cpu().data.numpy().item(),'%;','BAM:',bap_tot)

        
        bap_abl_store = np.array(bap_abl_store)
        test_acc_store = np.array(test_acc_store)
        

        if not os.path.exists('./results_ablation'):
            os.makedirs('./results_ablation')
            
        if not os.path.exists('./results_ablation'):
            os.makedirs('./results_ablation')
        if not os.path.exists('./results_ablation/'+model):
            os.makedirs('./results_ablation/'+model)
        if not os.path.exists('./results_ablation/'+model+'/'+dataset):
            os.makedirs('./results_ablation/'+model+'/'+dataset)
            
        save_directory = './results_ablation/'+ model + '/' + dataset + '/'
        
        np.save(save_directory+'test_accs.npy',test_acc_store)
        np.save(save_directory+'bam.npy',bap_abl_store)  
        np.save(save_directory+'ablation_levels.npy',ablation_levels)
    
        data_arrays = [(test_acc_store,bap_abl_store), (ablation_levels,bap_abl_store)]
        titles = ['('+model+','+dataset+')'+' Test Acc. vs. BAM','('+model+','+dataset+')'+' Ablation. vs. BAM']
        xlabels = ['Percentage','Ablation Percentage']
        #save_files = ['bap_corrupt_train.png','bap_corrupt_test.png','bap_corrupt_diff.png','bap_corrupt_abs.png']
        for i in range(len(data_arrays)):
            plt.plot(data_arrays[i][0], data_arrays[i][1], 'o', color = 'blue', markersize = 8, linewidth = 2.0)
            plt.xlabel(xlabels[i], fontsize = 16)
            plt.ylabel("Value", fontsize = 16)
            plt.title(titles[i], fontsize = 20)
            plt.savefig(save_directory + titles[i]+'.png')
            plt.clf()
        #plt.show()

        
save_directory = './results_ablation/'
all_test_accs = np.array(all_test_accs)
all_bam = np.array(all_bam)
np.save(save_directory+'model_test_accs.npy',all_test_accs)
np.save(save_directory+'model_bam.npy',all_bam)    

data_arrays = [(all_test_accs,all_bam)]
titles = ['Total Test Acc. vs. BAM']
#save_files = ['bap_corrupt_train.png','bap_corrupt_test.png','bap_corrupt_diff.png','bap_corrupt_abs.png']
for i in range(len(data_arrays)):
    plt.plot(data_arrays[i][0], data_arrays[i][1], 'o', color = 'blue', markersize = 8, linewidth = 2.0)
    plt.xlabel("Percentage", fontsize = 16)
    plt.ylabel("Value", fontsize = 16)
    plt.title(titles[i], fontsize = 20)
    plt.savefig(save_directory + titles[i]+'.png')
    plt.clf()
#plt.show()
