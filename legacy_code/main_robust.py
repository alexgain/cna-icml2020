from datasets import *
from models import *

##lists to run:
models = ['VGG18','ResNet18']
datasets = ['mnist','fashion','svhn','cifar10','cifar100']#,'fashion','svhn','cifar10','cifar100']
metrics = ['bam']

robust_metrics = ['noise']
noise_levels = np.arange(0,6,1.2)
# ablation_levels = np.arange(0,1,0.1)

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
        
        md_test_accs = []
        md_bam = []
        
        if model == 'MLP' and (dataset=='mnist' or dataset=='fashion' or dataset=='svhn'):
            continue
        
        for eps in noise_levels:
    
            if (dataset == 'cifar10' or dataset == 'cifar100'):
                image_aug = True
            else:
                image_aug = False
            image_aug = False
            
            N_sub = 0
            train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 10000, image_aug = image_aug)
            
            num_classes = 10
            
            if model == 'MLP':
                epochs = 50
            elif model == 'ResNet18' or model == 'VGG18':
                if dataset == 'mnist' or dataset == 'fashion' or dataset == 'svhn':
                    epochs = 30
                else:
                    epochs = 70
            
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
                my_net = MLP(input_size = input_shape, num_classes = num_classes)
            elif model == 'VGG18':
                my_net = VGG18(num_classes = num_classes, channels = channels, input_padding = input_padding)
            elif model == 'ResNet18':
                my_net = ResNet18(num_classes = num_classes, channels = channels, input_padding = input_padding)

            for layer in my_net.children():
               if hasattr(layer, 'reset_parameters'):
                   layer.reset_parameters()
            
            if gpu_boole:
                my_net = my_net.cuda()
                
            LR = 0.01
            
            loss_metric = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
            # optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)#, momentum = 0.9)
            #optimizer = torch.optim.RMSprop(my_net.parameters(), lr = LR/2)
                    
            # from metrics import *
            def train_acc(verbose = 1, flatten=False, input_shape = 28*28):
                correct = 0
                total = 0
                loss_sum = 0
                for images, labels in train_loader:
                    if flatten:
                        images = images.view(-1, input_shape)
                        
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
                    if gpu_boole:
                        images, labels = images.cuda(), labels.cuda()
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
                        images = Variable(images.view(-1, input_shape), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
    
                    # images = images.no_grad()
                    
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
                        images = Variable(images.view(-1, input_shape), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
                    
                    # images = images.no_grad()
                    
                    slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
            
                slopes = np.array(slopes)
                
                bap = np.corrcoef(slopes,entropy_test)[0,1]
                
                if verbose:
                    print('BAP Test:',bap)
            
                return bap
    
            bap_train_max = -1
            bap_test_max = -1
            
            bap_train_store = []
            bap_test_store = []
            loss_train_store = []
            loss_test_store = []
            train_perc_store = []
            test_perc_store = []
            
            train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)
            test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape)
            bap_train = bap_calc(flatten=flatten, input_shape=input_shape)
            bap_test = bap_calc_test(flatten=flatten, input_shape=input_shape)
            
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
                    
                    ##cuda:
                    if gpu_boole:
                        x = x.cuda()
                        y = y.cuda()
            
                    ##noise:
                    if abs(eps) > 0:
                        noise = torch.abs(eps*torch.randn(*list(x.shape)))
                        if gpu_boole:
                            noise = noise.cuda()
                        x += noise
                        x -= x.min()
                        x /= x.max()
            
                    ##data preprocessing for optimization purposes:        
                    x = Variable(x)
                    y = Variable(y)
                    if flatten:
                        x = x.view(-1,input_shape)
                                            
                    ###regular BP gradient update:
                    optimizer.zero_grad()
                    outputs = my_net.forward(x)
                    loss = loss_metric(outputs,y)# - 0.1*bap_test
                    loss.backward()
                            
                    ##performing update:
                    optimizer.step()
                    
                    ##printing statistics:
                    BS = 128
                    if dataset == 'mnist' or dataset=='fashion':
                        N = 60000
                    if dataset == 'svhn':
                        N = 73257
                    if dataset == 'cifar10' or dataset=='cifar100':
                        N = 50000
                    
                    if N_sub > 0:
                        N = N_sub
                    
                    if (i+1) % np.floor(N/BS) == 0:
                        print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                               %(epoch+1, epochs, i+1, N//BS, loss.data[0]))
            
                        train_perc, loss_train = train_acc(flatten=flatten, input_shape=input_shape)
                        test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape)
            #            bap_train = bap_calc()
            #            bap_test = bap_calc_test()
                        bap_train = bap_calc(flatten=flatten, input_shape=input_shape)
                        bap_test = bap_calc_test(flatten=flatten, input_shape=input_shape)
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
            
            if not os.path.exists('./results_noise'):
                os.makedirs('./results_noise')
            if not os.path.exists('./results_noise/'+model):
                os.makedirs('./results_noise/'+model)
            if not os.path.exists('./results_noise/'+model+'/'+dataset):
                os.makedirs('./results_noise/'+model+'/'+dataset)
            if not os.path.exists('./results_noise/'+ model + '/' + dataset + '/' + str(eps) + '/'):
                os.makedirs('./results_noise/'+ model + '/' + dataset + '/' + str(eps) + '/')
                
            save_directory = './results_noise/'+ model + '/' + dataset + '/' + str(eps) + '/'
            
            all_test_accs.append(test_perc_store[-1])
            all_bam.append(bap_test_store[-1])
            model_test_accs.append(test_perc_store[-1])
            model_bam.append(bap_test_store[-1])
            md_test_accs.append(test_perc_store[-1])
            md_bam.append(bap_test_store[-1])
            
            
            # print()
            # print("BAP train sum:",bap_train_store.sum())
            # print("BAP train abs sum:",bap_train_store.sum()+1)
            # print("BAP train sum / epochs:", bap_train_store.sum() / epochs)
            # print("BAP train abs sum / epochs:", (bap_train_store.sum() + 1) / epochs)
            # print()
            
            # print("BAP test sum:",bap_test_store.sum())
            # print("BAP test abs sum:",bap_test_store.sum()+1)
            # print("BAP test sum / epochs:", bap_test_store.sum() / epochs)
            # print("BAP test abs sum / epochs:", (bap_test_store.sum() + 1) / epochs)
            # print()
            
            # print("BAP train max:", bap_train_max)
            # print("BAP test max:", bap_test_max)
            # print()
            
            # print("Correlations:", np.array([np.corrcoef(bap_train_store[1:],loss_train_store[1:])[0,1],np.corrcoef(bap_test_store[1:],loss_test_store[1:])[0,1],np.corrcoef(bap_train_store[1:],train_perc_store[1:])[0,1],np.corrcoef(bap_test_store[1:],test_perc_store[1:])[0,1]]))
            # print()
            
            # print("Hyperparams:")
            # print("(LR, epochs, BS)",(LR,epochs,BS))
            # print()
            
            # print("Full arrays:")
            # print("BAP train:",bap_train_store)
            # print("BAP test:",bap_test_store)
            # print("Loss train:",loss_train_store)
            # print("Loss test:",loss_test_store)
            # print("Train Acc.:",train_perc_store)
            # print("Test Acc:",test_perc_store)
            
            np.save(save_directory+'bap_train.npy',bap_train_store)
            np.save(save_directory+'bap_test.npy',bap_test_store)
            np.save(save_directory+'loss_train.npy',loss_train_store)
            np.save(save_directory+'loss_test.npy',loss_test_store)
            np.save(save_directory+'train_perc.npy',train_perc_store)
            np.save(save_directory+'test_perc.npy',test_perc_store)
            np.save(save_directory+'hyperparams_bap_maxes.npy',np.array([LR,epochs,BS,bap_train_max,bap_test_max]))
            np.save(save_directory+'bap_loss_acc_corr.npy',np.array([np.corrcoef(bap_train_store,loss_train_store)[0,1],np.corrcoef(bap_test_store,loss_test_store)[0,1],np.corrcoef(bap_train_store,train_perc_store)[0,1],np.corrcoef(bap_test_store,test_perc_store)[0,1]]))
            np.save(save_directory+'bap_train_areas.npy',np.array([bap_train_store.sum(),bap_train_store.sum()+1,bap_train_store.sum() / epochs,bap_train_store.sum() / epochs]))
            np.save(save_directory+'bap_test_areas.npy',np.array([bap_test_store.sum(),bap_test_store.sum()+1,bap_test_store.sum() / epochs,bap_test_store.sum() / epochs]))
        
        save_directory = './results_noise/'+ model + '/' + dataset
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        save_directory = save_directory + '/'

        md_test_accs = np.array(md_test_accs)
        md_bam = np.array(md_bam)
        np.save(save_directory+'md_test_accs.npy',md_test_accs)
        np.save(save_directory+'md_bam.npy',md_bam)
        np.save(save_directory+'noise_levels.npy',noise_levels)

        data_arrays = [(md_test_accs,md_bam), (noise_levels,md_bam)]
        titles = [model+' '+dataset+' Test Acc. vs. BAM',model+' '+dataset+' Noise. vs. BAM']
        xlabels = ['Percentage','Noise']
        #save_files = ['bap_corrupt_train.png','bap_corrupt_test.png','bap_corrupt_diff.png','bap_corrupt_abs.png']
        for i in range(len(data_arrays)):
            plt.plot(data_arrays[i][0], data_arrays[i][1], 'o', color = 'blue', markersize = 8, linewidth = 2.0)
            plt.xlabel(xlabels[i], fontsize = 16)
            plt.ylabel("Value", fontsize = 16)
            plt.title(titles[i], fontsize = 20)
            plt.savefig(save_directory + titles[i]+'.png')
            plt.clf()
        #plt.show()
    
    
    if not os.path.exists('./results_noise/'+model):
        os.makedirs('./results_noise/'+model)
        
    save_directory = './results_noise/'+model + '/'
        
    model_test_accs = np.array(model_test_accs)
    model_bam = np.array(model_bam)
    np.save(save_directory+'model_test_accs.npy',model_test_accs)
    np.save(save_directory+'model_bam.npy',model_bam)  
    np.save(save_directory+'noise_levels.npy',noise_levels)

    data_arrays = [(model_test_accs,model_bam), (np.array([list(noise_levels)*len(datasets)]).reshape(-1),model_bam)]
    titles = [model+' Test Acc. vs. BAM',model+' Noise. vs. BAM']
    xlabels = ['Percentage','Noise']
    #save_files = ['bap_corrupt_train.png','bap_corrupt_test.png','bap_corrupt_diff.png','bap_corrupt_abs.png']
    for i in range(len(data_arrays)):
        plt.plot(data_arrays[i][0], data_arrays[i][1], 'o', color = 'blue', markersize = 8, linewidth = 2.0)
        plt.xlabel(xlabels[i], fontsize = 16)
        plt.ylabel("Value", fontsize = 16)
        plt.title(titles[i], fontsize = 20)
        plt.savefig(save_directory + titles[i]+'.png')
        plt.clf()
    #plt.show()


if not os.path.exists('./results_noise'):
    os.makedirs('./results_noise')
    
save_directory = './results_noise/'
 
all_test_accs = np.array(all_test_accs)
all_bam = np.array(all_bam)
np.save(save_directory+'all_test_accs.npy',all_test_accs)
np.save(save_directory+'all_bam.npy',all_bam)    

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
