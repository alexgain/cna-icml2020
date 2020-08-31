from datasets import *
from models import *

##lists to run:
#models = ['VGG18','MLP','ResNet18']
models = ['MLP']
datasets = ['mnist']
#datasets = ['mnist','fashion','svhn','cifar10','cifar100']
metrics_list = ['bam']

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
        train_loader, test_loader, train_loader_bap, test_loader_bap, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, N2 = 60000, image_aug = image_aug, shuffle = False, BS = 128)
        num_classes = 10
        
        if model == 'MLP':
            epochs = 40
        elif model == 'ResNet18' or model == 'VGG18':
            if dataset == 'mnist' or dataset == 'fashion' or dataset == 'svhn':
                epochs = 30
        
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
        
        if gpu_boole:
            my_net = my_net.cuda()
            
        # LR = 0.000001
        LR = 0.01
        
        loss_metric = nn.CrossEntropyLoss()
        # loss_metric = nn.MSELoss()
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
                    
                if gpu_boole:
                    images, labels = images.cuda(), labels.cuda()
                images = Variable(images)
                outputs = my_net(images)
                _, predicted = torch.max(outputs.data, 1)
            ##    labels = torch.max(labels.float(),1)[1]
            ##    predicted = torch.round(outputs.data).view(-1).long() 
                total += labels.size(0)
                correct += (predicted.float() == labels.float()).sum()#.cpu()
                # correct += (torch.round(outputs).view(-1) == labels.float()).sum()#.cpu()
                loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()
                # loss_sum += loss_metric(outputs.view(-1),Variable(labels).float()).cpu().data.numpy().item()
                
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
                # correct += (torch.round(outputs).view(-1) == labels.float()).sum()#.cpu()

                loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()        
                # loss_sum += loss_metric(outputs.view(-1),Variable(labels).float()).cpu().data.numpy().item()
        
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
                
                # slopes = slopes + list(my_net.beta_torch(images))
                slopes = slopes + list(my_net.beta_torch(images).cpu().data.numpy())

        
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
                
                # slopes = slopes + list(my_net.beta_torch(images))
                slopes = slopes + list(my_net.beta_torch(images).cpu().data.numpy())
        
            slopes = np.array(slopes)
            
            bap = np.corrcoef(slopes,entropy_test)[0,1]
            
            if verbose:
                print('BAP Test:',bap)
        
            return bap
        
        def get_class_conditional_errors(y_batch, y_hat, loss = None):
            
            y_batch, y_hat = y_batch.cpu().data.numpy(), y_hat.cpu().data.numpy()
            
            if loss != None:
                loss = loss
            else:
                loss = lambda x, y: (x - y)**2
            
            class_errors = dict()
            classes = np.unique(y_batch)
            for c_num in classes:
                # c_num_errors = loss(torch.Tensor(y_hat[y_batch==c_num.item()]).float(), torch.Tensor(y_batch[y_batch==c_num.item()]).float()).cpu().data.numpy()
                c_num_errors = loss(torch.Tensor(y_hat[y_batch==c_num.item()]).float(), torch.Tensor(y_batch[y_batch==c_num.item()]).long()).cpu().data.numpy()
                class_errors[c_num] = c_num_errors
            
            return class_errors        

        def get_class_conditional_entropy(entropy, y_batch):
        
            y_batch = y_batch.cpu().data.numpy()
                
            class_entropy = dict()
            classes = np.unique(y_batch)
            for c_num in classes:
                c_num_entropy = entropy[y_batch==c_num.item()]
                class_entropy[c_num] = c_num_entropy
            
            return class_entropy

        def conditional_corr(x,y):
            #assumes x and y are dicts.
            list1 = []
            list2 = []
            for key in x:
                list1.append(x[key].mean())
                list2.append(y[key].mean())
            list1, list2 = np.array(list1), np.array(list2)
            
            return np.corrcoef(list1,list2)[0,1]


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
        
        cce_list = []
        bap_changes = []
        prev_bap = bap_train
        
        act_store = []
        
        
        ###training loop (w/corr):
        t1 = time()
        for epoch in range(epochs):
            cce = 0
            
            ##time-keeping 1:
            time1 = time()
        
            for i, (x,y) in enumerate(train_loader):
                
                ##cuda:
                if gpu_boole:
                    x = x.cuda()
                    y = y.cuda()
        
                ##data preprocessing for optimization purposes:        
                x = Variable(x)
                y = Variable(y)
                if flatten:
                    x = x.view(-1,input_shape)
                    
                params = np.concatenate([x[1].cpu().data.numpy().reshape([-1]) for x in list(my_net.named_parameters()) if x[0][:2]!='bn'])               
                
                # ###getting net angle::
                # optimizer.zero_grad()
                # outputs = my_net.forward(x)
                # loss = outputs.mean()
                # loss.backward()
                   
                # bp_angle = []
                # for p in my_net.parameters():
                #     if p.grad is not None:
                #         bp_angle += list(p.grad.cpu().data.numpy().flatten())
                # bp_angle = deepcopy(np.array(bp_angle))
 
                # ###getting beta angle::
                # optimizer.zero_grad()
                # outputs = my_net.beta(x)
                # loss = outputs.mean()
                # loss.backward()
                   
                # beta_angle = []
                # for p in my_net.parameters():
                #     if p.grad is not None:
                #         beta_angle += list(p.grad.cpu().data.numpy().flatten())
                # beta_angle = deepcopy(np.array(beta_angle))
 
                # ##printing angle:
                # def get_angle(a1,b1):
                #     return (180/np.pi)*np.arccos(a1.dot(b1)/(np.linalg.norm(a1)*np.linalg.norm(b1)))
                # print('beta-net angle:',get_angle(bp_angle,beta_angle))          
                
                ###regular BP gradient update:
                optimizer.zero_grad()
                outputs = my_net.forward(x)
                # errors = outputs.view(-1) - y.float()
                
                class_conditional_errors = get_class_conditional_errors(y, outputs, loss = nn.CrossEntropyLoss(reduce=False))
                class_conditional_entropy = get_class_conditional_entropy(entropy[i*BS:(i*BS+y.shape[0])], y)
                
                cce += conditional_corr(class_conditional_errors,class_conditional_entropy)
                
                # loss = loss_metric(outputs,y)# - 0.1*bap_test
                # loss = 2*my_net.beta_error(x,errors).mean()
                # loss = 2*my_net.forward_error(x,errors).mean()
                # loss = nn.MSELoss()(outputs.view(-1),y.float())
                loss = loss_metric(outputs,y)
                loss.backward()
                               
                # error_i = (outputs - y.float()).view(-1)
                # for p in my_net.parameters():
                #     if p.grad is not None:
                #         p.grad *= error_i  # or whatever other operation
                if epoch >0:
                    optimizer.step()                
                        
                ##performing update:
                # if epoch >0:
                #     optimizer.step()
                
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
        
                    print('Class conditional error-complexity corr:',cce/(i+1))

                    ###getting net angle::
                    optimizer.zero_grad()
                    outputs = my_net.forward(x)
                    loss = outputs.mean()
                    loss.backward()
                       
                    params = list(my_net.parameters())
                    bp_angle = []
                    for j in range(len(params)):
                        if params[j].grad is not None and j<=len(params)-3:
                            bp_angle += list(params[j].grad.cpu().data.numpy().flatten())
                    bp_angle = deepcopy(np.array(bp_angle))
     
                    ###getting beta angle::
                    optimizer.zero_grad()
                    outputs = my_net.beta_torch(x)
                    loss = outputs.mean()
                    loss.backward()
                    
                    params = list(my_net.parameters())
                    beta_angle = []
                    for j in range(len(params)):
                        if params[j].grad is not None and j<=len(params)-3:
                            beta_angle += list(params[j].grad.cpu().data.numpy().flatten())
                    beta_angle = deepcopy(np.array(beta_angle))
     
                    ##printing angle:
                    def get_angle(a1,b1):
                        return (180/np.pi)*np.arccos(a1.dot(b1)/(np.linalg.norm(a1)*np.linalg.norm(b1)))
                    print('beta-net angle:',get_angle(bp_angle,beta_angle))          

                    cce_list.append(cce/(i+1))
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
                        
                    bap_changes.append(bap_train - prev_bap)
                    prev_bap = bap_train
        
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
        
        if not os.path.exists('./results'):
            os.makedirs('./results')
        if not os.path.exists('./results/'+model):
            os.makedirs('./results/'+model)
        if not os.path.exists('./results/'+model+'/'+dataset):
            os.makedirs('./results/'+model+'/'+dataset)
            
        save_directory = './results/'+model+'/'+dataset+'/'
        
        all_test_accs.append(test_perc_store[-1])
        all_bam.append(bap_test_store[-1])
        model_test_accs.append(test_perc_store[-1])
        model_bam.append(bap_test_store[-1])
        
        
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

    if not os.path.exists('./results/'+model):
        os.makedirs('./results/'+model)
        
    save_directory = './results/'+model + '/'
        
    model_test_accs = np.array(model_test_accs)
    model_bam = np.array(model_bam)
    np.save(save_directory+'model_test_accs.npy',model_test_accs)
    np.save(save_directory+'model_bam.npy',model_bam)    

    data_arrays = [(model_test_accs,model_bam)]
    titles = [model+' Test Acc. vs. BAM']
    #save_files = ['bap_corrupt_train.png','bap_corrupt_test.png','bap_corrupt_diff.png','bap_corrupt_abs.png']
    for i in range(len(data_arrays)):
        plt.plot(data_arrays[i][0], data_arrays[i][1], 'o', color = 'blue', markersize = 8, linewidth = 2.0)
        plt.xlabel("Percentage", fontsize = 16)
        plt.ylabel("Value", fontsize = 16)
        plt.title(titles[i], fontsize = 20)
        plt.savefig(save_directory + titles[i]+'.png')
        plt.clf()
    #plt.show()


if not os.path.exists('./results'):
    os.makedirs('./results')
    
save_directory = './results/'
 
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
