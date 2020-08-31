from datasets import *
from models import *
import sklearn.datasets, sklearn.decomposition
from mpl_toolkits.mplot3d import Axes3D

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
            epochs = 20
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
            my_net = MLP(input_size = input_shape, num_classes = num_classes,width=50)
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
                
                slopes = slopes + list(my_net.beta(images))
                # slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
        
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
                
                slopes = slopes + list(my_net.beta(images))
                # slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
        
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
                c_num_errors = loss(torch.Tensor(y_hat[y_batch==c_num.item()]), torch.Tensor(y_batch[y_batch==c_num.item()]).long()).cpu().data.numpy()
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
        
        param_store = []
        bam_param_store = []
        test_param_store = []
        loss_param_store = []
        
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
                param_store.append(params)
                # test_perc, loss_test = test_acc(flatten=flatten, input_shape=input_shape,verbose=False)
                # test_param_store.append(test_perc)
                # loss_param_store.append(loss_test)
                # bap_test = bap_calc_test(flatten=flatten, input_shape=input_shape, verbose = False)   
                # bam_param_store.append(bap_test)

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
                
                # class_conditional_errors = get_class_conditional_errors(y, outputs, loss = nn.CrossEntropyLoss(reduce=False))
                # class_conditional_entropy = get_class_conditional_entropy(entropy[i*BS:(i*BS+y.shape[0])], y)
                
                # cce += conditional_corr(class_conditional_errors,class_conditional_entropy)
                
                # loss = loss_metric(outputs,y)# - 0.1*bap_test
                # loss = 2*my_net.beta_error(x,errors).mean()
                # loss = 2*my_net.forward_error(x,errors).mean()
                # loss = nn.MSELoss()(outputs.view(-1),y.float())
                loss = nn.CrossEntropyLoss()(outputs,y)
                loss.backward()
                               
                # error_i = (outputs - y.float()).view(-1)
                # for p in my_net.parameters():
                #     if p.grad is not None:
                #         p.grad *= error_i  # or whatever other operation
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
        
                    # print('Class conditional error-complexity corr:',cce/(i+1))

                    # ###getting net angle::
                    # optimizer.zero_grad()
                    # outputs = my_net.forward(x)
                    # loss = outputs.mean()
                    # loss.backward()
                       
                    # params = list(my_net.parameters())
                    # bp_angle = []
                    # for j in range(len(params)):
                    #     if params[j].grad is not None and j<=len(params)-3:
                    #         bp_angle += list(params[j].grad.cpu().data.numpy().flatten())
                    # bp_angle = deepcopy(np.array(bp_angle))
     
                    # ###getting beta angle::
                    # optimizer.zero_grad()
                    # outputs = my_net.beta(x)
                    # loss = outputs.mean()
                    # loss.backward()
                    
                    # params = list(my_net.parameters())
                    # beta_angle = []
                    # for j in range(len(params)):
                    #     if params[j].grad is not None and j<=len(params)-3:
                    #         beta_angle += list(params[j].grad.cpu().data.numpy().flatten())
                    # beta_angle = deepcopy(np.array(beta_angle))
     
                    # ##printing angle:
                    # def get_angle(a1,b1):
                    #     return (180/np.pi)*np.arccos(a1.dot(b1)/(np.linalg.norm(a1)*np.linalg.norm(b1)))
                    # print('beta-net angle:',get_angle(bp_angle,beta_angle))          

                    # cce_list.append(cce/(i+1))

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


###PCA

param_store = np.array(param_store)
bam_param_store = np.array(bam_param_store)
test_param_store = np.array(test_param_store)
loss_param_store = np.array(loss_param_store)
# param_store = np.load('param_store.npy')
N = param_store.shape[0]
N = 1000
param_store_sub = param_store[::round(param_store.shape[0]/N)]
mu = np.mean(param_store_sub, axis=0)
pca = sklearn.decomposition.PCA()
pca.fit(param_store_sub)
nComp = 2

param_store_tf = pca.transform(param_store_sub)[:,:nComp]

def inv_pca(x):
    if len(x.shape) == 1:
        x = x.reshape([1,-1])
    xhat = np.dot(x[:,:nComp], pca.components_[:nComp,:])
    xhat += mu
    return xhat

param_eps = param_store[::469]

# plt.plot(param_store_tf[:,0],param_store_tf[:,1],'o')


def set_net_params(net, p_instance):
    p_instance = inv_pca(p_instance)[0]
    model_dict = net.state_dict()
    cur_ind = 0
    for key in model_dict.keys():
        if key[:2]=='bn':
            continue
        else:
            cur_shape = list(model_dict[key].shape)
            cur_N = int(np.prod(cur_shape))
            model_dict[key] = torch.Tensor(p_instance[cur_ind:cur_ind+cur_N].reshape(cur_shape))
            cur_ind += cur_N
    net = net.load_state_dict(model_dict)
            
def set_net_true(net, p_instance):
    # p_instance = inv_pca(p_instance)[0]
    model_dict = net.state_dict()
    cur_ind = 0
    for key in model_dict.keys():
        if key[:2]=='bn':
            continue
        else:
            cur_shape = list(model_dict[key].shape)
            cur_N = int(np.prod(cur_shape))
            model_dict[key] = torch.Tensor(p_instance[cur_ind:cur_ind+cur_N].reshape(cur_shape))
            cur_ind += cur_N
    net = net.load_state_dict(model_dict)
                
    
##sample BAM:
sampling_rt = 1.0
x = np.arange(param_store_tf[:,0].min()-1, param_store_tf[:,0].max(), sampling_rt)
y = np.arange(param_store_tf[:,1].min()-1, param_store_tf[:,1].max(), sampling_rt)
bam_mesh = np.zeros([y.shape[0],x.shape[0]])
count = 0
print('# of evaluations:',x.shape[0]*y.shape[0])
for counterx, x1 in enumerate(x):
    for countery, y1 in enumerate(y):
        t1 = time()
        set_net_params(my_net, np.array([x1,y1]))
        bam_mesh[countery][counterx] = bap_calc_test(flatten=flatten, input_shape=input_shape, verbose = False)
        t2 = time()
        if count % 5 == 0:
            print('ETA:',((t2-t1)/60)*(x.shape[0]*y.shape[0] - count),'minutes')
        count += 1
        
        
##2D plot:
cmap_cur = 'viridis'

bam_mesh -= bam_mesh.min()
bam_mesh /= bam_mesh.max()
plt.contourf(x, y, bam_mesh, 20, cmap=cmap_cur)
# plt.plot(param_store_tf[::50,0],param_store_tf[::50,1],'>',c='black',linewidth=7,markersize=11)
# plt.plot(param_store_tf[60,0],param_store_tf[60,1],'o',color='blue',markersize=25)
plt.text(param_store_tf[60,0], param_store_tf[60,1], 'START', bbox=dict(facecolor='red', alpha=0.9))
plt.text(param_store_tf[-1,0], param_store_tf[-1,1], 'END', bbox=dict(facecolor='red', alpha=0.9))
plt.plot(param_store_tf[::50,0],param_store_tf[::50,1],'->',c='red',linewidth=2.5,markersize=6,label='Optimization Path')
plt.plot(param_store_tf[::50,0],param_store_tf[::50,1],'>',c='black',linewidth=7,markersize=11)
plt.plot(param_store_tf[::50,0],param_store_tf[::50,1],'>',c='red',linewidth=7,markersize=6)
plt.ylim(y.min(),y.max())
plt.xlim(x.min(),x.max())
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
# plt.gca().invert_xaxis()
plt.colorbar(label="BAM Value")
plt.legend(loc='lower center')#, bbox_to_anchor=(0.3, 0.1))
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.savefig('optimization-path-2D.png',bbox_inches = "tight", dpi=500)
        

## 3D-plot:

xflat = np.stack([x]*bam_mesh.shape[0],axis=1).flatten()
yflat = np.stack([y]*bam_mesh.shape[1],axis=0).flatten()
zflat = bam_mesh.T.flatten()

def fmt(y1, x1):
    
    # get closest point with known data
    dist = np.linalg.norm(np.vstack([yflat - y1, xflat - x1]), axis=0)
    idx = np.argmin(dist)
    z1 = zflat[idx]
    return z1


opt_path_3d = []
for i in range(param_store_tf.shape[0]):
    x_cur = param_store_tf[i,0]
    y_cur = param_store_tf[i,1]
    z_cur = fmt(y_cur,x_cur)
    opt_path_3d.append([x_cur,y_cur,fmt(y_cur,x_cur)])
    
opt_path_3d = np.array(opt_path_3d)
opt_path_3d = opt_path_3d[30:][::50]


# bam_mesh_1d = []
# xtot = len(param_store_tf[::50,0])
# ytot = len(param_store_tf[::50,0])
# xtot /= bam_mesh.shape[0]
# ytot /= bam_mesh.shape[1]

# for xcount,x1 in enumerate(param_store_tf[::50,0]):
#     for ycount,y1 in enumerate(param_store_tf[::50,1]):
#         x_rat = np.floor((xcount/param_store_tf[::50,0].shape[0])*bam_mesh.shape[0])
#         y_rat = np.floor((ycount/param_store_tf[::50,1].shape[0])*bam_mesh.shape[1])
#         bam_mesh_1d.append(bam_mesh[int(x_rat),int(y_rat)])
        
# bam_mesh_1d = np.array(bam_mesh_1d)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, Y = np.mgrid[param_store_tf[:,0].min()-1:param_store_tf[:,0].max():sampling_rt,param_store_tf[:,1].min()-1:param_store_tf[:,1].max():sampling_rt]
ax.plot(opt_path_3d[:,0],opt_path_3d[:,1],opt_path_3d[:,2],'->',c='red',linewidth=1.5,markersize=5.5,label='Optimization Path',zorder=100)
ax.plot_surface(X.T, Y.T, bam_mesh,cmap=cmap_cur,lw=0.5, rstride=1, cstride=1, alpha=0.95)
ax.contour(X.T, Y.T, bam_mesh, linestyles="solid", cmap=cmap_cur, linewidth=4)


# ax.set_ylim(y.min(),y.max()-1)
# ax.set_xlim(x.min(),x.max()-1)

ax.view_init(azim=70, elev = 50)
# ax.view_init(azim=10)

# ax.plot(param_store_tf[::50,0],param_store_tf[::50,1],'>',zdir='x',c='black',linewidth=7,markersize=11)
# ax.plot(param_store_tf[::50,0],param_store_tf[::50,1],'>',zdir='x',c='red',linewidth=7,markersize=6)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('BAM Value')
# plt.colorbar(label="BAM Value")

fig.savefig('optimization-path-3D.png',bbox_inches = "tight", dpi=500)

# ax.contour(X.T, Y.T, bam_mesh, linestyles="solid", cmap=cmap_cur, linewidth=3, offset=bam_mesh.min())


















