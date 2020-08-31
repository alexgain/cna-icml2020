from datasets import *
from models import *
import copy
import measures
import sys
sys.path.insert(0, './utils/')
from util import *

# models = ['MLP','VGG18','ResNet18','ResNet101']
# datasets = ['mnist','fashion','svhn','cifar10','cifar100']
models = ['ResNet101','MLP','VGG18','ResNet18']
# datasets = ['mnist','fashion','svhn','cifar10','cifar100']#,'imagenet']
# datasets = ['random','mnist','fashion','svhn','cifar10','cifar100']#,'imagenet']
datasets = ['mnist','fashion','svhn','cifar10','cifar100','imagenet32']#,'imagenet']
# initialized = ['yes','no']
initialized = ['yes','no']
source_directory = './saved_models/'
# metrics = ['BAM','BAM+','L1','L2','L2-path','Spectral norm','2018 Bound']# '2019 Bound']
metrics = ['BAM','BAM+','L2','L2-path','Spectral norm','2018 Bound']# '2019 Bound']
# metrics = ['L1','L2']

def weight_reset(m):
    m.reset_parameters()
# def weight_init(m):#, mean, std):
#     for m in self._modules:
#         normal_init(self._modules[m], mean, std)
def normal_init(m):
    try:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.zero_()
    except:
        pass

csv_array = []
csv_array_orig = []

bam_vals_tot = []
bam_vals2_tot = []
l1_vals_tot = []
l2_vals_tot = []
l2_path_vals_tot = []
spectral_vals_tot = []
new_bound_vals_tot = []
test_vals_tot = []

bam_vals_orig_tot = []
bam_vals2_orig_tot = []
l1_vals_orig_tot = []
l2_vals_orig_tot = []
l2_path_vals_orig_tot = []
spectral_vals_orig_tot = []
new_bound_vals_orig_tot = []
test_vals_orig_tot = []

csv_array.append(['']+metrics)
csv_array_orig.append(['']+metrics)
for model in models:
    
    margin_list = []
    
    csv_row = [model]
    bam_vals = []
    bam_vals2 = []
    l1_vals = []
    l2_vals = []
    l2_path_vals = []
    spectral_vals = []
    new_bound_vals = []
    test_vals = []
    
    bam_vals_orig = []
    bam_vals2_orig = []
    l1_vals_orig = []
    l2_vals_orig = []
    l2_path_vals_orig = []
    spectral_vals_orig = []
    new_bound_vals_orig = []
    test_vals_orig = []

    for dataset in datasets:
        
        for initial in initialized:
            
            flatten, image_aug, input_shape, channels, input_padding, num_classes, epochs, img_dim = get_dataset_params(model, dataset)
            
            N_sub = 0
            if model != 'ResNet101':
                train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, image_aug = image_aug, BS = 128, BS2 = 128)
            else:
                train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, image_aug = image_aug, BS = 128, BS2 = 128)

                
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
    
            if gpu_boole:
                state_dict = torch.load(source_directory+model+'_'+dataset+'.state')
            else:
                state_dict = torch.load(source_directory+model+'_'+dataset+'.state',map_location='cpu')            
    
            if list(state_dict.keys())[0][:6] == 'module':
                for key in list(state_dict.keys()):
                    state_dict[key[7:]] = state_dict.pop(key)
    
            else:
                if model != 'ResNet18' or model != 'ResNet101':
                    continue
                elif model == 'ResNet18':
                    my_net = resnet18(pretrained=True)
                elif model == 'ResNet101':
                    my_net = resnet101(pretrained=True)
                    
                
                
            if gpu_boole:
                my_net = my_net.cuda()
            
            init_model = copy.deepcopy(my_net)
            # init_model.apply(normal_init)            
            # my_net.apply(normal_init)
            
            
            if initial == "no":
                my_net.load_state_dict(state_dict)

            # my_net.load_state_dict(state_dict)
                
            my_net = nn.DataParallel(my_net)
            init_model = nn.DataParallel(init_model)
            my_net.beta = my_net.module.beta
    
            #BAM calc functions:
            loss_metric = nn.CrossEntropyLoss()
            def bam_calc(verbose=1, flatten=False, input_shape = 28*28):
                
                slopes = []
            
                for images, labels in train_loader_bam:
                    if flatten:
                        images = Variable(images.view(-1, input_shape), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
            
                    # images = images.no_grad()
                    # try:
                    slopes = slopes + list(my_net.beta(images))
                    # slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
                    # except:
                    #     slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
                        
            
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
                    
                    # try:
                    slopes = slopes + list(my_net.beta(images))
                    # slopes = slopes + list(my_net.beta(images).cpu().data.numpy())
                    # except:
                    #     slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
            
                slopes = np.array(slopes)
                
                bam = np.corrcoef(slopes,entropy_test)[0,1]
                
                if verbose:
                    print('BAM Test:',bam)
            
                return bam
            
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
            
                return 100.0 * np.float(correct) / np.float(total), loss_sum/np.float(total)
                
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
            
                return 100.0 * np.float(correct) / np.float(total), loss_sum/np.float(total)
            
            def get_l1_norm(model):
                norm_val = 0
                for param in model.parameters():
                    norm_val += torch.sum(torch.abs(param))    
                return norm_val.cpu().data.numpy().item()
    
            def get_l2_norm(model):
                norm_val = 0
                for param in model.parameters():
                    norm_val += torch.norm(torch.norm(param))    
                return norm_val.cpu().data.numpy().item()
            
            def margin_calc(verbose = 0, flatten=False, input_shape = 28*28):
                margin = []
                for images, labels in test_loader:
                    if gpu_boole:
                        images, labels = images.cuda(), labels.cuda()
                    if flatten:
                        images = images.view(-1, input_shape)
                    images = Variable(images)
                    # print(images.shape)
                    # print(labels.shape)
                    output_m = my_net(images)
                    for k in range(labels.size(0)):
                        output_m[k, labels[k]] = output_m[k,:].min()
                        margin = np.concatenate((margin, output_m[:, labels].diag().cpu().data.numpy() - output_m[:, output_m.max(1)[1]].diag().cpu().data.numpy()), 0)
                val_margin = np.percentile( margin, 5 )
                if verbose:
                    print('Margin is',val_margin)
                
                return val_margin
            
            tr_margin = margin_calc(flatten=flatten, input_shape=input_shape)
            
            if 'L2-path' in metrics or 'Spectral norm' in metrics:
                measure_dict, bounds_dict = measures.calculate(my_net, init_model, torch.device("cuda" if gpu_boole else "cpu"), train_loader, tr_margin, channels, num_classes, img_dim)#, model_name=model)
            
            # print(copy.copy(measure_dict['Frobenius1'].cpu().data.numpy().item()))
            # print(copy.copy(measure_dict['Our bound'].cpu().data.numpy().item()))
            # print(measure_dict['Frobenious norm'])
            # print(type(measure_dict['Frobenious norm']))
            
            trials = 1
            bam_val = bam_calc_test(flatten=flatten, input_shape=input_shape)
            if np.isnan(bam_val):
                bam_val = 0
            bam_val_list = [bam_val]
            if initial == "yes":
                for K in range(trials - 1):
                    bam_val = copy.copy(bam_calc_test(flatten=flatten, input_shape=input_shape))
                    if np.isnan(bam_val):
                        bam_val = 0
                    bam_val_list.append(copy.copy(bam_val))
                    my_net.apply(weight_reset)
                    # my_net.apply(normal_init)
            
            bam_val = np.mean(bam_val_list)
            if 'BAM' in metrics:
                bam_vals.append(copy.copy(bam_val))
                bam_vals_tot.append(copy.copy(bam_val))
                if initial == 'no':
                    bam_vals_orig.append(copy.copy(bam_val))
                    bam_vals_orig_tot.append(copy.copy(bam_val))
            
            bam_val += 1
            bam_val /= 2
            print('Normalized BAM:',bam_val)
            # bam_val *= np.log(np.sqrt(tr_margin*(-1)))
            # bam_val *= 1 - np.exp(tr_margin*(1))
            # bam_val *= 1 - 1/(1-tr_margin)
            
            import math
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))
            
            # bam_val *= np.tanh(tr_margin*(-1))
            bam_val *= sigmoid(tr_margin*(-1) - 3)
            # bam_val /= 2
            
            # bam_val *= np.exp(tr_margin)
            print('Modified BAM:',bam_val)
            print('Margin:',tr_margin)
            margin_list.append(tr_margin)
            
            # if 'BAM' in metrics:
            #     bam_vals.append(bam_val)
            #     bam_vals_tot.append(bam_val)
            #     if initial == 'no':
            #         bam_vals_orig.append(bam_val)
            #         bam_vals_orig_tot.append(bam_val)
            if 'BAM+' in metrics:
                bam_vals2.append(bam_val)
                bam_vals2_tot.append(bam_val)
                if initial == 'no':
                    bam_vals2_orig.append(bam_val)
                    bam_vals2_orig_tot.append(bam_val)
                                        
            if 'L1' in metrics:
                l1_vals.append(get_l1_norm(my_net))
                l1_vals_tot.append(get_l1_norm(my_net))
                if initial == 'no':
                    l1_vals_orig.append(get_l1_norm(my_net))
                    l1_vals_orig_tot.append(get_l1_norm(my_net))
            # if 'L2' in metrics:
            #     l2_vals.append(get_l2_norm(my_net))
            if 'L2' in metrics:
                l2_vals.append(copy.copy(measure_dict['Frobenious norm']))
                l2_vals_tot.append(copy.copy(measure_dict['Frobenious norm']))
                if initial == 'no':
                    l2_vals_orig.append(copy.copy(measure_dict['Frobenious norm']))
                    l2_vals_orig_tot.append(copy.copy(measure_dict['Frobenious norm']))
            if 'L2-path' in metrics:
                l2_path_vals.append(copy.copy(measure_dict['L2_path norm']))
                l2_path_vals_tot.append(copy.copy(measure_dict['L2_path norm']))
                if initial == 'no':
                    l2_path_vals_orig.append(copy.copy(measure_dict['L2_path norm']))
                    l2_path_vals_orig_tot.append(copy.copy(measure_dict['L2_path norm']))
            if 'Spectral norm' in metrics:
                spectral_vals.append(copy.copy(measure_dict['Spectral norm']))
                spectral_vals_tot.append(copy.copy(measure_dict['Spectral norm']))
                if initial == 'no':
                    spectral_vals_orig.append(copy.copy(measure_dict['Spectral norm']))
                    spectral_vals_orig_tot.append(copy.copy(measure_dict['Spectral norm']))
            if '2018 Bound' in metrics:
                new_bound_vals.append(copy.copy(bounds_dict['Spec_Fro Bound (Neyshabur et al. 2018)']))
                new_bound_vals_tot.append(copy.copy(bounds_dict['Spec_Fro Bound (Neyshabur et al. 2018)']))
                if initial == 'no':
                    new_bound_vals_orig.append(copy.copy(bounds_dict['Spec_Fro Bound (Neyshabur et al. 2018)']))
                    new_bound_vals_orig_tot.append(copy.copy(bounds_dict['Spec_Fro Bound (Neyshabur et al. 2018)']))
            if '2019 Bound' in metrics:
                new_bound_vals.append(copy.copy(measure_dict['Our bound'].cpu().data.numpy().item()))
                new_bound_vals_tot.append(copy.copy(measure_dict['Our bound'].cpu().data.numpy().item()))
                if initial == 'no':
                    l1_vals_orig.append(bam_val)
                    l1_vals_orig_tot.append(bam_val)

            # if 'Spectral norm' in metrics:
            #     spectral_vals.append(copy.copy(measure_dict['Spectral norm']))
            test_acc_val = test_acc(flatten=flatten, input_shape=input_shape)[0]
            test_vals.append(test_acc_val)
            test_vals_tot.append(test_acc_val)
            if initial == "no":
                test_vals_orig.append(test_acc_val)
                test_vals_orig_tot.append(test_acc_val)
            
    np.save('./results/'+'bam_initial_'+model+'.npy',np.array(bam_vals))
    np.save('./results/'+'bam+_initial_'+model+'.npy',np.array(bam_vals2))
    np.save('./results/'+'bam_'+model+'.npy',np.array(bam_vals_orig))
    np.save('./results/'+'bam+_'+model+'.npy',np.array(bam_vals2_orig))
    np.save('./results/'+'test_accs_initial_'+model+'.npy',np.array(test_vals))
    np.save('./results/'+'test_accs_'+model+'.npy',np.array(test_vals_orig))
                
    print("Margin Corr:",np.corrcoef(margin_list,test_vals))
    csv_row = [model] 
    csv_row_orig = [model]
    if 'BAM' in metrics:
        csv_row.append(np.corrcoef(bam_vals,test_vals)[0,1]) #BAM
        csv_row_orig.append(np.corrcoef(bam_vals_orig,test_vals_orig)[0,1]) #BAM
    if 'BAM+' in metrics:
        csv_row.append(np.corrcoef(bam_vals2,test_vals)[0,1]) #BAM
        csv_row_orig.append(np.corrcoef(bam_vals2_orig,test_vals_orig)[0,1]) #BAM
    if 'L1' in metrics:
        csv_row.append(np.corrcoef(l1_vals,test_vals)[0,1]) #L1
        csv_row_orig.append(np.corrcoef(l1_vals_orig,test_vals_orig)[0,1]) #L1
    if 'L2' in metrics:
        csv_row.append(np.corrcoef(l2_vals,test_vals)[0,1]) #L2    
        csv_row_orig.append(np.corrcoef(l2_vals_orig,test_vals_orig)[0,1]) #L2    
    if 'L2-path' in metrics:
        csv_row.append(np.corrcoef(l2_path_vals,test_vals)[0,1]) #L2    
        csv_row_orig.append(np.corrcoef(l2_path_vals_orig,test_vals_orig)[0,1]) #L2    
    if 'Spectral norm' in metrics:
        csv_row.append(np.corrcoef(spectral_vals,test_vals)[0,1]) #L2    
        csv_row_orig.append(np.corrcoef(spectral_vals_orig,test_vals_orig)[0,1]) #L2    
    if '2018 Bound' in metrics:
        csv_row.append(np.corrcoef(new_bound_vals,test_vals)[0,1]) #L2    
        csv_row_orig.append(np.corrcoef(new_bound_vals_orig,test_vals_orig)[0,1]) #L2    
    # if '2019 Bound' in metrics:
    #     csv_row.append(np.corrcoef(new_bound_vals,test_vals)[0,1]) #L2    
    # csv_row.append(np.corrcoef(bam_vals,test_vals)[0,1]) #BAM    
    csv_array.append(csv_row)
    csv_array_orig.append(csv_row_orig)
    print(csv_row)
    print(csv_row_orig)
    
csv_row = ['Total']
csv_row_orig = ['Total']
if 'BAM' in metrics:
    csv_row.append(np.corrcoef(bam_vals_tot,test_vals_tot)[0,1]) #BAM
    csv_row_orig.append(np.corrcoef(bam_vals_orig_tot,test_vals_orig_tot)[0,1]) #BAM
if 'BAM+' in metrics:
    csv_row.append(np.corrcoef(bam_vals2_tot,test_vals_tot)[0,1]) #BAM
    csv_row_orig.append(np.corrcoef(bam_vals2_orig_tot,test_vals_orig_tot)[0,1]) #BAM
if 'L1' in metrics:
    csv_row.append(np.corrcoef(l1_vals_tot,test_vals_tot)[0,1]) #L1
    csv_row_orig.append(np.corrcoef(l1_vals_orig_tot,test_vals_orig_tot)[0,1]) #L1
if 'L2' in metrics:
    csv_row.append(np.corrcoef(l2_vals_tot,test_vals_tot)[0,1]) #L2    
    csv_row_orig.append(np.corrcoef(l2_vals_orig_tot,test_vals_orig_tot)[0,1]) #L2    
if 'L2-path' in metrics:
    csv_row.append(np.corrcoef(l2_path_vals_tot,test_vals_tot)[0,1]) #L2    
    csv_row_orig.append(np.corrcoef(l2_path_vals_orig_tot,test_vals_orig_tot)[0,1]) #L2    
if 'Spectral norm' in metrics:
    csv_row.append(np.corrcoef(spectral_vals_tot,test_vals_tot)[0,1]) #L2    
    csv_row_orig.append(np.corrcoef(spectral_vals_orig_tot,test_vals_orig_tot)[0,1]) #L2    
if '2018 Bound' in metrics:
    csv_row.append(np.corrcoef(new_bound_vals_tot,test_vals_tot)[0,1]) #L2    
    csv_row_orig.append(np.corrcoef(new_bound_vals_orig_tot,test_vals_orig_tot)[0,1]) #L2    

csv_array.append(csv_row)
csv_array = np.array(csv_array).T
print(csv_array)
np.save('./results/csv_array.npy',csv_array)    
    
csv_array_orig.append(csv_row_orig)
csv_array_orig = np.array(csv_array_orig).T
print(csv_array_orig)
np.save('./results/csv_array_orig.npy',csv_array_orig)    

np.save('./results/'+'bam_initial_tot.npy',np.array(bam_vals))
np.save('./results/'+'bam+_initial_tot.npy',np.array(bam_vals2))
np.save('./results/'+'bam_tot.npy',np.array(bam_vals_orig))
np.save('./results/'+'bam+_tot.npy',np.array(bam_vals2_orig))
np.save('./results/'+'test_accs_initial_tot.npy',np.array(test_vals_tot))
np.save('./results/'+'test_accs_tot.npy',np.array(test_vals_orig_tot))

# plt.plot(np.load('test_accs_initial_tot.npy'),np.load('bam_initial_tot.npy'),'o')
# plt.plot(np.load('test_accs_initial_tot.npy'),np.load('bam+_initial_tot.npy'),'o')
# plt.plot(np.load('test_accs_tot.npy'),np.load('bam_tot.npy'),'o')
# plt.plot(np.load('test_accs_tot.npy'),np.load('bam+_tot.npy'),'o')
