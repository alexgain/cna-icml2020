from datasets import *
from models import *
import copy
import measures
import sys
sys.path.insert(0, './utils/')
from util import *
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff, pdist, cdist

# models = ['MLP','VGG18','ResNet18','ResNet101']
models = ['MLP','VGG18','ResNet18']
datasets = ['mnist','fashion','svhn','cifar10','cifar100','random']
source_directory = './saved_models/'
metrics = ['BAM','BAM+','L2','L2-path','Spectral norm','2018 Bound']# '2019 Bound']

corrupt = 'shuffle'
if corrupt == 'shuffle':
    corrupt_vals = [0.1,0.2,0.3,0.4,0.5]
if corrupt == 'noise':
    corrupt_vals = [0.1,0.2,0.4,0.8,1.6]
    

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

import math
def get_entropy(x):
    hist = np.histogram(x, bins=1000, range = (0,1),density=True)
##    hist = np.histogram(x, bins=5, range = (x_train.min(),x_train.max()),density=True)
    data = hist[0]
    ent=0
    for i in hist[0]:
        if i!=0:
            ent -= i * np.log2(abs(i))
    return ent

csv_array = []

bam_vals_tot = []
bam_vals2_tot = []
l1_vals_tot = []
l2_vals_tot = []
l2_path_vals_tot = []
spectral_vals_tot = []
new_bound_vals_tot = []
test_vals_tot = []

csv_array.append(['']+metrics)
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
    
    for dataset in datasets:
                
        flatten, image_aug, input_shape, channels, input_padding, num_classes, epochs, img_dim = get_dataset_params(model, dataset)
        
        for perc in corrupt_vals:
            if dataset == 'random':
                break
        
            N_sub = 0
            if model != 'ResNet101':
                train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, image_aug = image_aug, BS = 128, BS2 = 128, shuffle = False)
            else:
                train_loader, test_loader, train_loader_bam, test_loader_bam, entropy, entropy_test = get_loaders(dataset, N_sub = N_sub, image_aug = image_aug, BS = 64, BS2 = 64, shuffle = False)                

            if corrupt == 'noise':               
                noise_data = torch.Tensor(np.load('./data_noise/'+model+'_'+dataset+'_noise_'+str(int(perc*100))+'.npy'))

                
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
    
            state_dict = torch.load(source_directory+model+'_'+dataset+'_'+corrupt+str(int(perc*100))+'.state')                    
    
            if list(state_dict.keys())[0][:6] == 'module':
                for key in list(state_dict.keys()):
                    state_dict[key[7:]] = state_dict.pop(key)                    
                
            if gpu_boole:
                my_net = my_net.cuda()
            
            init_model = copy.deepcopy(my_net)
            
            my_net.load_state_dict(state_dict)
                
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
                    print('Accuracy of the network on the test images: %f %%' % (100.0 * np.float(correct) / np.float(total)))
            
                return 100.0 * np.float(correct) / np.float(total), loss_sum/np.float(total)

            def get_slopes_train(verbose=1, flatten=False, input_shape = 28*28):
                
                slopes = []
            
                for i, (images,labels) in enumerate(train_loader):
                    
                    if corrupt == "noise":
                        cur_ind = i*train_loader.batch_size
                        images += noise_data[cur_ind:cur_ind+images.shape[0]]

                    if flatten:
                        images = Variable(images.view(images.shape[0], -1), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
            
                    # images = images.no_grad()
                    # try:
                    slopes = slopes + list(my_net.beta(images))
                    # except:
                    #     slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
                        
                slopes = np.array(slopes)
            
                return slopes
            
            def get_slopes_test(verbose=1, flatten=False, input_shape = 28*28):
                
                slopes = []
            
                for i, (images,labels) in enumerate(test_loader):
                    
                    if corrupt == "noise":
                        cur_ind = i*test_loader.batch_size
                        images += noise_data[cur_ind:cur_ind+images.shape[0]]

                    if flatten:
                        images = Variable(images.view(images.shape[0], -1), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
            
                    # images = images.no_grad()
                    # try:
                    slopes = slopes + list(my_net.beta(images))
                    # except:
                    #     slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
                        
            
                slopes = np.array(slopes)
            
                return slopes
            
            def get_ent_train(verbose=1, flatten=False, input_shape = 28*28):
                
                ent = []
            
                for i, (images,labels) in enumerate(train_loader):
                    
                    if corrupt == "noise":
                        cur_ind = i*train_loader.batch_size
                        images += noise_data[cur_ind:cur_ind+images.shape[0]]

                    if flatten:
                        images = Variable(images.view(images.shape[0], -1), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
            
                    # images = images.no_grad()
                    # try:
                    ent = ent + [get_entropy(images.cpu().numpy()[i]) for i in range(images.shape[0])]
                    # except:
                    #     slopes = slopes + list(my_net.module.beta(images).cpu().data.numpy())
                        
                ent = np.array(ent)
            
                return ent
            
            def get_ent_test(verbose=1, flatten=False, input_shape = 28*28):
                
                ent = []
            
                for i, (images,labels) in enumerate(test_loader):
                    
                    if corrupt == "noise":
                        cur_ind = i*test_loader.batch_size
                        images += noise_data[cur_ind:cur_ind+images.shape[0]]

                    if flatten:
                        images = Variable(images.view(images.shape[0], -1), volatile = True)
                    images = Variable(images, volatile = True)
                    if gpu_boole:
                        images = images.cuda()
            
                    # images = images.no_grad()
                    # try:
                    ent = ent + [get_entropy(images.cpu().numpy()[i]) for i in range(images.shape[0])]
                    # except:
                    #     ent = ent + list(my_net.module.beta(images).cpu().data.numpy())
                        
            
                ent = np.array(ent)
            
                return ent
            
            
            
            # slopes = get_slopes_train(verbose=False, flatten=True)
            
            def get_perc_idx(percentile, test = False):
                if test:
                    entropy_cur = entropy_test
                else:
                    entropy_cur = entropy
                    
                idx = np.where(entropy_cur>np.percentile(entropy_cur,percentile))[0]
            
                return idx
            
            def top_bottom_ind(percentile, test = False):
                if test:
                    entropy_cur = entropy_test
                else:
                    entropy_cur = entropy
            
                idx_top = np.where(entropy_cur>np.percentile(entropy_cur,percentile))[0]
                idx_bottom = np.where(entropy_cur<np.percentile(entropy_cur, 100 - percentile))[0]
            
                return idx_top, idx_bottom
            
            def top_bottom_slope_entropy(percentile, test = False, flatten = True):
                idx_top, idx_bottom = top_bottom_ind(percentile, test)
                
                if test:
                    slopes = get_slopes_test(0, flatten = flatten)
                    entropy_cur = entropy_test
                else:
                    slopes = get_slopes_train(0, flatten = flatten)
                    entropy_cur = entropy
                
                slopes = np.concatenate([slopes[idx_bottom],slopes[idx_top]])
                entropy_cur = np.concatenate([entropy_cur[idx_bottom],entropy_cur[idx_top]])
            
                return slopes, entropy_cur
            
            def get_bam_percentile(percentile ,test = False, flatten = True):
                slopes_tb, entropy_tb = top_bottom_slope_entropy(percentile, test, flatten = flatten)   
                return np.corrcoef(slopes_tb,entropy_tb)[0,1]

            def running_mean(x, N):
                cumsum = np.cumsum(np.insert(x, 0, 0)) 
                return (cumsum[N:] - cumsum[:-N]) / float(N)
            
            def curve_similarity(slope_train,ent_train,slope_test,ent_test, points=15):
            
                x, y, x2, y2 = slope_train,ent_train,slope_test,ent_test
                
                f = interp1d(x, y)
                f2 = interp1d(x2,y2)
                    
                xnew = np.linspace ( min(x), max(x), num = points) 
                xnew2 = np.linspace ( min(x2), max(x2), num = points) 
                
                ynew = f(xnew) 
                ynew2 = f2(xnew2) 
                
                return np.corrcoef(ynew, ynew2)[0,1], ss.spearmanr(ynew, ynew2), np.correlate(ynew, ynew2, mode='valid')[0], ssd.correlation(ynew, ynew2)
            
            def curve_est(slope_train,ent_train,slope_test,ent_test, points=15):
            
                x, y, x2, y2 = slope_train,ent_train,slope_test,ent_test
                
                f = interp1d(x, y)
                f2 = interp1d(x2,y2)
                    
                xnew = np.linspace ( min(x), max(x), num = points) 
                xnew2 = np.linspace ( min(x2), max(x2), num = points) 
                
                ynew = f(xnew) 
                ynew2 = f2(xnew2) 
                
                return xnew, ynew, xnew2, ynew2


            def get_area_measure(flatten = True):
            
                slopes_train = get_slopes_train(0, flatten = flatten)
                slopes_test = get_slopes_test(0, flatten = flatten)
                # if corrupt == "noise":
                #     entropy = get_ent_train(0, flatten = flatten)
                #     entropy_test = get_ent_test(0, flatten = flatten)
                
                ent_ind_train = np.argsort(entropy)
                ent_ind_test = np.argsort(entropy_test)
            
                K = 25
                
                slopes_train_smooth, entropy_train_smooth = running_mean(slopes_train[ent_ind_train],K), running_mean(entropy[ent_ind_train],K)
                slopes_test_smooth, entropy_test_smooth = running_mean(slopes_test[ent_ind_test],K), running_mean(entropy_test[ent_ind_test],K)
                # slopes_train_smooth, entropy_train_smooth, slopes_test_smooth, entropy_test_smooth = min_max_scale(slopes_train_smooth), min_max_scale(entropy_train_smooth), min_max_scale(slopes_test_smooth), min_max_scale(entropy_test_smooth) 
                
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
            
                xnew, ynew, xnew2, ynew2 = curve_est(slopes_train_smooth, entropy_train_smooth, slopes_test_smooth, entropy_test_smooth, points = 100)

                ## original area calc:
                array1 = np.array([xnew, ynew]).T
                array2 = np.array([xnew2, ynew2]).T
                array2 = np.flip(array2,axis=0)
                polygon_points = np.concatenate((array1,array2,array1[0].reshape([1,2])),axis=0)
                
                polygon = Polygon(polygon_points)
                area = polygon.area
                
                # area = np.abs(ynew - ynew2).var()
                
                bam_reg_train = np.corrcoef(slopes_train,entropy)[0,1]
                bam_reg_test = np.corrcoef(slopes_test,entropy_test)[0,1]
                bam_smooth_train = np.corrcoef(slopes_train_smooth,entropy_train_smooth)[0,1]
                bam_smooth_test = np.corrcoef(slopes_test_smooth,entropy_test_smooth)[0,1]                                

                # final = (((bam_smooth_train + bam_smooth_test)/2)+1)/2  #Smoothed BAM only
                final = area    #Area only
                # final = (((bam_smooth_train + bam_smooth_test)/2)+1)/2 + 4*area #BAM minus Area

                # final = directed_hausdorff(array1,array2)[0] #DH
                
                print('BAM Test:', bam_reg_test)                
                print('BAM Train Smoothed:',bam_smooth_train)
                print('BAM Test Smoothed:',bam_smooth_test)
                print('BAM Smooth Diff:', bam_smooth_train - bam_smooth_test)
                print('Area:',area)
                print('Final Metric', final)
                
                return final, bam_smooth_test

            
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
                for i, (images,labels) in enumerate(test_loader):
                    
                    if corrupt == "noise":
                        cur_ind = i*test_loader.batch_size
                        images += noise_data[cur_ind:cur_ind+images.shape[0]]

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
            
            # train_acc_cur, train_loss_cur = train_acc(flatten=flatten, input_shape=input_shape)
            # test_acc_cur, test_loss_cur = test_acc(flatten=flatten, input_shape=input_shape)
            
            train_loss_cur = np.load('./results/'+model+'_'+dataset+'_'+corrupt+str(int(perc*100))+'_train_loss'+'.npy')
            test_loss_cur = np.load('./results/'+model+'_'+dataset+'_'+corrupt+str(int(perc*100))+'_test_loss'+'.npy')
            train_acc = np.load('./results/'+model+'_'+dataset+'_'+corrupt+str(int(perc*100))+'_train_acc'+'.npy')
            test_acc = np.load('./results/'+model+'_'+dataset+'_'+corrupt+str(int(perc*100))+'_test_acc'+'.npy')
            
            # try:
            bam_val, bam_smooth_test = get_area_measure(flatten)
            # except:
            #     continue
                        
            # bam_val = np.abs(train_acc_cur/100 - bam_val)
            # bam_val_margin = copy.copy(bam_val)
            bam_val_margin = copy.copy(bam_val)
            # bam_val = np.abs(train_loss_cur - bam_val)            
            
            print('BAM measure:',bam_val)
            
            if 'BAM' in metrics:
                bam_vals.append(copy.copy(bam_val))
                bam_vals_tot.append(copy.copy(bam_val))
            
            
            import math
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))
            
            # bam_val *= sigmoid(tr_margin*(-1) - 3)
            # bam_val_margin *= tr_margin**2
            bam_val_margin *= tr_margin
            
            print('Modified BAM:',bam_val_margin)
            print('Margin:',tr_margin)
            margin_list.append(tr_margin)
                        
            if 'BAM+' in metrics:
                bam_vals2.append(bam_val_margin)
                bam_vals2_tot.append(bam_val_margin)
                                        
            if 'L1' in metrics:
                l1_vals.append(get_l1_norm(my_net))
                l1_vals_tot.append(get_l1_norm(my_net))
            if 'L2' in metrics:
                l2_vals.append(copy.copy(measure_dict['Frobenious norm']))
                l2_vals_tot.append(copy.copy(measure_dict['Frobenious norm']))
            if 'L2-path' in metrics:
                l2_path_vals.append(copy.copy(measure_dict['L2_path norm']))
                l2_path_vals_tot.append(copy.copy(measure_dict['L2_path norm']))
            if 'Spectral norm' in metrics:
                spectral_vals.append(copy.copy(measure_dict['Spectral norm']))
                spectral_vals_tot.append(copy.copy(measure_dict['Spectral norm']))
            if '2018 Bound' in metrics:
                new_bound_vals.append(copy.copy(bounds_dict['Spec_Fro Bound (Neyshabur et al. 2018)']))
                new_bound_vals_tot.append(copy.copy(bounds_dict['Spec_Fro Bound (Neyshabur et al. 2018)']))
            if '2019 Bound' in metrics:
                new_bound_vals.append(copy.copy(measure_dict['Our bound'].cpu().data.numpy().item()))
                new_bound_vals_tot.append(copy.copy(measure_dict['Our bound'].cpu().data.numpy().item()))
    
            # test_acc_val = test_acc(flatten=flatten, input_shape=input_shape)[0]
            
            # test_acc_val = np.abs(train_loss_cur - test_loss_cur)
            test_acc_val = train_loss_cur - test_loss_cur
            # test_acc_val = np.abs(train_acc_cur - test_acc_cur)
            # test_acc_val = test_acc_cur
            
            # test_acc_val = train_acc_cur - test_acc(flatten=flatten, input_shape=input_shape)[0]
            # test_acc_val *= (-1)
            print('gen gap:',test_acc_val)
            test_vals.append(test_acc_val)
            test_vals_tot.append(test_acc_val)
            
    np.save('./results/'+'bam_'+model+'.npy',np.array(bam_vals))
    np.save('./results/'+'bam+_'+model+'.npy',np.array(bam_vals2))
    np.save('./results/'+'test_accs_'+model+'.npy',np.array(test_vals))
                
    print("Margin Corr:",np.corrcoef(margin_list,test_vals))
    csv_row = [model] 
    if 'BAM' in metrics:
        csv_row.append(np.corrcoef(bam_vals,test_vals)[0,1]) #BAM
    if 'BAM+' in metrics:
        csv_row.append(np.corrcoef(bam_vals2,test_vals)[0,1]) #BAM
    if 'L1' in metrics:
        csv_row.append(np.corrcoef(l1_vals,test_vals)[0,1]) #L1
    if 'L2' in metrics:
        csv_row.append(np.corrcoef(l2_vals,test_vals)[0,1]) #L2    
    if 'L2-path' in metrics:
        csv_row.append(np.corrcoef(l2_path_vals,test_vals)[0,1]) #L2    
    if 'Spectral norm' in metrics:
        csv_row.append(np.corrcoef(spectral_vals,test_vals)[0,1]) #L2    
    if '2018 Bound' in metrics:
        csv_row.append(np.corrcoef(new_bound_vals,test_vals)[0,1]) #L2    
    csv_array.append(csv_row)
    print(csv_row)
    
csv_row = ['Total']
if 'BAM' in metrics:
    csv_row.append(np.corrcoef(bam_vals_tot,test_vals_tot)[0,1]) #BAM
if 'BAM+' in metrics:
    csv_row.append(np.corrcoef(bam_vals2_tot,test_vals_tot)[0,1]) #BAM
if 'L1' in metrics:
    csv_row.append(np.corrcoef(l1_vals_tot,test_vals_tot)[0,1]) #L1
if 'L2' in metrics:
    csv_row.append(np.corrcoef(l2_vals_tot,test_vals_tot)[0,1]) #L2    
if 'L2-path' in metrics:
    csv_row.append(np.corrcoef(l2_path_vals_tot,test_vals_tot)[0,1]) #L2    
if 'Spectral norm' in metrics:
    csv_row.append(np.corrcoef(spectral_vals_tot,test_vals_tot)[0,1]) #L2    
if '2018 Bound' in metrics:
    csv_row.append(np.corrcoef(new_bound_vals_tot,test_vals_tot)[0,1]) #L2    

csv_array.append(csv_row)
csv_array = np.array(csv_array).T
print(csv_array)
np.save('./results/csv_array_corrupt_all.npy',csv_array)    
    
# np.save('./results/'+'bam_tot.npy',np.array(bam_vals))
# np.save('./results/'+'bam+_tot.npy',np.array(bam_vals2))
# np.save('./results/'+'test_accs_tot.npy',np.array(test_vals_tot))

# np.save('./results/'+'bam_smoothed_total.npy',np.array(bam_smoothed_total))
# np.save('./results/'+'test_smoothed_total.npy',np.array(test_smoothed_total))
# np.save('./results/'+'test_loss_total.npy',np.array(test_loss_total))
