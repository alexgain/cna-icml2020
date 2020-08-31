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
        outputs = my_net.forward_rand_ablate(images,p)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

    if verbose:
        print('Ablation accuracy (Random) of the network on the 10000 test images: %f %%' % (100 * correct / total))
    
    return 100 * correct / total

def bap_abl_val_test(p=0.1, verbose=0, flatten=False, input_shape = 28*28):
    
    slopes = []

    for images, labels in test_loader_bap:
        if flatten:
            images = Variable(images.view(-1, input_shape))
        images = Variable(images)
        if gpu_boole:
            images = images.cuda()
        
        slopes = slopes + list(my_net.beta(images).cpu().data.numpy())

    slopes = np.array(slopes)
    
    bap = my_net.corr_ablate(images,entropy_test,float(p))
    bap_cpu = bap.cpu().data.numpy().item()
    
    if verbose:
        print('BAP test with ablation:', bap_cpu)

    return bap_cpu

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

if loss != None:
loss = loss
else:
loss = lambda x, y: (x - y)**2

class_errors = dict()
classes = np.unique(y_batch.cpu().data.numpy())
for c_num in classes:
c_num_errors = loss(y_batch[y_batch==c_num.item()], y_hat[y_batch==c_num.item()])
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
        
    











