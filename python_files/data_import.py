import torch
from torch import Tensor
import dlc_bci as bci

def import_data(folder='./data_bci', use_full_data = False, normalize=True, cuda=True):
    """Utility function to import the data.
    :argument folder where to store the data
    :argument use_full_data boolean to use the 100Hz data (False) or the full 1KHz data (True)
    :argument normalize normalize the data with the train mean and std"""
    
    train_input, train_target = bci.load(root = folder, one_khz = use_full_data) 
    test_input , test_target = bci.load(root = folder, train = False, one_khz = use_full_data)

    if torch.cuda.is_available() and cuda:
        train_input , train_target = train_input.cuda(), train_target.cuda()
        test_input , test_target = test_input.cuda(), test_target.cuda()

    train_input, train_target = train_input, train_target
    test_input, test_target = test_input, test_target
    
    if (normalize):
        tr_mean = train_input.mean()
        tr_std = train_input.std()
        
        train_input = (train_input-tr_mean)/tr_std
        test_input = (test_input-tr_mean)/tr_std
        
    return train_input, train_target, test_input, test_target

def generate_noise(std, shape, div=10):
    """Generate a gaussian noise Tensor with the desired std.
    :argument std Tensor containing the std of each channel of the training data
    :argument shape shape of the desired noise Tensor
    :argument div fraction of the real std to add as noise"""
    
    x = Tensor(*shape).zero_()

    for i, t in enumerate(x):
        t.normal_(0, std[i]/div)
    
    if torch.cuda.is_available():
        x = x.cuda()
    
    return x

def extend_data(train_input, train_target, nb_data=10, div=10):
    """Extend the training inputs by adding the same vectors with some noise.
    :argument train_input training input data
    :argument train_target training target data
    :argument nb_data how many noise Tensor to add for each training input
    :argument div fraction of the real std to add as noise"""
    
    std_channels = train_input.data.std(2).mean(0)
    
    to_add_input = list(train_input.data)
    to_add_target = list(train_target)

    for i, t in enumerate(train_input.data):
        for j in range(nb_data):
            tmp = t + generate_noise(std_channels, train_input[0].shape, div)
            to_add_input.append(tmp)
            to_add_target.append(train_target[i])
            
    for i in range(len(to_add_input)):
        tmp = to_add_input[i]
        to_add_input[i] = tmp.view(1, *tmp.shape)
    
    new_input = Variable(torch.cat(to_add_input))
    new_target = torch.cat(to_add_target)
    
    if torch.cuda.is_available():
        new_input = new_input.cuda()
        new_target = new_target.cuda()
        
    return new_input, new_target