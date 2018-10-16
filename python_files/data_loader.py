
import torch
from torchvision import datasets

import argparse
import os

from utils import spyOn, remove_spying

######################################################################

parser = argparse.ArgumentParser(description='DLC prologue file for practical sessions.')

parser.add_argument('--full',
                    action='store_true', default=False,
                    help = 'Use the full set, can take ages (default False)')

parser.add_argument('--tiny',
                    action='store_true', default=False,
                    help = 'Use a very small set for quick checks (default False)')

parser.add_argument('--force_cpu',
                    action='store_true', default=False,
                    help = 'Keep tensors on the CPU, even if cuda is available (default False)')

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed (default 0, < 0 is no seeding)')

parser.add_argument('--cifar',
                    action='store_true', default=False,
                    help = 'Use the CIFAR data-set and not MNIST (default False)')

parser.add_argument('--data_dir',
                    type = str, default = None,
                    help = 'Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

# Timur's fix
parser.add_argument('-f', '--file', help='quick hack for jupyter')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.force_cpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

######################################################################
# The data

def convert_to_one_hot_labels(input, target):
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True, full=False):
    args.full = full

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = '../data'

    if args.cifar or (cifar is not None and cifar):
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)

        train_input = torch.from_numpy(cifar_train_set.train_data)
        # Dirty hack to handle the change between torchvision 1.0.6 and 1.0.8
        if train_input.size(3) == 3:
            train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        else:
            train_input = train_input.float()
        train_target = torch.LongTensor(cifar_train_set.train_labels)

        test_input = torch.from_numpy(cifar_test_set.test_data).float()
        # Dirty hack to handle the change between torchvision 1.0.6 and 1.0.8
        if test_input.size(3) == 3:
            test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        else:
            test_input = test_input.float()
        test_target = torch.LongTensor(cifar_test_set.test_labels)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

        train_input = mnist_train_set.train_data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.train_labels
        test_input = mnist_test_set.test_data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.test_labels

    if flatten:
        train_input = train_input.clone().view(train_input.size(0), -1)
        test_input = test_input.clone().view(test_input.size(0), -1)

    if args.full:
        if args.tiny:
            raise ValueError('Cannot have both --full and --tiny')
    else:
        if args.tiny:
            print('** Reduce the data-set to the tiny setup')
            train_input = train_input.narrow(0, 0, 500)
            train_target = train_target.narrow(0, 0, 500)
            test_input = test_input.narrow(0, 0, 100)
            test_target = test_target.narrow(0, 0, 100)
        else:
            print('** Reduce the data-set (use --full for the full thing)')
            train_input = train_input.narrow(0, 0, 1000)
            train_target = train_target.narrow(0, 0, 1000)
            test_input = test_input.narrow(0, 0, 1000)
            test_target = test_target.narrow(0, 0, 1000)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    # Move to the GPU if we can

    if torch.cuda.is_available() and not args.force_cpu:
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target


######################################################################

def generate_newdataset(train_dataset, test_dataset, split=0.7):
    """
    Generate the new dataset containing a mix of the train_dataset and the test_dataset.
    A target will have value '1' if the original data is from the train set, '0' if it 
    is from the test set.
    Args:
    -train_dataset: train_dataset
    -test_dataset: test_dataset
    -split: percentage of the data to keep for the new trainset
    Returns:
    -g_train_input: new train_input
    -g_train_target: new train_target
    -g_test_input: new test_input
    -g_test_target: new test_target
    """
    train_input = train_dataset[0]
    train_target = train_dataset[1]
    test_input = test_dataset[0]
    test_target = test_dataset[1]
    
    N_train = int(split* len(train_input))
    N_test = int(split * len(test_input))
    
    g_train_input = torch.cat((train_input[:N_train], test_input[:N_test]), 0)
    g_train_target = torch.cat((torch.ones(N_train), torch.zeros(N_test)), 0)
    g_test_input = torch.cat((train_input[N_train:], test_input[N_test:]), 0)
    g_test_target = torch.cat((torch.ones(len(train_input) - N_train), torch.zeros(len(test_input) - N_test)), 0)
    
    if train_input.is_cuda:
        g_train_target = g_train_target.cuda()
        g_test_target = g_test_target.cuda()
    
    return g_train_input, g_train_target, g_test_input, g_test_target
    
    

def get_snapshots_f(model, layers, layer_names, data):
    """
    Get a snapshot of the given layers when the model is doing the forward pass on the data.
    Args:
    -model : Model 
    -layers : Array containing the model layers to spy on
    -layer_names : Array containing the names of the layer, to be used as keys in the returned dicts
    -data : Data to feed the model
    Returns:
    -outpus: Array of values spied from the layers
    """
    with torch.no_grad():
        model.eval()
        
        output_d, handle_d = spyOn(layers, layer_names)
        _ = model(data)
        
        outputs = None
        
        for name in layer_names:
            output = output_d[name].reshape(data.shape[0], -1)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), 0)
        
        remove_spying(handle_d)
                
        return outputs
    

def generate_dataset_g(model, train_dataset, test_dataset, layers, layer_names, split=0.7, full=True):
    """
    Generate the dataset for g with the values spied from the given layers as input and the
    labels taking value '1' if the original data was part of the train set, '0' otherwise.
    Args:
    -model : model to spy
    -train_dataset: original train dataset
    -test_dataset: original test dataset
    -layers: Array containing the model layers to spy on
    -layer_name: Array containing the names of the layer, to be used as keys in the returned
    -split: Percentage of data to keep for the new train dataset
    Returns:
    -new train dataset
    -new test dataset
    """
    new_train_input, new_train_target, new_test_input, new_test_target = generate_newdataset(train_dataset, test_dataset, split)
    
    g_train_input = get_snapshots_f(model, layers, layer_names, new_train_input)
    g_test_input = get_snapshots_f(model, layers, layer_names, new_test_input)
    
    if not full:
        idx_train = torch.randperm(g_train_input.shape[0])
        idx_test = torch.randperm(g_test_input.shape[0])
        if(g_train_input.is_cuda): 
            idx_train = idx_train.cuda()
            idx_test = idx_test.cuda()
        
        g_train_input = g_train_input[idx_train].narrow(0, 0, 1000)
        new_train_target = new_train_target[idx_train].narrow(0, 0, 1000)
        g_test_input = g_test_input[idx_test].narrow(0, 0, 1000)
        new_test_target = new_test_target[idx_test].narrow(0, 0, 1000)
        
    g_train_input = g_train_input.unsqueeze(1).type_as(train_dataset[0])
    new_train_target = new_train_target.type_as(train_dataset[1])
    
    g_test_input = g_test_input.unsqueeze(1).type_as(test_dataset[0])
    new_test_target = new_test_target.type_as(test_dataset[1])
    
    return (g_train_input, new_train_target), (g_test_input, new_test_target)


######################################################################