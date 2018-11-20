import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import Tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import roc_curve

from model_trainer import ModelTrainer
from history import History
from data_import import *
from nn_modules import *


#######################################################################################
def debug_layers_dims(model, input_shape, headers=True):
    """Debug problems with layers dimensions.
    Args:
    model: the model to debug
    input_shape: pass train_input.shape"""
    if input_shape:
        if headers:
            print("******************** Debugging layers sizes: ********************")
        print("input_shape= (_,{})".format(",".join(map(str,input_shape[1:]))))
        out = torch.rand(2, *input_shape[1:])    
        for m in list(model.modules())[1:]:
            if type(m)==nn.Sequential:
                print("**************************************")
                out = debug_layers_dims(m, out.shape, False)
            else:
                print("------------------")
                print(m)
                tmp = m(out)
                outs = "(_,{})".format(",".join(map(str, out.shape[1:]))) if type(out)==Tensor else "???"
                tmps = "(_,{})".format(",".join(map(str, tmp.shape[1:]))) if type(tmp)==Tensor else "???"
                print(outs, "-->", tmps)
                out=tmp
    if headers:
        print("*****************************************************************")

    return out

#######################################################################################

def computeClassesWeights(labels):
    """Compute the weights of each label class when they aren't equally distributed.
    Can be for example used as the weigths of the nn.CrossEntropyLoss loss function.
    Args: 
    -labels : Tensor containing the target classes
    Returns:
    -Tensor with the weigths of the classes, sorted by classes number
    """
    target_classes = np.array(labels.tolist())
    unique, counts = np.unique(target_classes, return_counts=True)

    return Tensor(counts/len(target_classes))

#######################################################################################

def spyOn(layers, names, verbose=False):
    """Add hooks to the specified layers to capture their outputs.
    The outputs are stored in a dict(), with the names given as keys.
    Args: 
    -layers  : Array containing the model layers to spy on.
    -names   : Array containing the names of the layer, to be used as keys in the returned dicts
    -verbose : Boolean, if set to True each hook will display a message when activated.
    Returns:
    Two dict()
    -the first where the outputs captured by the hooks will be stored.
    -The second one where the hooks handles will be stored.
    """
    if (len(layers) != len(names)):
        print("The names array must have the same lengths as the layers array")
        
    output_dict = dict()
    handle_dict = dict()
    
    for i, layer in enumerate(layers):
        
        def make_f(n):
            def f(m, input_, output_): 
                output_dict[n] = output_
                if (verbose):
                    print("captured output at layer : "+str(m)) 
            return f
        
        handle_dict[names[i]] = layer.register_forward_hook(make_f(names[i]))
        
    return output_dict, handle_dict

def remove_spying(handles_dict):
    """given a dict() of handles, remove all of them.
    Args: 
    -handles_dict : dict of handles to remove."""
    for h in list(handles_dict.values()):
        h.remove()
        
#######################################################################################

def save_model_state(model, filename):
    """
    save the specified model state in memory, with the given filename.
    
    Args: 
    -model : the pytorch model which state to save
    -filename : name of the file in which to store the model state
    """
    model_file = open(filename, mode='wb')
    pickle.dump(model.state_dict(), model_file)
    model_file.close()
    
def load_model_state(model, filename):
    """
    load and replace the given model state from a previously saved
    state in memory, under the given filename.
    
    Args: 
    -model : the pytorch model which state to replace
    -filename : name of the file in which the previous model state is saved
    """
    model_file = open(filename, mode='rb')
    model.load_state_dict(pickle.load(model_file))
    model_file.close()
    
#######################################################################################    
    
def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

#######################################################################################  

def diagnostic_plots(model, train_dataset, test_dataset):
    positive_train = model(train_dataset[0][train_dataset[1] == 1]).detach().cpu().numpy()
    negative_train = model(train_dataset[0][train_dataset[1] == 0]).detach().cpu().numpy()

    positive_test = model(test_dataset[0][test_dataset[1] == 1]).detach().cpu().numpy()
    negative_test = model(test_dataset[0][test_dataset[1] == 0]).detach().cpu().numpy()

    print("false negative percentage :", 100 - 100*positive_test.argmax(1).sum()/positive_test.shape[0])
    print("false positive percentage :", 100*negative_test.argmax(1).sum()/negative_test.shape[0])
    
    f, axs = plt.subplots(2,2,figsize=(15,15))
    sns.set_style('whitegrid')
    g = sns.kdeplot(positive_train[:, 1]-positive_train[:, 0], bw=0.1, label='target=1', ax=axs[0,0])
    sns.kdeplot(negative_train[:, 1]-negative_train[:, 0], bw=0.1, ax=g, label='target=0')
    axs[0,0].set_title('Distribution train set')

    sns.set_style('whitegrid')
    g = sns.kdeplot(positive_test[:, 1]-positive_test[:, 0], bw=0.1, label='target=1', ax=axs[0,1])
    sns.kdeplot(negative_test[:, 1]-negative_test[:, 0], bw=0.1, ax=g, label='target=0')
    axs[0,1].set_title('Distribution test set')
    
    fpr_train, tpr_train, threshold_train = roc_curve(train_dataset[1], model(train_dataset[0]).detach().cpu().numpy()[:,1])
    axs[1,0].plot(fpr_train, tpr_train)
    axs[1,0].set_title('ROC of train set')
    axs[1,0].set_xlabel('False Positive Rate')
    axs[1,0].set_ylabel('True Positive Rate')
    
    fpr_test, tpr_test, threshold_train = roc_curve(test_dataset[1], model(test_dataset[0]).detach().cpu().numpy()[:,1])
    axs[1,1].plot(fpr_test, tpr_test)
    axs[1,1].set_title('ROC of test set')
    axs[1,1].set_xlabel('False Positive Rate')
    axs[1,1].set_ylabel('True Positive Rate')
    plt.show()