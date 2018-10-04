import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import Tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def spyOn(models, names, verbose=False):
    if (len(models) != len(names)):
        print("The names array must have the same lengths as the models array")
        
    output_dict = dict()
    handle_dict = dict()
    
    for i, model in enumerate(models):
        
        def make_f(n):
            def f(m, input_, output_): 
                output_dict[n] = output_
                if (verbose):
                    print("captured output at layer : "+str(m)) 
            return f
        
        handle_dict[names[i]] = model.register_forward_hook(make_f(names[i]))
        
    return output_dict, handle_dict