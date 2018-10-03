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
def sample(features, targets, split=5):
    """Sample the signal: 1khz signal --> split * (1khz/split) splitted signals"""
    n, f, t = features.shape
    x = features.unfold(2, t//split, t//split)
    x = x.transpose(1,3).transpose(2,3).contiguous()
    x = x.view(-1, f, t//split)
    
    n = targets.shape
    y = targets.view(-1, 1).repeat(1, split).contiguous().view(-1)
    
    return x, y

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
def convert_to_one_hot_labels(input_):
    """Convert one column input to hot labels.
    :argument input_ the single column input to convert to one_hot_labels"""
    
    tmp = input_.new(input_.size(0), 2).fill_(0)
    tmp.scatter_(1, input_.view(-1, 1).long(), 1)
    return tmp

#######################################################################################
def plot_models(hist_data, avg_w_size=20, colors=['C0', 'C1']):
    """plot the resulting history of multiples model_trainer fit functions.
    :argument hist_data list with all History instance of the fitting
    :argument avg_w_size plot smoothing function window size
    :argument colors colors for the plotting of training and validation curves"""
    
    results = list()
    columns = ['train loss', 'val loss','train acc', 'val acc']

    nb_runs = len(hist_data)
    invalid_runs = 0

    #minimum validation acc should be above 
    #reasonnable threshold based on the data
    minimum_acc = 0.6

    for c in columns:
        tmp = pd.DataFrame()
        tmp["average"] = hist_data[0][c] * 0
        for i in range(nb_runs):
            if (hist_data[i]['val acc'].max() > minimum_acc):
                tmp["run "+str(i)] = hist_data[i][c]
                tmp["average"] += hist_data[i][c]
            else:
                invalid_runs += 1
        tmp["average"] /= (len(tmp.columns)-1)
        results.append(tmp)

    invalid_runs = invalid_runs//4
    print(invalid_runs, "run with no learning.")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

    fig.suptitle("results of "+str(nb_runs - invalid_runs)+" different runs of training")

    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('epochs')
    axes[0].set_yscale('log')
    axes[0].set_title('loss evolution')

    axes[1].set_ylabel('accuracy [% correct]')
    axes[1].set_xlabel('epochs')
    axes[1].set_title('accuracy evolution')

    #best run is where best val accuracy was achieved
    #best_run = results[3].tail(1).idxmax(1).values[0]

    for i in range(len(columns)):
        #results[i][[best_run]].ewm(span=avg_w_size).mean().plot(ax=axes[i>>1], color=colors[i%2], legend=False)
        #results[i].drop(best_run, 1, inplace=True)

        for k in results[i].columns:
            results[i][[k]].ewm(span=avg_w_size).mean().plot(ax=axes[i>>1], alpha=0.2, color=colors[i%2], legend=False)

        results[i][['average']].ewm(span=avg_w_size).mean().plot(ax=axes[i>>1], alpha=1, color=colors[i%2], legend=False)

    axes[0].grid(color='0.8', linewidth=0.5, ls='--')
    axes[1].grid(color='0.8', linewidth=0.5, ls='--')
    
#######################################################################################
def multiple_training(create_model_fun, train_dataset, test_dataset, criterion, optim_fun, lr, crit_fun, epochs=100, batch_size=None, nb_runs=1, verbose=0, avg_w_size=15, plot_figures=True):
    """wrapper function to run multiple model_trainer on the same model to see the variations.
    :argument create_model_fun function to create the model to fit
    :argument train_dataset (x_train, y_train)
    :argument test_dataset (x_validation, y_validation)
    :argument criterion criterion to use
    :argument optim_fun optimiser function to use
    :argument lr learning rate for the optimiser
    :argument epochs nb of epochs to train
    :argument batch_size the mini-batchs size
    :argument nb_runs number of time to run the model_trainer
    :argument verbose print the current loss and accuracies if current_epoch%verbose == 0
    :argument avg_w_size plot smoothing function window size"""

    hist_data = list()    
    best_models = list()
    
    if verbose == 0:
        print("progress: ", end='')

    for i in range(nb_runs):
        model = create_model_fun()
        optimiser = optim_fun(model.parameters(), lr)
        mt = ModelTrainer(model, criterion, optimiser, criterion_fun=crit_fun)

        hist_data.append(mt.fit(train_dataset, test_dataset, epochs, batch_size, verbose).get_hist())
        best_models.append(mt.get_best_model())
        
        if verbose == 0:
            print(i, " ", end='')

    if verbose == 0:
        print(' done !')
    
    if(plot_figures):
        if (len(hist_data) == 1):
            mt.plot_training("Learning curves")
        else:
            plot_models(hist_data, avg_w_size)

    return hist_data, best_models

#######################################################################################
def split_data(features, targets):
    """function to split every single 1kHz input into ten different 100Hz samples.
    :argument features features inputs
    :argument targets targets inputs"""
    
    to_add_input = list()
    to_add_target = list()

    for i, t in enumerate(features.data):
        for j in range(10):
            tmp = t[:, j::10]
            to_add_input.append(tmp)
            to_add_target.append(targets[i])

    for i in range(len(to_add_input)):
        tmp = to_add_input[i].contiguous()
        to_add_input[i] = tmp.view(1, *tmp.shape)

    new_input = Variable(torch.cat(to_add_input))
    new_target = torch.cat(to_add_target)

    if torch.cuda.is_available():
        new_input = new_input.cuda()
        new_target = new_target.cuda()
        
    return new_input, new_target

#######################################################################################