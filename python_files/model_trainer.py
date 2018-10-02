import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import Tensor

from history import History

import numpy as np
import copy

class ModelTrainer(object):
    """Utility class to train models. Compatible with SimpleNet and PyTorch models.
    Usage:
        model = ...
        train_data = (x, y)
        test_data = (...)
        mt = ModelTrainer(model, MSELoss(), SGD(model.parameters()))
        mt.fit(train_data, test_data, epochs=250, batch_size=100, verbose=10)
        mt.plot_training("Learning curves")
    """
    def __init__(self, model, criterion, optimizer, y_hat_fun=lambda y: y, criterion_fun=lambda x, y:(x, y)):
        """Initialize a ModelTrainer.
        :argument model a SimpleNet or PyTorch model
        :argument criterion the loss function, see criterion.py
        :argument optimizer the optimization algo to use, see optimizers.py
        :argument y_hat_fun function to process the output of the last layer
        :argument criterion_fun function to process the output before passing it to the criterion"""
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.y_hat_fun = y_hat_fun
        self.criterion_fun = criterion_fun
        self.best_model = None
            
    def fit(self, train_data, validation_data=None, epochs=25, batch_size=None, verbose=1):
        """Fit the model on the training data.
        :argument train_data (x_train, y_train)
        :argument validation_data (x_validation, y_validation)
        :argument epochs nb of epochs to train
        :argument batch_size the mini-batchs size
        :argument verbose print the current loss and accuracies if current_epoch%verbose == 0"""
        x_train, y_train = train_data
        self.history = History()
        
        if x_train.is_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        for epoch in range(epochs):
            #Set the training mode for dropout
            #or bathNorm to apply
            self.model.train(True)
            
            idxs = torch.randperm(x_train.shape[0])
            if(x_train.is_cuda): 
                idxs = idxs.cuda()
            idxs = [idxs] if batch_size is None else idxs.split(batch_size)
            
            train_correct = 0
            train_loss = 0
            for batch in idxs:
                self.optimizer.zero_grad()
                #torch.cuda.synchronize()
                y_hat = self.model(x_train[batch])
                #torch.cuda.synchronize()
                loss = self.criterion(*self.criterion_fun(y_hat, y_train[batch]))
                #torch.cuda.synchronize()
                loss.backward()
                #torch.cuda.synchronize()
                self.optimizer.step()
                #torch.cuda.synchronize()
                train_correct += (self.y_hat_fun(y_hat) == y_train[batch]).sum()
                #torch.cuda.synchronize()
                train_loss += loss
            
            val_loss = np.nan
            val_acc = np.nan
            
            if validation_data is not None:
                #Disable training mode
                self.model.train(False)
                
                x_test, y_test = validation_data
                y_hat = self.model(x_test)
                val_loss = self.criterion(*self.criterion_fun(y_hat, y_test)).item()/len(x_test)
                val_acc = (self.y_hat_fun(y_hat)==y_test).float().sum().item()/len(x_test)
                
            
            self.history.add([
                train_loss.item()/x_train.shape[0],
                train_correct.item()/x_train.shape[0],
                val_loss,
                val_acc
            ])
            
            if verbose != 0 and epoch%verbose == 0:
                if epoch==0:
                    print("******************************** Train log ************************************")
                print(self.history.get_last().to_string(col_space=15, header=epoch==0, formatters=History.formatters))

        if verbose!=0:
            print(self.history.get_last().to_string(col_space=15, header=False, formatters=History.formatters))
            print("*******************************************************************************")
            
        self.best_model = copy.deepcopy(self.model)
        
        return self.history

    def get_best_model(self):
        return self.best_model
    
    def plot_training(self, title, avg_w_size=20):
        if self.history is None:
            print("Train model first!")
        else:
            self.history.plot(title, avg_w_size)