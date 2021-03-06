import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import Tensor

from history import History

import numpy as np
import copy

from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

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
    def __init__(self, model, criterion, optimizer, y_hat_fun=lambda y: y, criterion_fun=lambda x, y:(x, y), batch_fun=lambda x, y: x,
                 tsx_name=None, embedding_log=10, nb_labels=10):
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
        self.batch_fun = batch_fun
        self.best_model = None
        self.writer = SummaryWriter(tsx_name)
        self.use_tensorboard = tsx_name is not None
        self.embedding_log = embedding_log
        self.nb_labels = nb_labels

    def fit(self, train_data, validation_data=None, epochs=25, batch_size=None, verbose=1):
        """Fit the model on the training data.
        :argument train_data (x_train, y_train)
        :argument validation_data (x_validation, y_validation)
        :argument epochs nb of epochs to train
        :argument batch_size the mini-batchs size
        :argument verbose print the current loss and accuracies if current_epoch%verbose == 0"""
        x_train, y_train = train_data
        self.history = History()
        init_batch_size = batch_size

        if x_train.is_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        for epoch in range(epochs):
            #Set the training mode for dropout
            #or bathNorm to apply
            self.model.train(True)
            self.criterion.train(True)

            if batch_size is not None:
                batch_size = self.batch_fun(init_batch_size, epoch)

            idxs = torch.randperm(x_train.shape[0])
            if(x_train.is_cuda):
                idxs = idxs.cuda()

            idxs = [idxs] if batch_size is None else idxs.split(batch_size)

            train_acc = 0
            train_loss = 0
            for batch in idxs:
                self.optimizer.zero_grad()
                y_hat = self.model(x_train[batch])
                loss = self.criterion(*self.criterion_fun(y_hat, y_train[batch]))
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    # weighted accuracy
                    batch_idx_ones = y_train[batch] == 1
                    batch_idx_zeros = y_train[batch] == 0
                    y_correct_1 = (self.y_hat_fun(y_hat[batch_idx_ones]) == y_train[batch[batch_idx_ones]]).long().sum().item()
                    y_correct_0 = (self.y_hat_fun(y_hat[batch_idx_zeros]) == y_train[batch[batch_idx_zeros]]).long().sum().item()
                    # regular accuracy
                    train_acc += (self.y_hat_fun(y_hat) == y_train[batch]).long().sum().item()/x_train.shape[0]
                    train_loss += loss.item()/len(idxs)

                    val_acc = np.nan
                    val_loss = np.nan
            
            # weighted accuracy
            total_ones = y_train[y_train == 1].shape[0]
            total_zeros = y_train[y_train == 0].shape[0]
            train_weighted_acc = (y_correct_1/total_ones + y_correct_0/total_zeros)/2.


            if validation_data is not None:
                #Disable training mode
                with torch.no_grad():
                    self.model.eval()
                    self.criterion.eval()

                    x_test, y_test = validation_data
                    y_hat_val = self.model(x_test)
                    
                    # regular accuracy
                    val_acc = (self.y_hat_fun(y_hat_val) == y_test).long().sum().item()/x_test.shape[0]
                    
                    # weighted accuracy
                    idx_ones = y_test == 1
                    idx_zeros = y_test == 0
                    y_correct_val_1 = (self.y_hat_fun(y_hat_val[idx_ones]) == y_test[idx_ones]).long().sum().item()
                    y_correct_val_0 = (self.y_hat_fun(y_hat_val[idx_zeros]) == y_test[idx_zeros]).long().sum().item()
                    total_ones_val = y_test[y_test == 1].shape[0]
                    total_zeros_val = y_test[y_test == 0].shape[0]
                    val_weighted_acc = (y_correct_val_1/total_ones_val + y_correct_val_0/total_zeros_val)/2.
                    
                    val_loss = self.criterion(*self.criterion_fun(y_hat_val, y_test)).item()

                    if self.use_tensorboard:
                        for i in range(self.nb_labels):
                            idx = y_test == i
                            if y_test.is_cuda:
                                idx = idx.cuda()

                            self.writer.add_histogram('population/{}'.format(i), y_hat_val[idx, i].cpu().data.numpy(), epoch)
                        if self.nb_labels == 2:
                            label_1 = y_test == 1
                            label_0 = y_test == 0
                            self.writer.add_histogram('diff_population/0', (y_hat_val[label_0,1] - y_hat_val[label_0, 0]).cpu().data.numpy(), epoch)
                            self.writer.add_histogram('diff_population/1', (y_hat_val[label_1,1] - y_hat_val[label_1, 0]).cpu().data.numpy(), epoch)

                    """if epoch % self.embedding_log == 0:
                    # we need 3 dimension for tensor to visualize it!
                    y_hat_val = y_hat_val[:100]
                    y_hat_val = torch.cat((y_hat_val.cpu().data, torch.ones(len(y_hat_val), 1)), 1)
                    self.writer.add_embedding(
                        y_hat_val.cpu(),
                        metadata=y_test[:100].cpu().data,
                        label_img=x_test[:100].cpu().data,
                        global_step=epoch)"""

            if self.use_tensorboard:

                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), epoch)

                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Train/Weighted accuracy', train_weighted_acc)
                self.writer.add_scalar('Eval/Loss', val_loss, epoch)
                self.writer.add_scalar('Eval/Accuracy', val_acc, epoch)
                self.writer.add_scalar('Eval/Weighted accuracy', val_weighted_acc)


            self.history.add([
                train_loss,
                #train_acc,
                train_weighted_acc,
                val_loss,
                #val_acc,
                val_weighted_acc
            ])

            if verbose != 0 and epoch%verbose == 0:
                if epoch==0:
                    print("******************************** Train log ************************************")
                print(self.history.get_last().to_string(col_space=15, header=epoch==0, formatters=History.formatters))

        if verbose!=0:
            print(self.history.get_last().to_string(col_space=15, header=False, formatters=History.formatters))
            print("*******************************************************************************")

        self.best_model = copy.deepcopy(self.model)
        self.writer.close()

        return self.history

    def get_best_model(self):
        return self.best_model

    def plot_training(self, title, avg_w_size=20):
        if self.history is None:
            print("Train model first!")
        else:
            self.history.plot(title, avg_w_size)
