import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import Tensor
from nn_modules import View

#Basic LeNet architecture for MNIST
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#Model to differentiate train from test layer activations
class aliG(nn.Module):
    def __init__(self, vector_size):
        super(aliG, self).__init__()
        
        self.finalVectorSize = (((((vector_size-7)//2)-4)//2)-2)//2
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size = 8)
        self.maxpool1 = nn.MaxPool1d(kernel_size = 2)
        
        self.conv2 = nn.Conv1d(32, 128, kernel_size = 5)
        self.maxpool2 = nn.MaxPool1d(kernel_size = 2)
        
        self.conv3 = nn.Conv1d(128, 16, kernel_size = 3)
        self.maxpool3 = nn.MaxPool1d(kernel_size = 2)
        
        self.reshape = View([-1])
        self.fc1 = nn.Linear(16*self.finalVectorSize, 100)
        self.fc2 = nn.Linear(100, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.maxpool2(self.conv2(x)))
        x = F.relu(self.maxpool3(self.conv3(x)))
        x = self.reshape(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    