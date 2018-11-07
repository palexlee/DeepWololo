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
    
class LeoNet(nn.Module):
    def __init__(self):
        super(LeoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5)
        self.mxp1 = nn.MaxPool2d(kernel_size = 4)
        self.view = View([-1])
        self.fc1 = nn.Linear(576, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.mxp1(self.conv1(x)))
        x = self.view(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def shadow_model() :
    model = nn.Sequential(
          nn.Conv2d(1, 32, kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(32 ,32, kernel_size=3),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout(0.25),

          nn.Conv2d(32, 64, kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(64 ,64, kernel_size=3),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout(0.25),

          View([-1]),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(512, 10)
    )
    return model
    
def CifarOverfit():
    model = nn.Sequential(
        View([-1]),
        nn.Linear(3072, 128),
        nn.ReLU(),
        
        nn.Linear(128, 32),
        nn.ReLU(),
        
        nn.Linear(32, 10)
    )
    return model
  
def denseG(vector_size):
    model = nn.Sequential(
        View([-1]),
        nn.Linear(vector_size, 128),
        nn.RReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(128, 128),
        nn.RReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(128, 2),
    )
    return model
    
