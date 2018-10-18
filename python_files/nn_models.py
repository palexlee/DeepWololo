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
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.maxpool2(self.conv2(x)))
        x = F.relu(self.maxpool3(self.conv3(x)))
        x = self.reshape(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def aliGs(vector_size):
    finalVectorSize = (((((vector_size-7)//2)-4)//2)-2)//2
    
    model = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size = 8),
        nn.RReLU(),
        nn.BatchNorm1d(32),
        nn.MaxPool1d(kernel_size = 2),
        
        nn.Conv1d(32, 64, kernel_size = 5),
        nn.RReLU(),
        nn.BatchNorm1d(64),
        nn.MaxPool1d(kernel_size = 2),
        
        nn.Conv1d(64, 16, kernel_size = 3),
        nn.RReLU(),
        nn.BatchNorm1d(16),
        nn.MaxPool1d(kernel_size = 2),
        
        View([-1]),
        nn.Linear(16* finalVectorSize, 100),
        nn.RReLU(),
        nn.Dropout(),
        
        nn.Linear(100, 32),
        nn.RReLU(),
        nn.Dropout(),
        
        nn.Linear(32, 2)
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
    
