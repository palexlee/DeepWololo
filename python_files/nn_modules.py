import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import Tensor

class View(nn.Module):
    """View module to transform input between layers."""
    def __init__(self, dims):
        super(View, self).__init__()
        self.dims = dims
   
    def forward(self, x):
        return x.view(x.shape[0], *self.dims)
    
    def __repr__(self):
        return "View(_, {})".format(self.dims)

class Rescale(nn.Module):
    """Module to rescale an input in {0, 1} to {-1, 1}."""
    def __init__(self):
        super(Rescale, self).__init__()
   
    def forward(self, x):
        return (2*x) - 1
    
    def __repr__(self):
        return "Rescale to -1, 1"
    
class Unsqueeze(nn.Module):
    """Module to add extra leading dimension to input."""
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
   
    def forward(self, x):
        return x.unsqueeze(self.dim)
    
    def __repr__(self):
        return "Unsqueeze({})".format(self.dim)