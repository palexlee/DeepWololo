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
