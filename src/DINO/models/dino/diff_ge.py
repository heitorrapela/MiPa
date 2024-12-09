import torch
from torch import nn
from torch.nn import functional as F

class DifferentiableGreaterEqual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        nonzero_min = x[x>0].min()
        return (torch.clamp(F.relu(x), max=nonzero_min) / nonzero_min).unsqueeze(-1) 

class DifferentiableStepUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        nonzero_min = x[x>0].min()
        return (torch.clamp(F.relu(x), max=nonzero_min) / nonzero_min)#.unsqueeze(-1) 