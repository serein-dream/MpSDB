import torch
from torch import nn


class Linear(nn.Module):  
    #原方法
    def __init__(self, n_f):
        super().__init__()
        self.linear = nn.Linear(n_f, 1)

    def forward(self, x):
        x = self.linear(x)
        return x
    
class BottomModel(nn.Module):
    def __init__(self, input_size):#dim_in
        super(BottomModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


class MiddleModel(nn.Module):
    def __init__(self):#dim_hidden
        super(MiddleModel, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        x = x
        return x


class TopModel(nn.Module):
    def __init__(self,dim_out):
        super(TopModel, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        x = x
        return x