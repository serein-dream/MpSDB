import torch
from torch import nn


class Logistic(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.linear = nn.Linear(n_f, 1)

    def forward(self, x):
        x = self.linear(x)
        y = torch.sigmoid(x)
        return y
    
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
        #x = x
        return x


class TopModel(nn.Module):
    def __init__(self,dim_out):
        super(TopModel, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        #x = x
        y = torch.sigmoid(x)
        return y
    '''
    #relu
    def __init__(self, n_f):
        super().__init__()
        self.linear1=torch.nn.Linear(n_f,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.activate=torch.nn.ReLU()
    def forward(self,x):
        x=self.activate(self.linear1(x))
        x=self.activate(self.linear2(x))
        x=self.sigmoid(self.linear3(x))

        return x
    '''
    '''
    #tanh
    def __init__(self, n_f):
        super().__init__()
        self.linear1=torch.nn.Linear(n_f,16)
        self.linear2=torch.nn.Linear(16,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.activate=torch.nn.Tanh()
    def forward(self,x):
        x=self.activate(self.linear1(x))
        x=self.activate(self.linear2(x))
        x=self.sigmoid(self.linear3(x))

    '''