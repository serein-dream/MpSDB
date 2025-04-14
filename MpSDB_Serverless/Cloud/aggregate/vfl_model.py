import torch
from torch import nn
import torch.nn.functional as F

class MLP_MiddleModel(nn.Module):
    def __init__(self)
        super(MLP_MiddleModel, self).__init__()
        self.layer_hidden = nn.Linear(in_features=150, out_features=10)

    def forward(self, x):
        x = self.layer_hidden(x)
        return x

class CNN_MiddleModel(nn.Module):
    def __init__(self):
        super(CNN_MiddleModel, self).__init__()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        return x
