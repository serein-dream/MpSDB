import torch
from torch import nn
import torch.nn.functional as F
class LinearModel(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.linear = nn.Linear(n_f, 1)
    def forward(self, x):
        return self.linear(x)
class LogisticModel(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.linear = nn.Linear(n_f, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
class MLPModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.layer_input = nn.Linear(dim_in, int(dim_hidden * 1.5))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(int(dim_hidden * 1.5), dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.dropout(self.relu(self.layer_input(x)))
        x = self.sigmoid(self.layer_hidden2(self.layer_hidden1(x)))
        return x
class CNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)