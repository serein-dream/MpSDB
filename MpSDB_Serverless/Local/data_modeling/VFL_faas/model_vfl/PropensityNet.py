import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class BottomModel(nn.Module):
    def __init__(self, input_size):
        super(BottomModel, self).__init__()
        self.dense = nn.Linear(in_features=input_size, out_features=input_size)
        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        x = x.float()
        x = self.dense(x)
        # x = torch.tanh(x)
        x = torch.relu(x)
        return x


class MiddleModel(nn.Module):
    def __init__(self, input_size):
        super(MiddleModel, self).__init__()
        self.dense = nn.Linear(in_features=input_size, out_features=input_size)
        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        x = x.float()
        x = self.dense(x)
        # x = torch.tanh(x)
        x = torch.relu(x)

        return x


class TopModel(nn.Module):
    def __init__(self, phase):
        super(TopModel, self).__init__()
        self.phase = phase
        self.ps_out = nn.Linear(in_features=25, out_features=2)

    def forward(self, x):
        # if torch.cuda.is_available():
        #     x = x.float().cpu()
        # else:
        #     x = x.float()
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.ps_out(x)
        if self.phase == "eval":
            return F.softmax(x, dim=-1)
        else:
            return x
