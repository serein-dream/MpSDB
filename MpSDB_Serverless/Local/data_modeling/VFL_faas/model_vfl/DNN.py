import torch
from torch import nn


class DNN(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.dense_1 = nn.Linear(n_f, n_f)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(n_f, 2)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        return x

class BottomModel(nn.Module):
    def __init__(self, input_size):
        super(BottomModel, self).__init__()
        self.dense = nn.Linear(in_features=input_size, out_features=input_size)
        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        x = x.float()
        x = self.dense(x)
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
        x = torch.relu(x)

        return x


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel, self).__init__()
        self.t_predictions = nn.Linear(in_features=input_size, out_features=2)
        nn.init.xavier_uniform_(self.t_predictions.weight)

    def forward(self, x):
        x = x.float()
        t_predict = self.t_predictions(x)

        return t_predict
