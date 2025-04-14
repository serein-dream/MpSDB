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
        x = torch.tanh(x)

        return x


class MiddleModel(nn.Module):
    def __init__(self, input_size):
        super(MiddleModel, self).__init__()
        self.dense = nn.Linear(in_features=input_size, out_features=200)
        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        x = x.float()
        x = self.dense(x)
        x = torch.relu(x)

        return x


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel, self).__init__()
        # representation
        # self.rep_fc1 = nn.Linear(in_features=25, out_features=200)
        # nn.init.xavier_uniform_(self.rep_fc1.weight)

        self.rep_fc2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.rep_fc2.weight)

        self.rep_fc3 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.rep_fc3.weight)

        self.t_predictions = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.t_predictions.weight)

        # HYPOTHESIS
        self.y0_hidden_fc1 = nn.Linear(in_features=200, out_features=100)
        nn.init.xavier_uniform_(self.y0_hidden_fc1.weight)
        self.y1_hidden_fc1 = nn.Linear(in_features=200, out_features=100)
        nn.init.xavier_uniform_(self.y1_hidden_fc1.weight)
        self.y0_hidden_fc2 = nn.Linear(in_features=100, out_features=100)
        nn.init.xavier_uniform_(self.y0_hidden_fc2.weight)
        self.y1_hidden_fc2 = nn.Linear(in_features=100, out_features=100)
        nn.init.xavier_uniform_(self.y1_hidden_fc2.weight)

        self.y0_predictions = nn.Linear(in_features=100, out_features=1)
        nn.init.xavier_uniform_(self.y0_predictions.weight)
        self.y1_predictions = nn.Linear(in_features=100, out_features=1)
        nn.init.xavier_uniform_(self.y1_predictions.weight)

        # EpsilonLayer
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        nn.init.xavier_uniform_(self.epsilon.weight)

    def forward(self, x):
        x = x.float()
        # x = F.relu(self.rep_fc1(x))
        x = F.relu(self.rep_fc2(x))
        x = F.relu(self.rep_fc3(x))
        t_predict = self.t_predictions(x)
        t_predict = torch.sigmoid(t_predict)

        y0_hidden = F.elu(self.y0_hidden_fc1(x))
        y0_hidden = F.elu(self.y0_hidden_fc2(y0_hidden))

        y1_hidden = F.elu(self.y1_hidden_fc1(x))
        y1_hidden = F.elu(self.y1_hidden_fc2(y1_hidden))

        y0_predict = self.y0_predictions(y0_hidden)
        y1_predict = self.y1_predictions(y1_hidden)

        epsilons = self.epsilon(t_predict)

        concat_pred = torch.cat((y0_predict, y1_predict, t_predict, epsilons), dim=-1)

        return concat_pred
