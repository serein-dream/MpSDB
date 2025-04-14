#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conf import args_parser

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)

        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer_hidden(x)

        x = self.layer_hidden(x)        
        return self.softmax(x)

class BottomModel(nn.Module):
    def __init__(self, input_size):#dim_in
        super(BottomModel, self).__init__()
        self.layer_input = nn.Linear(in_features=input_size, out_features=input_size*5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)

        return x


class MiddleModel(nn.Module):
    def __init__(self, input_size, out_features):#dim_hidden
        super(MiddleModel, self).__init__()
        #args = args_parser()
        #num = args.num_users
        self.layer_hidden = nn.Linear(in_features=input_size, out_features=out_features)

    def forward(self, x):
        x = self.layer_hidden(x)

        return x


class TopModel(nn.Module):
    def __init__(self,dim_out):
        super(TopModel, self).__init__()
        self.layer_input = nn.Linear(in_features=10, out_features=dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_input(x) 
        #x = self.softmax(x)  #mnist
        x = torch.sigmoid(x)  #cancer
        #wine
        return x
