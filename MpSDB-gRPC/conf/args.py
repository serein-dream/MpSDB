#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

#mlp 40 0.05 0.97
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='the client rank')
    parser.add_argument('--num_users',
                        default=3,
                        type=int,
                        help='the client numbers')
    parser.add_argument('--rounds',
                        default=10,
                        type=int,
                        help='total communication rounds')
    parser.add_argument('--epoch',
                        default=20,
                        type=int,
                        help='number of local epochs')
    parser.add_argument('--batch_size',
                        default=512,
                        type=int,
                        help='local batch size')
    parser.add_argument('--lr',
                        default=0.05,
                        type=float,
                        help='learning rate')
    parser.add_argument('--seed',
                        default=1231,
                        type=int,
                        help='random seed')
    parser.add_argument('--trainer_address',
                        default='127.0.0.1:60000',
                        type=str,
                        help='init method')
    parser.add_argument('--server_address',
                        default='127.0.0.1:60091',
                        type=str,
                        help='init method')
    parser.add_argument('--sample_num',
                        default=3,
                        type=int,
                        help='local_count/sum_count')
    parser.add_argument('--n_features',
                        default=28*28,
                        type=int,
                        help='the number of features')
    parser.add_argument('--ctx_file',
                        default='/home/maxvyang01/FaaS_FL2023/SecureDatabase/transmission/ts_ckks.config',
                        type=str,
                        help='the number of features')
    
    #cnn
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset',
                        default='mnist',
                        type=str,
                        help='')
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        help='the gpu rank')
    parser.add_argument('--local_bs', type=int, default=512,  
                        help="local batch size: B   len(train_idx) = local_bs * batch(伦次)")
    #kmeans
    parser.add_argument('--n_clusters', type=int, default=3,  
                        help="k the number of centroids")
    parser.add_argument('--n_dims', type=int, default=11,  
                        help="k the number of centroids")
    args = parser.parse_args()
    return args
