#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help='the client rank')
    parser.add_argument('--num_users', default=3, type=int, help='the client numbers')
    parser.add_argument('--rnd', default=30, type=int, help='the data / batch_size')
    parser.add_argument('--epoch', default=1, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='local batch size')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--kmeans_lr', default=0.05, type=float, help='learning rate for k-means')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--seed', default=3456, type=int, help='random seed')
    parser.add_argument('--label_owner_address', default='127.0.0.1:59290', type=str, help='init method')
    parser.add_argument('--server_address', default='127.0.0.1:59291', type=str, help='init method')
    parser.add_argument('--num_clients', default=2, type=int, help='local_count/sum_count')
    parser.add_argument('--ctx_file', default='ADD_by_yourself/transmission/ts_ckks.config', type=str, help='context file path')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels of imgs')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--n_clusters', type=int, default=3, help='k the number of centroids')
    parser.add_argument('--n_dims', type=int, default=11, help='the number of one point')
    args = parser.parse_args()
    args.rnd = calculate_rnd(args.model, args.batch_size)
    return args
def calculate_rnd(model, batch_size):
    if "LINEAR" in model.upper() or "KMEANS" in model.upper():
        return round(5120 / batch_size)
    elif "LR" in model.upper():
        return int(480 / batch_size)
    elif model.upper() == "MLP":
        return int(480 / batch_size)
    elif model.upper() == "CNN":
        return int(60000 / batch_size)
    else:
        raise ValueError(f"Unsupported model: {model}")