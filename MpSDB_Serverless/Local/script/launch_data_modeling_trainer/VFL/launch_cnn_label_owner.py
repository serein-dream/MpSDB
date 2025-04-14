import torch
from torch.utils.data import TensorDataset
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.CNN import TopModel
from conf import args_parser
from label_owner.CNNLabelOwner import CNNLabelOwner
from data.mnist.data_partition import MnistData
from model.utils import seed_torch, get_device
seed_torch()
def load_labels(rank):
    mn = MnistData()
    partitioned_train_loader, _ = mn.get_data_loader(rank)
    return partitioned_train_loader.labels
def launch_label_owner_server(label_list, args):
    device = get_device()
    model = TopModel(args.num_classes).to(device)
    lo = CNNLabelOwner(model, label_list, args)
    lo.run_while()
if __name__ == '__main__':
    args = args_parser()
    args.rank = 0
    label_list = load_labels(args.rank)
    launch_label_owner_server(label_list, args)