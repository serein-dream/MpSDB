import torch
from torch.utils.data import TensorDataset
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.Linear import TopModel
from conf import args_parser
from label_owner.LinearLabelOwner import LinearLabelOwner
from data.data_partition import DataPartition
from model.utils import seed_torch, get_device
seed_torch()
def load_labels(csv_path, client_id):
    dL = DataPartition(csv_path, client_id)
    datadict = dL.getwineTensor(csv_path, client_id)
    train_y_t = datadict["wine_trainData"][2].cuda()
    return TensorDataset(train_y_t)
def launch_label_owner_server(label_set, args):
    device = get_device()
    model = TopModel(1).to(device)
    lo = LinearLabelOwner(model, label_set, args)
    lo.run_while()
if __name__ == '__main__':
    args = args_parser()
    csv_path = "DATA_PATH"
    client_id = 1
    label_set = load_labels(csv_path, client_id)
    launch_label_owner_server(label_set, args)