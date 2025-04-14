import torch
from torch.utils.data import TensorDataset
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.Linear import TopModel
from conf import args_parser
from label_owner.LinearLabelOwner import LinearLabelOwner
from data.data_partition import DataPartition
from model.utils import seed_torch,get_device
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
import oss2

seed_torch()


def launch_label_owner_server(label_set,args):
    device = get_device()
    model = TopModel(1).to(device)
    lo = LinearLabelOwner(model, label_set, args)
    lo.run_while()

if __name__ == '__main__':
    args = args_parser()
    csv_path = "ADD_by_yourself/FaaS_FL2023/VFL/data/wine_data/data_owner_3/train_wine_client2.csv"#注意在最后一个
    dL = DataPartition(csv_path, 1)
    lab = 1
    datadict = dL.getwineTensor(csv_path, 1)
    train_y_t = datadict["wine_trainData"][2].cuda()
    label_set = torch.utils.data.TensorDataset(train_y_t)
    print(label_set.__getitem__([0, 1]))
    launch_label_owner_server(label_set, args)
