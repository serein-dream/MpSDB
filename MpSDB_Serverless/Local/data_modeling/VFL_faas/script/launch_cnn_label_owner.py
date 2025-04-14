import torch
from torch.utils.data import TensorDataset
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.CNN import TopModel
from conf import args_parser
from label_owner.CNNLabelOwner import CNNLabelOwner
from data.mnist.data_partition import MnistData
from model.utils import seed_torch,get_device
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser

seed_torch()


def launch_label_owner_server(label_list,args):
    
    args = args_parser()
    device = get_device()
    model = TopModel(args.num_classes).to(device)
    lo = CNNLabelOwner(model, label_list, args)
    lo.run_while()

if __name__ == '__main__':
    args = args_parser()
    #print()
    
    # print(f"model:  {args.model}")
    # print(f"num_users:  {args.num_users}")
    # print(f"rounds:  {args.epoch}")
    # print(f"batch_size:  {args.batch_size}")
    # print(f"learning rate:  {args.lr}\n")
    # print(f"CNN Label Owner start\nepoch:1   batch_size:20000   batch in one epoch: 1    loss:0.26618823409080505\nsend grad to Serverless middle model\nepoch:1   batch_size:20000   batch in one epoch: 2    loss:0.267877459526062\nsend grad to Serverless middle model\nepoch:1   batch_size:20000   batch in one epoch: 3    loss:0.25242337584495544\nsend grad to Serverless middle model\nepoch:2   batch_size:20000   batch in one epoch: 1    loss:0.2695057988166809\nsend grad to Serverless middle model\nepoch:2   batch_size:20000   batch in one epoch: 2    loss:0.25678911805152893\nsend grad to Serverless middle model\nepoch:2   batch_size:20000   batch in one epoch: 3    loss:0.25590330362319946\nsend grad to Serverless middle model\nTrain done successfully!     test accuracy:  97.1%  ")

    mn = MnistData()
    partitioned_train_loader,_ = mn.get_data_loader(args.rank)
    label_list = partitioned_train_loader.labels
    launch_label_owner_server(label_list, args)
