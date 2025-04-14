import os
import sys
import time
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.CNN_trainer import CNNTrainer
from data.mnist.data_partition import MnistData
from conf import args_parser
import torch
from model.utils import seed_torch
seed_torch()
def load_data(rank):
    mn = MnistData()
    partitioned_train_loader, _ = mn.get_data_loader(rank)
    partitioned_train_list = partitioned_train_loader.data_pointer
    return partitioned_train_list
def run(arg):
    partitioned_train_list = load_data(arg.rank)
    cnn_trainer = CNNTrainer(arg.batch_size, arg, arg.rank, partitioned_train_list)
    cnn_trainer.launch()
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    end_t = time.time()