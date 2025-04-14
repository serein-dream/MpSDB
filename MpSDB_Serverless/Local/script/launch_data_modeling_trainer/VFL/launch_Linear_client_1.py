import os
import sys
import time
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.Linear_trainer import LinearTrainer
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from conf import args_parser
from model.utils import seed_torch
seed_torch()
def load_data(csv_path, label):
    dL = DataPartition(csv_path, 1)
    datadict = dL.getwineTensor(csv_path, label)
    idx_train = datadict["wine_trainData"][0]
    Linear_train_x = datadict["wine_trainData"][1]
    dataset = TensorDataset(idx_train, Linear_train_x)
    n_f = Linear_train_x.shape[1]
    return dataset, n_f
def run(arg):
    rank = arg.rank
    csv_path = "/path/to/train_wine_client{0}.csv".format(rank)
    label = args.num_users - rank
    dataset, n_f = load_data(csv_path, label)
    Linear_trainer = LinearTrainer(n_f, arg, rank, dataset)
    Linear_trainer.launch()
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()