import os,sys
import time
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.Logistic_trainer import LogisticTrainer
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from conf import args_parser
from model.utils import seed_torch

seed_torch()


def run(arg):
    rank = arg.rank    
    csv_path = "ADD_by_yourself/FaaS_FL2023/VFL/data/cancer/train_cancer_client{0}.csv".format(rank)
    dL = DataPartition(csv_path, 1)
    lab = args.num_users - rank
    datadict = dL.getwineTensor(csv_path, lab)
    idx_train = datadict["wine_trainData"][0]
    Logistic_train_x = datadict["wine_trainData"][1]
    train_dataset = TensorDataset(idx_train, Logistic_train_x)
    n_f = Logistic_train_x.shape[1]
    Logistic_trainer = LogisticTrainer(n_f, arg, rank, train_dataset)
    Logistic_trainer.launch()


if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 1
    run(args)
    et = time.time()
    print(f"time:{et - st}")
