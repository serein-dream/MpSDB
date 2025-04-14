import os,sys
import time
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.MLP_trainer import MLPTrainer
from data.mnist.data_partition import MnistData
from conf import args_parser
import torch
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from model.utils import seed_torch

seed_torch()

def run(arg):
    time1 = time.perf_counter()

    rank = arg.rank    
    csv_path = "ADD_by_yourself/FaaS_FL2023/VFL/data/cancer/train_cancer_client{0}.csv".format(rank)
    dL = DataPartition(csv_path, 1)
    lab = args.num_users - rank
    datadict = dL.getwineTensor(csv_path, lab)
    idx_train = datadict["wine_trainData"][0]
    Logistic_train_x = datadict["wine_trainData"][1]
    train_dataset = TensorDataset(idx_train, Logistic_train_x)
    n_f = Logistic_train_x.shape[1]
    print(f"n_f: {n_f}")
    MLP_trainer = MLPTrainer(n_f, arg, rank, train_dataset)
    
    time2 = time.perf_counter()
    
    MLP_trainer.launch()
    
    time3 = time.perf_counter()
    # print(f"\n\nnum:{MLP_trainer.bottom_commu_num}\nbottom_commu_time:{MLP_trainer.bottom_commu_time}\nall_time:{time3 - time2 + (time2 - time1)/10}\ncompute:{MLP_trainer.comp_time}\n")

    print(f"\n\nnum:{MLP_trainer.bottom_commu_num}\ncompute:{MLP_trainer.comp_time}\n")


if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()
    print(f"time:{et - st}")
