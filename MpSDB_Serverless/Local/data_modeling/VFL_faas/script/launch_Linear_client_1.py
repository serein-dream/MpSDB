import os,sys
import time
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.Linear_trainer import LinearTrainer
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from conf import args_parser
from model.utils import seed_torch

seed_torch()


def run(arg):
    rank = arg.rank    
    csv_path = "ADD_by_yourself/FaaS_FL2023/VFL/data/wine_data/data_owner_3/train_wine_client{0}.csv".format(rank)
    dL = DataPartition(csv_path, 1)
    lab = args.num_users - rank
    datadict = dL.getwineTensor(csv_path, lab)
    idx_train = datadict["wine_trainData"][0]
    Linear_train_x = datadict["wine_trainData"][1]
    dataset = TensorDataset(idx_train, Linear_train_x)
    n_f = Linear_train_x.shape[1]
    Linear_trainer = LinearTrainer(n_f, arg, rank, dataset)
    Linear_trainer.launch()

    print(f"\n\nnum:{Linear_trainer.bottom_commu_num}\ncompute:{Linear_trainer.comp_time}\nwait_time:{Linear_trainer.wait_time}\n")

if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()
    print(f"time:{et - st}")
