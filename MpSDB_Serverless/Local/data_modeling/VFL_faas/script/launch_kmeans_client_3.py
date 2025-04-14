import os,sys
import time
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.tenseal_K_means_trainer import KmeansTrainer
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from conf import args_parser
from model.utils import seed_torch

seed_torch()


def run(arg):
    rank = arg.rank    
    train_csv_path = "ADD_by_yourself/FaaS_FL2023/VFL/data/wine_data/data_owner_3/train_wine_client{0}.csv".format(rank)
    dL = DataPartition(train_csv_path, 1)
    lab = args.num_users - rank
    _, train_dataset_x_np, _ = dL.wine_preprocess_data_from_csv(train_csv_path, lab)
    test_csv_path = "ADD_by_yourself/FaaS_FL2023/VFL/data/wine_data/data_owner_3/test_wine_client{0}.csv".format(rank)
    dL = DataPartition(test_csv_path, 1)
    lab = args.num_users - rank
    _, test_dataset_x_np, _ = dL.wine_preprocess_data_from_csv(test_csv_path, lab)
    n_f = train_dataset_x_np.shape[1]
    Kmeans_trainer =KmeansTrainer(n_f, arg, rank, train_dataset_x_np, test_dataset_x_np)
    Kmeans_trainer.launch()


if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 2
    run(args)
    et = time.time()
    print(f"time:{et - st}")
