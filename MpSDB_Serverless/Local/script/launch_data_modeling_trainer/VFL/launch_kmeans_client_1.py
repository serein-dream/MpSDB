import os
import sys
import time
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.tenseal_K_means_trainer import KmeansTrainer
from data.data_partition import DataPartition
from conf import args_parser
from model.utils import seed_torch
seed_torch()
def load_data(csv_path, label):
    dL = DataPartition(csv_path, 1)
    _, dataset_x_np = dL.wine_preprocess_data_from_csv(csv_path, label)
    return dataset_x_np
def run(arg):
    rank = arg.rank
    train_csv_path = "/path/to/train_wine_client{0}.csv".format(rank)
    test_csv_path = "/path/to/test_wine_client{0}.csv".format(rank)
    label = args.num_users - rank
    train_dataset_x_np = load_data(train_csv_path, label)
    test_dataset_x_np = load_data(test_csv_path, label)
    n_f = train_dataset_x_np.shape[1]
    Kmeans_trainer = KmeansTrainer(n_f, arg, rank, train_dataset_x_np, test_dataset_x_np)
    Kmeans_trainer.launch()
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()