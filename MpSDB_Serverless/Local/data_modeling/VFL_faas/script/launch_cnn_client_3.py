import os,sys
import time
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.CNN_trainer import CNNTrainer
from data.mnist.data_partition import MnistData
from conf import args_parser
import torch
from model.utils import seed_torch

seed_torch()

def run(arg):
    mn = MnistData()
    partitioned_train_loader,_ = mn.get_data_loader(arg.rank)
    partitioned_train_list = partitioned_train_loader.data_pointer
    #n_f = partitioned_train_list[0].shape[-1] * partitioned_train_list[0].shape[-2]
    #print(f"n_f: {n_f}, partitioned_train_list shape:{partitioned_train_list[0].shape}    {len(partitioned_train_list)}")
    CNN_Trainer = CNNTrainer(arg.batch_size, arg, arg.rank, partitioned_train_list)
    CNN_Trainer.launch()
    print(f"Train done successfully!")
    # print(f"\n\nnum:{CNN_Trainer.bottom_commu_num}\nbottom_commu_time:{CNN_Trainer.bottom_commu_time}\n\n")
    print(f"\n\nnum:{CNN_Trainer.bottom_commu_num}\ncompute:{CNN_Trainer.comp_time}\n")


if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 2
    run(args)
    end_t = time.time()
    print(f"run time:{end_t - st}")
