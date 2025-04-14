from torch.multiprocessing import Process
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.mlp_trainer import MLPTrainer
import torch
from data_modeling.src.data_utils import only_get_dataset
from data_modeling.data_loader import MysqlDataSet
import numpy as np

import time
def run(arg):
    before_train_start_time = time.perf_counter()
    loss_time = []
    ini_time_s = time.perf_counter()

    cols_list = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','y']
    mysql_dataset_train = MysqlDataSet(f"database_1", "cancer", cols_list)
    mysql_dataset_eval = MysqlDataSet("test_data", "cancer", cols_list)

    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - ini_time_s
    print(args.rank)
    lr_trainer = MLPTrainer(args, mysql_dataset_train, mysql_dataset_eval, args.rank)

    for rnd in range(args.rounds):
        print("round: ", rnd)
        update_flag = lr_trainer.is_update()
        print(update_flag)
        if update_flag:
            lr_trainer.one_local_round(rnd)
            #lr_trainer.test(test_dataset)
            l_time =  time.perf_counter() - ini_time_s
        else:
            lr_trainer.set_loss_a()
            print("not participate in this round")
            l_time = 0
        loss_time.append(l_time)
    # lr_trainer.write(load_data_time)
    end_train_time =  time.perf_counter()
    total_time = end_train_time - before_train_start_time
    # print(f"rank--{args.rank}--rounds--{args.rounds}--epoch--{args.epoch}MLP total time: {total_time}")
    # pa  = "/home/maxvyang01/SecureDatabase/script/l_result/temp/mlp"
    # with open(f"{pa}/loss_time.txt", 'w') as train_los_0:
    #     train_los_0.write(str(loss_time))

if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 6
    args.model = "mlp"
    processes = []
    run(args)
    et = time.time()
    print(f"time:{et - st}")
    time.sleep(args.rank*0.2)
    with open(f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_modeling_trainer/log.txt", 'a') as train_los_0:
        train_los_0.write(f"rank {args.rank}  time:{et - st}\n")
        if args.rank==args.num_users-1:
            train_los_0.write(f"\n\n")