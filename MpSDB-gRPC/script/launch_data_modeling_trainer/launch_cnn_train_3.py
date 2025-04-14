from torch.multiprocessing import Process
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.cnn_trainer import CNNTrainer
import torch
from data_modeling.src.data_utils import only_get_dataset
import numpy as np
import time
def run(args):
    before_train_start_time = time.perf_counter()
    loss_time = []
    ini_time_s = time.perf_counter()

    filename = "/home/maxvyang01/FaaS_FL2023/faas_fed/script/launch_data_modeling_trainer/mnist_idx_3/cnn_clients_dataidx_" + str(args.rank) + ".txt"
    dataidx = np.loadtxt(filename, dtype=bytes).astype(int)
    dataidx_list = list(dataidx)
    train_dataset, test_dataset = only_get_dataset(args)
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - ini_time_s
    lr_trainer = CNNTrainer(args, train_dataset, test_dataset, dataidx_list, args.rank)
    # lr_trainer.get_data(train_dataset, test_dataset)
    # raise
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
    # end_train_time =  time.perf_counter()
    # total_time = end_train_time - before_train_start_time
    # print(f"rank--{args.rank}--rounds--{args.rounds}--epoch--{args.epoch}cnn total time: {total_time}")
    # pa  = "/home/maxvyang01/SecureDatabase/script/l_result/temp/cnn"
    # with open(f"{pa}/loss_time2.txt", 'w') as train_los_0:
    #     train_los_0.write(str(loss_time))
    print(f"com + aggre time: {lr_trainer.com_time}")

if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 2
    processes = []
    run(args)
    et = time.time()
    print(f"time:{et - st}")
