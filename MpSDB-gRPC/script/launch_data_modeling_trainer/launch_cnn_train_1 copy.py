from torch.multiprocessing import Process
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.cnn_trainer import CNNTrainer
import torch
from data_modeling.src.data_utils import get_dataset
import time

def run(args):
    before_train_start_time = time.perf_counter()
    loss_time = []
    ini_time_s = time.perf_counter()

    train_dataset, test_dataset, user_groups = get_dataset(args)
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - ini_time_s
    #print(user_groups[0])
    for i in range(args.num_users):
        filename = "cnn_clients_dataidx_" + str(i) + ".txt"
        f = open(filename,"w")
        #for j in range(len(user_groups[i])):
        to_w = str(list(user_groups[i])).replace('[', '').replace(']', '').replace('.', '')
        to_w = to_w.replace("'", '').replace(',', '') + '\n'
        f.write(to_w)
        f.close()
    lr_trainer = CNNTrainer(args, train_dataset, test_dataset, user_groups[args.rank], args.rank)

    before_train_end_time = time.perf_counter()

    for rnd in range(args.rounds):
        print("round: ", rnd)
        update_flag = lr_trainer.is_update()
        print(update_flag)
        if update_flag:
            lr_trainer.one_local_round(rnd)
            lr_trainer.test(test_dataset)
            l_time =  time.perf_counter() - ini_time_s
        else:
            lr_trainer.set_loss_a()
            print("not participate in this round")
            l_time = 0
        loss_time.append(l_time)
    lr_trainer.write(load_data_time)
    end_train_time =  time.perf_counter()
    total_time = end_train_time - before_train_start_time
    print(f"rank--{args.rank}--rounds--{args.rounds}--epoch--{args.epoch}cnn total time: {total_time}")
    pa  = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/l_result/temp/cnn"
    with open(f"{pa}/loss_time1.txt", 'w') as train_los_0:
        train_los_0.write(str(loss_time))
if __name__ == '__main__':
    args = args_parser()
    args.rank = 0
    args.model = "cnn"
    #args.server_address="10.254.19.19:12345"
    #args.trainer_address="10.254.19.25:12346"
    args.server_address="127.0.0.1:12345"
    #args.trainer_address="10.254.19.25:12346"
    args.lr = 0.05
    processes = []
    run(args)
