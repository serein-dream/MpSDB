from torch.multiprocessing import Process
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.logistic_trainer import LogisticTrainer
import torch
from data_modeling.data_loader import MysqlDataSet
import time
def run(args):
    before_train_start_time = time.perf_counter()
    loss_time = []
    ini_time_s = time.perf_counter()
    cols_list = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','y']
    mysql_dataset_1 = MysqlDataSet(f"database_{args.rank+1}", "cancer", cols_list)
    mysql_dataset_eval = MysqlDataSet("test_data", "cancer", cols_list)
    # mysql_dataset_1 = MysqlDataSet(f"database_1", "h_cancer_0", cols_list)
    # mysql_dataset_eval = MysqlDataSet("database_1", "h_cancer_test", cols_list)
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - ini_time_s

    lr_trainer = LogisticTrainer(args, mysql_dataset_1,mysql_dataset_eval,args.rank)

    for rnd in range(args.rounds):
        print("round: ", rnd)
        update_flag = lr_trainer.is_update()
        print(update_flag)
        if update_flag:
            lr_trainer.one_local_round(rnd)
            #lr_trainer.test()
            l_time =  time.perf_counter() - ini_time_s
        else:
            lr_trainer.set_loss_a()
            print("not participate in this round")
            l_time = 0
        # print(rnd, logistic_trainer.model.linear.bias, '\n')     
        #l_time =  time.perf_counter() - ini_time_s
        loss_time.append(l_time)
    # lr_trainer.write(load_data_time)
    # end_train_time =  time.perf_counter()
    # total_time = end_train_time - before_train_start_time
    # print(f"rank--{args.rank}--rounds--{args.rounds}--epoch--{args.epoch}LR total time: {total_time}")
    # pa  = "/home/maxvyang01/SecureDatabase/script/l_result/temp/LR"
    # with open(f"{pa}/loss_time1.txt", 'w') as train_los_0:
    #     train_los_0.write(str(loss_time))


if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    args.model = "LR"
    args.n_features=30
    args.lr = 0.002
    processes = []
    # for rank in range(2):
    #     p = Process(target=run, args=args)
    #     processes.append(p)
    #     p.start()
    # for p in processes:
    #     p.join()
    run(args)
    et = time.time()
    print(f"time:{et - st}")

# def run(arg, rank):
#     arg.rank = rank
#     cols_list = ['fixed acidity', 'volatile acidity', 'citric acid',
#                  'residual sugar', 'chlorides', 'free sulfur dioxide',
#                  'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#                  'color']
#     mysql_dataset = MysqlDataSet("database_{0}".format(rank+1), "wine_quality", cols_list)
#     lr_trainer = LogisticTrainer(arg, mysql_dataset)
#     lr_trainer.launch()
#
#
# if __name__ == '__main__':
#     args = args_parser()
#     processes = []
#     for i in range(3):
#         p = Process(target=run, args=(args, i))
#         processes.append(p)
#         p.start()
#     for p in processes:
#         p.join()
