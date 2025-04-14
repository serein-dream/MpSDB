import os
import sys
import time
import pickle
import oss2
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.mlp_trainer import MLPTrainer
from data_modeling.src.data_utils import only_get_dataset
from data_modeling.data_loader import MysqlDataSet
def load_data(args):
    cols_list = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','y']
    mysql_dataset_train = MysqlDataSet(f"database_{args.rank+1}", "cancer", cols_list, "ip")
    mysql_dataset_eval = MysqlDataSet("test_data", "cancer", cols_list, "ip")
    return mysql_dataset_train, mysql_dataset_eval
def wait_for_file(bucket, file_name):
    while not bucket.object_exists(file_name):
        time.sleep(0.1)
def get_object_from_bucket(bucket, file_name):
    return pickle.loads(bucket.get_object(file_name).read())
def update_global_params(lr_trainer, bucket):
    last_global_params_dumps = get_object_from_bucket(bucket, 'global_params.txt')
    last_global_params = pickle.loads(last_global_params_dumps)
    init_params_tensor = lr_trainer.enc_dec(last_global_params)
    lr_trainer.update_params(init_params_tensor)
def handle_participation(lr_trainer, args, bucket, rnd, selection_inf):
    acc_name = f"accum_count_this_client_{args.rank}.txt"
    accum_count_this_client_dumps = pickle.dumps(args.sample_num)
    bucket.put_object(acc_name, accum_count_this_client_dumps)
    lr_trainer.one_round_serverless(rnd)
def handle_non_participation(lr_trainer, args, bucket):
    lr_trainer.set_loss_a()
    average_done = False
    while not average_done:
        average_done = bucket.object_exists('average_done.txt')
        if not average_done:
            time.sleep(0.1)
def send_timestamp(bucket, args):
    cur_time = time.time()
    cur_time_dumps = pickle.dumps(cur_time)
    cur_name = f"live_{args.rank}.txt"
    bucket.put_object(cur_name, cur_time_dumps)
def run(args):
    ini_time_s = time.perf_counter()
    load_data_time_s = time.perf_counter()
    mysql_dataset_train, mysql_dataset_eval = load_data(args)
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - load_data_time_s
    lr_trainer = MLPTrainer(args, mysql_dataset_train, mysql_dataset_eval, args.rank)
    auth = oss2.Auth('<your-access-key-id>', '<your-access-key-secret>')
    bucket = oss2.Bucket(auth, "<your-bucket-endpoint>", "<your-bucket-name>")
    
    wait_for_file(bucket, 'selection.txt')
    ini_time_e = time.perf_counter()
    ini_time = ini_time_e - ini_time_s
    for rnd in range(args.rounds):
        wait_for_file(bucket, 'init_list_done.txt')
        selection_inf = get_object_from_bucket(bucket, 'selection.txt')
        count_dict = get_object_from_bucket(bucket, 'count_dict.txt')
        count_dict[args.rank] = args.sample_num
        if rnd != selection_inf[0][0]:
            comp_st = time.perf_counter()
        if rnd > 0:
            update_global_params(lr_trainer, bucket)
        if selection_inf[1][args.rank] == 1:
            handle_participation(lr_trainer, args, bucket, rnd)
        else:
            handle_non_participation(lr_trainer, args, bucket)
        send_timestamp(bucket, args)
    update_global_params(lr_trainer, bucket)
    end_time = time.perf_counter()
    total_time = end_time - ini_time_s
    print(f"\nnum:{len(lr_trainer.waiting_for_average_done_time_list)}\ncomp_time:{lr_trainer.comp_time}\n\n")
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()
    print(f"time:{et - st}")