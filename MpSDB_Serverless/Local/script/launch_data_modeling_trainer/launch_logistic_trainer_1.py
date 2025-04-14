import os
import sys
import time
import pickle
import oss2
import torch
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.logistic_trainer import LogisticTrainer
from data_modeling.data_loader import MysqlDataSet
def initialize_bucket():
    auth = oss2.Auth('<your-access-key-id>', '<your-access-key-secret>')
    bucket = oss2.Bucket(auth, '<your-bucket-endpoint>', '<your-bucket-name>')
    return bucket
def wait_for_file(bucket, file_name):
    while not bucket.object_exists(file_name):
        time.sleep(0.05)
def load_object_from_bucket(bucket, file_name):
    obj_dumps = bucket.get_object(file_name).read()
    return pickle.loads(obj_dumps)
def save_object_to_bucket(bucket, file_name, obj):
    obj_dumps = pickle.dumps(obj)
    bucket.put_object(file_name, obj_dumps)
def run(args):
    ini_time_s = time.perf_counter()
    load_data_time_s = time.perf_counter()
    cols_list = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'y']
    mysql_dataset_1 = MysqlDataSet(f"database_{args.rank + 1}", "cancer", cols_list, "<your-ip>")
    mysql_dataset_eval = MysqlDataSet("test_data", "cancer", cols_list, "<your-ip>")
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - load_data_time_s
    lr_trainer = LogisticTrainer(args, mysql_dataset_1, mysql_dataset_eval, args.rank)
    bucket = initialize_bucket()
    wait_for_file(bucket, 'selection.txt')
    ini_time_e = time.perf_counter()
    ini_time = ini_time_e - ini_time_s
    for rnd in range(args.rounds):
        wait_for_file(bucket, 'init_list_done.txt')
        selection_inf = load_object_from_bucket(bucket, 'selection.txt')
        if rnd != selection_inf[0][0]:
            continue
        if rnd > 0:
            last_global_params = load_object_from_bucket(bucket, 'global_params.txt')
            init_params_tensor = lr_trainer.enc_dec(last_global_params)
            lr_trainer.update_params(init_params_tensor)
        count_dict = load_object_from_bucket(bucket, 'count_dict.txt')
        count_dict[args.rank] = args.sample_num
        save_object_to_bucket(bucket, 'count_dict.txt', count_dict)
        if selection_inf[1][args.rank] == 1:
            acc_name = f"accum_count_this_client_{args.rank}.txt"
            save_object_to_bucket(bucket, acc_name, args.sample_num)
            lr_trainer.one_round_serverless(rnd)
            one_round_time = time.perf_counter() - (time.perf_counter() - load_data_time)
        else:
            lr_trainer.set_loss_a()
            average_done = False
            while not average_done:
                average_done = bucket.object_exists('average_done.txt')
                if not average_done:
                    time.sleep(0.05)
        l_time = time.perf_counter() - ini_time_s
        cur_time = time.time()
        save_object_to_bucket(bucket, f"live_{args.rank}.txt", cur_time)
    last_global_params = load_object_from_bucket(bucket, 'global_params.txt')
    init_params_tensor = lr_trainer.enc_dec(last_global_params)
    lr_trainer.update_params(init_params_tensor)
    end_time = time.perf_counter()
    total_time = end_time - ini_time_s
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()
    print(f"time:{et - st}")