import os
import sys
import time
import pickle
import torch
import oss2
import pandas as pd
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.linear_trainer import LinearTrainer
from data_modeling.data_loader import MysqlDataSet
def get_data_from_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    data_array = np.array(data).astype('float64')
    tensor_data = torch.from_numpy(data_array).float()
    return tensor_data
def initialize_trainer_and_datasets(args):
    cols_list = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                 'quality']
    mysql_dataset_1 = MysqlDataSet(f"database_{args.rank+1}", "wine_quality", cols_list, "ip")
    mysql_dataset_eval = MysqlDataSet("test_data", "wine_quality", cols_list, "ip")
    init_enc_client_centroids_d = [0.9651141, 3.06628916, 2.77072469]
    lr_trainer = LinearTrainer(args, init_enc_client_centroids_d, mysql_dataset_1, mysql_dataset_eval, args.rank)
    return lr_trainer
def wait_for_file(bucket, file_name):
    while not bucket.object_exists(file_name):
        time.sleep(0.05)
def update_trainer_with_global_params(lr_trainer, bucket):
    last_global_params_dumps = bucket.get_object('global_params.txt').read()
    last_global_params = pickle.loads(last_global_params_dumps)
    init_params_tensor = lr_trainer.enc_dec(last_global_params)
    lr_trainer.update_params(init_params_tensor)
def update_count_dict(bucket, args):
    count_dict_dumps = bucket.get_object('count_dict.txt').read()
    count_dict = pickle.loads(count_dict_dumps)
    count_dict[args.rank] = args.sample_num
    return count_dict
def handle_participation(lr_trainer, bucket, args, rnd, waiting_for_init_list_e, one_round_time_list):
    acc_name = f"accum_count_this_client_{args.rank}.txt"
    accum_count_this_client_dumps = pickle.dumps(args.sample_num)
    bucket.put_object(acc_name, accum_count_this_client_dumps)
    lr_trainer.one_round_serverless(rnd)
    one_round_time = time.perf_counter() - waiting_for_init_list_e
    one_round_time_list.append(one_round_time)
def handle_non_participation(lr_trainer, bucket, args, rnd):
    lr_trainer.set_loss_a()
    average_done = False
    while not average_done:
        average_done = bucket.object_exists('average_done.txt')
        if not average_done:
            time.sleep(0.05)
def write_results(lr_trainer, load_data_time, ini_time_s, waiting_for_init_list_time_list, one_round_time_list, rnd):
    if rnd in [10, 20, 30, 40]:
        end_time = time.perf_counter()
        total_time = end_time - ini_time_s
        lr_trainer.write(load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list)
def send_timestamp(bucket, args):
    cur_time = time.time()
    cur_time_dumps = pickle.dumps(cur_time)
    cur_name = f"live_{args.rank}.txt"
    bucket.put_object(cur_name, cur_time_dumps)
def run(args):
    ini_time_s = time.perf_counter()
    load_data_time_s = time.perf_counter()
    lr_trainer = initialize_trainer_and_datasets(args)
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - load_data_time_s
    auth = oss2.Auth('<your-access-key-id>', '<your-access-key-secret>')
    bucket = oss2.Bucket(auth, "<your-bucket-url>", "<your-bucket-name>")
    wait_for_file(bucket, 'selection.txt')
    ini_time_e = time.perf_counter()
    ini_time = ini_time_e - ini_time_s
    one_round_time_list = []
    waiting_for_init_list_time_list = []
    loss_time = []
    for rnd in range(args.rounds):
        wait_for_file(bucket, 'init_list_done.txt')
        selection_inf_dumps = bucket.get_object('selection.txt').read()
        selection_inf = pickle.loads(selection_inf_dumps)
        if rnd > 0:
            update_trainer_with_global_params(lr_trainer, bucket)
        count_dict = update_count_dict(bucket, args)
        waiting_for_init_list_e = time.perf_counter()
        waiting_for_init_list_time = waiting_for_init_list_e - ini_time_s
        waiting_for_init_list_time_list.append(waiting_for_init_list_time)
        if selection_inf[1][args.rank] == 1:
            handle_participation(lr_trainer, bucket, args, rnd, waiting_for_init_list_e, one_round_time_list)
        else:
            handle_non_participation(lr_trainer, bucket, args, rnd)
        l_time = time.perf_counter() - ini_time_s
        loss_time.append(l_time)
        send_timestamp(bucket, args)
        write_results(lr_trainer, load_data_time, ini_time_s, waiting_for_init_list_time_list, one_round_time_list, rnd)
    update_trainer_with_global_params(lr_trainer, bucket)
    end_time = time.perf_counter()
    total_time = end_time - ini_time_s
    lr_trainer.write(load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list)
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()
    print(f"time:{et - st}")