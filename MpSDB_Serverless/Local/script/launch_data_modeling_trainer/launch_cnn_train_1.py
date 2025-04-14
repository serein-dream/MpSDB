import pathlib
import os
import sys
import time
import pickle
import numpy as np
import oss2
from conf import args_parser
from data_modeling.trainer.cnn_trainer import CNNTrainer
from data_modeling.src.data_utils import only_get_dataset
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def load_data(args):
    filename = f"PATH_TO_DATAIDX_FILE_{args.rank}.txt"
    dataidx = np.loadtxt(filename, dtype=bytes).astype(int)
    dataidx_list = list(dataidx)
    train_dataset, test_dataset = only_get_dataset(args)
    return train_dataset, test_dataset, dataidx_list
def initialize_oss_bucket():
    auth = oss2.Auth('YOUR_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_SECRET')
    bucket = oss2.Bucket(auth, "YOUR_OSS_ENDPOINT", "YOUR_BUCKET_NAME")
    return bucket
def wait_for_file(bucket, file_name):
    while not bucket.object_exists(file_name):
        time.sleep(0.05)
def update_local_params(lr_trainer, bucket):
    last_global_params_dumps = bucket.get_object('global_params.txt').read()
    last_global_params = pickle.loads(last_global_params_dumps)
    init_params_tensor = lr_trainer.enc_dec(last_global_params)
    lr_trainer.update_params(init_params_tensor)
def update_accum_count(bucket, args, lr_trainer):
    accum_count_this_client_dumps = pickle.dumps(args.sample_num)
    bucket.put_object(f"accum_count_this_client_{args.rank}.txt", accum_count_this_client_dumps)
    lr_trainer.one_round_serverless(args.rnd)
def handle_participation(lr_trainer, bucket, args, selection_inf):
    if selection_inf[1][args.rank] == 1:
        update_accum_count(bucket, args, lr_trainer)
    else:
        lr_trainer.set_loss_a()
        wait_for_file(bucket, 'average_done.txt')
def write_results(lr_trainer, load_data_time, ini_time_s, waiting_for_init_list_time_list, one_round_time_list):
    end_time = time.perf_counter()
    total_time = end_time - ini_time_s
    lr_trainer.write(load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list)
    pa = "PATH_TO_RESULT_DIR/"
    with open(f"{pa}/loss_time.txt", 'w') as train_los_0:
        train_los_0.write(str(lr_trainer.loss_time))
def run(args):
    print("Client start! Waiting for others...\n")
    ini_time_s = time.perf_counter()
    train_dataset, test_dataset, dataidx_list = load_data(args)
    load_data_time = time.perf_counter() - ini_time_s
    bucket = initialize_oss_bucket()
    init_enc_client_centroids_d = [0.9651141, 3.06628916, 2.77072469]
    lr_trainer = CNNTrainer(args, init_enc_client_centroids_d, train_dataset, test_dataset, dataidx_list, args.rank)
    wait_for_file(bucket, 'selection.txt')
    ini_time = time.perf_counter() - ini_time_s
    waiting_for_init_list_time_list = []
    one_round_time_list = []
    for rnd in range(args.rounds):
        wait_for_file(bucket, 'init_list_done.txt')
        waiting_for_init_list_s = time.perf_counter()
        selection_inf_dumps = bucket.get_object('selection.txt').read()
        count_dict_dumps = bucket.get_object('count_dict.txt').read()
        waiting_for_init_list_e = time.perf_counter()
        waiting_for_init_list_time = waiting_for_init_list_e - waiting_for_init_list_s
        waiting_for_init_list_time_list.append(waiting_for_init_list_time)
        selection_inf = pickle.loads(selection_inf_dumps)
        count_dict = pickle.loads(count_dict_dumps)
        count_dict[args.rank] = args.sample_num
        handle_participation(lr_trainer, bucket, args, selection_inf)
        one_round_time = time.perf_counter() - waiting_for_init_list_e
        one_round_time_list.append(one_round_time)
        if rnd in [10, 20, 30, 40]:
            write_results(lr_trainer, load_data_time, ini_time_s, waiting_for_init_list_time_list, one_round_time_list)
    update_local_params(lr_trainer, bucket)
    write_results(lr_trainer, load_data_time, ini_time_s, waiting_for_init_list_time_list, one_round_time_list)
if __name__ == '__main__':
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(args)
    et = time.time()
    print(f"time:{et - st}")