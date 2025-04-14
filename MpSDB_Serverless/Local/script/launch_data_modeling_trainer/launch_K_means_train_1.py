import torch
import time as t
import pickle
import oss2
import numpy as np
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from conf import args_parser
from data_modeling.trainer.K_means_trainer import KMeans
from data_modeling.data_loader import MysqlDataSet
def load_oss_bucket():
    auth = oss2.Auth('<your-access-key-id>', '<your-access-key-secret>')
    bucket = oss2.Bucket(auth, "ADD_by_yourself", "<your-bucket-name>")
    return bucket
def wait_for_file(bucket, file_name):
    while not bucket.object_exists(file_name):
        t.sleep(0.05)
def initialize_kmeans_trainer(bucket, args, rank):
    init_enc_client_centroids_d = bucket.get_object('global_params.txt').read()
    cols_list = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                 'quality']
    mysql_dataset_1 = MysqlDataSet(f"database_{rank+1}", "wine_quality", cols_list, "ip")
    mysql_dataset_eval = MysqlDataSet("test_data", "wine_quality", cols_list, "ip")
    kmeans_trainer = KMeans(init_enc_client_centroids_d, args, mysql_dataset_1, mysql_dataset_eval, rank)
    return kmeans_trainer
def check_over_stop(bucket):
    return bucket.object_exists("over_stop.txt")
def handle_participation(kmeans_trainer, bucket, selection_inf, rank):
    not_change_number = kmeans_trainer.one_local_round()
    if not_change_number == 3:
        over_tape = pickle.dumps("over")
        bucket.put_object("over_stop.txt", over_tape)
    return not_change_number
def handle_non_participation(bucket):
    wait_for_file(bucket, 'average_done.txt')
def run(args, verbose=False, dummy=False):
    rank = args.rank
    bucket = load_oss_bucket()
    kmeans_trainer = initialize_kmeans_trainer(bucket, args, rank)
    not_change_number = 0
    wait_for_file(bucket, 'init_list_done.txt')
    for rnd in range(args.rounds):
        wait_for_file(bucket, 'init_list_done.txt')
        if check_over_stop(bucket):
            break
        selection_inf_dumps = bucket.get_object('selection.txt').read()
        selection_inf = pickle.loads(selection_inf_dumps)
        if rnd != selection_inf[0][0]:
            continue
        if selection_inf[1][rank] == 1:
            not_change_number = handle_participation(kmeans_trainer, bucket, selection_inf, rank)
        else:
            handle_non_participation(bucket)
        if check_over_stop(bucket):
            break
        t.sleep(0.05)
if __name__ == '__main__':
    st = t.time()
    args = args_parser()
    args.rank = 0
    run(args, dummy=True, verbose=False)
    et = t.time()
    print(f"time:{et - st}")