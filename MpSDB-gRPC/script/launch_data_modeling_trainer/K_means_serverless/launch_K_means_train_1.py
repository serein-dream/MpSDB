import torch
import time
import pickle
import tenseal as ts
import openpyxl

import oss2
from array import array
import numpy as np
import os,sys

sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from conf import args_parser
from data_modeling.trainer.K_means_trainer import KMeans
from data_modeling.data_loader import MysqlDataSet



def run(args, verbose=False, dummy=False):
    auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
    bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
    
    

    cols_list = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                 'quality']
    mysql_dataset_1 = MysqlDataSet(f"database_{args.rank+1}", "wine_quality", cols_list)
    mysql_dataset_eval = MysqlDataSet("test_data", "wine_quality", cols_list) 

    #last_global_centroids_dumps = bucket.get_object('global_params.txt').read()
    #init_centroids = pickle.loads(last_global_centroids_dumps)
    init_centroids = np.array([[6.4,0.25,0.28,4.9,0.03,29.0,98.0,0.99024,3.09,0.58,12.8],
                      [5.7,0.22,0.28,1.3,0.027,26.0,101.0,0.98948,3.35,0.38,12.5],
                      [6.8,0.475,0.33,3.95,0.047,16.0,81.0,0.98988,3.23,0.53,13.4]])
    kmeans_trainer = KMeans(init_centroids,args, mysql_dataset_1, mysql_dataset_eval ,args.rank)
    not_change_number = 0
    #return
    for rnd in range(0,args.rounds): #round        
        print("round: ", rnd)
        update_flag, to_stop = kmeans_trainer.is_update_kmeans()
        if to_stop == True:
            print("train have finished!\n")
            break
        #print(update_flag)
        if update_flag:
            not_change_number = kmeans_trainer.one_local_round()
            # if rnd>0 and rnd%1==0:
            #     kmeans_trainer.test()
        else:
            #lr_trainer.set_loss_a()
            print("not participate in this round")
            #l_time = 0
        '''
        if not_change_number == 3:
            print(f"rnd: {rnd}  ,train done!")
            kmeans_trainer.test()
            break
        '''
        print()
    print(f"rnd: {rnd}  ,train done!")
    # kmeans_trainer.test()
    # kmeans_trainer.write_kmeans()
        #loss_time.append(l_time)
    #kmeans_trainer.write(load_data_time)
    #end_train_time =  time.perf_counter()
    #total_time = end_train_time - before_train_start_time
    #print(f"rank--{args.rank}--rounds--{args.rounds}--epoch--{args.epoch}cnn total time: {total_time}")
    #pa  = "/home/maxvyang01/SecureDatabase/script/l_result/temp/cnn"
    #with open(f"{pa}/loss_time1.txt", 'w') as train_los_0:
        #train_los_0.write(str(loss_time))

if __name__ == '__main__':    
    st = time.time()
    args = args_parser()
    args.rank = 0
    run(
        args,
        dummy=True,
        verbose=False,
    )
    et = time.time()
    print(f"time:{et - st}")