'''
from conf import args_parser
import torch.nn as nn
from data_modeling.model import DNN
import sys
import torch.optim as optim
#from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
'''
import numpy as np
import random
#from utils_stats import plot_progress
from conf import args_parser
from torch.utils.data import DataLoader
import torch
import sys,os
from data_modeling.trainer.base_trainer import BaseTrainer
import time as t
import datetime
import matplotlib.pyplot as plt
sys.path.append("../../../")


def randomly_init_centroid(min_value, max_value, n_dims, repeats=1):
    if repeats == 1:
        return min_value + (max_value - min_value) * np.random.rand(n_dims)
    else:
        return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)
    
class KMeans(BaseTrainer):
    def __init__(self, client_centroids,args, dataset, test_dataset ,rank):
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        super().__init__(args, dataset, test_dataset,client_centroids)
        self.rank = rank
        self.clients_counts = np.zeros(self.client_centroids.shape[0])
        self.client_train_data = dataset
        self.test_data = test_dataset
        self.epoch_lr = args.lr#本地学习率
        self.n_clusters=args.n_clusters
        self.n_dims=args.n_dims
        self.clients_updates = np.zeros((self.n_clusters, self.n_dims))  
        self.train_davies_bouldin_list_0 = []
        self.train_davies_bouldin_list_1 = []
        self.train_davies_bouldin_list_2 = []  
        self.test_davies_bouldin_list_0 = []
        self.test_davies_bouldin_list_1 = []
        self.test_davies_bouldin_list_2 = []

    def __compute_step_for_client(self, train_data):
        centroids = self.client_centroids
        #print(f"client_data type:{type(client_data)}  shape : {client_data.shape}")  
        #client_data type:<class 'numpy.ndarray'>  shape : (10, 1)
        #检查簇心和数据维度
        #print(f"datat:\n{client_data}\n\ncentroids:\n{centroids}")
        #raise

        #print(f"client_data:\n{client_data}")
        #print(f"centroids:\n{centroids}")
        #print(f"here   np.expand_dims(client_data, axis=1)\n\n{np.expand_dims(client_data, axis=1)}\n\nnp.expand_dims(centroids, axis=0)\n\n{np.expand_dims(centroids, axis=0)}")
        #print(f"here   client_data:\n\n{client_data.shape}\n\centroids:\n\n{centroids.shape}\n\n")
        
        differences = np.expand_dims(train_data, axis=1) - np.expand_dims(centroids, axis=0)
        sq_dist = np.sum(np.square(differences), axis=2)
        labels = np.argmin(sq_dist, axis=1)
        #print(f"here   differences:\n\n{differences.shape}\n\nsq_dist:\n\n{sq_dist.shape}\n\n")
        #print(f"here   labels\n\n{labels}\n\n{sq_dist}")

        # update centroids
        centroid_updates = np.zeros_like(centroids)
        #print(f"here   centroids.shape[0]\n\n{centroids.shape}\n\n{centroids}")
        counts = np.zeros(centroids.shape[0],dtype=int)
        #print(f"here   centroids\n\n{centroids.shape}")
        for i in range(centroids.shape[0]):
            mask = np.equal(labels, i)
            counts[i] = np.sum(mask)
            if counts[i] > 0:
                centroid_updates[i] = np.sum(train_data[mask].numpy()  - centroids[i], axis=0)
            #print(f"i:  {i}   counts:  {counts}")
        #print(f"counts: \n\n{type(counts[0])}\n\n{counts}")
        return centroid_updates, counts


    def __davies_bouldin(self, x, labels, verbose=False):

        NUM_CLUSTERS = self.client_centroids.shape[0]
        distances = np.sqrt(np.sum(np.square(x - self.client_centroids[labels]), axis=1))
        # centroid distances
        centroid_dist_matrix = np.expand_dims(self.client_centroids, axis=0) - np.expand_dims(self.client_centroids, axis=1)
        centroid_dist_matrix = np.sqrt(np.sum(np.square(centroid_dist_matrix), axis=2))
        centroid_dist_matrix[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
        # print(centroid_dist_matrix)

        # intra cluster dist
        intra_dist = np.zeros(NUM_CLUSTERS)
        print(f"labels shape: {labels.shape}  labels type: {type(labels)}  labels[0] type: {type(labels[0])}  labels:{labels}  distances shape:  {distances.shape}")
        for i in range(NUM_CLUSTERS):
            intra_dist[i] = np.mean(distances[i == labels])

        s_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)
        d_i = np.nanmax(s_ij / centroid_dist_matrix, axis=1)
        db_score = np.nanmean(d_i)
        if verbose:
            print("centroid_min_dist", np.amin(centroid_dist_matrix, axis=1))
            print("intra_dist", intra_dist)
        return db_score

    def __predict(self,x):
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))#elf.n_clusters = 5
        for i in range(self.n_clusters):#self.n_clusters = 5
            t = np.square(x - self.client_centroids[i, :])
            #print(f"x:\n{x} \n centroids[{i}, :]:\n{self.client_centroids[i, :]}")
            sq_dist[:, i] = np.sum(np.square(x - self.client_centroids[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels
        
    def __evaluate(self, splits={'train','test'}, use_metric="davies_bouldin", federated=True, verbose=False):
        scores = {}
        '''
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
        data_d = bucket.get_object('data.txt').read()
        data = pickle.loads(data_d) 
        '''
        x = {}
        x['train'] = self.test_dataset['train']
        x['test'] = self.test_dataset['test']    
        for split in splits:
            x[split] = np.concatenate(x[split], axis=0)
            labels = self.__predict(x[split])
            if "davies_bouldin" == use_metric:
                score = self.__davies_bouldin(x[split], labels, verbose)
            scores[split] = score
        return scores


    def one_local_round(self):
        client_step_centroids = self.client_centroids
        #tensor转numpy
        #client_step_centroids = client_step_centroids.detach().numpy()
        for e in range(self.epoch):
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
            for batch in data_loader:
                train_data, train_y = batch
                client_updates_sum, client_counts = self.__compute_step_for_client(train_data)
                #print(f"here!!!    rank_{self.rank}\n\n{client_counts}")
                
                interim_updates = client_updates_sum / np.expand_dims(np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
                if self.epoch_lr is not None:
                    interim_updates = self.epoch_lr * interim_updates
                #得到局部簇心的更新信息
                client_step_centroids = client_step_centroids + interim_updates               
        clients_local_centroids_expanded = client_step_centroids * np.expand_dims(client_counts, axis=1)
        '''
        print(f"before aggre : ",end="")
        for i in range(len(clients_local_centroids_expanded)):
            print(clients_local_centroids_expanded[i],end="  ")
        print()
        '''
        self.average_params_kmeans(clients_local_centroids_expanded,client_counts)

        print("centroids:      ",end="")
        for i in range(len(self.client_centroids)):
            print(self.client_centroids[i],end="  ")
        print()
        return self.not_change_number

    def test(self):
        results = {}
        results['train'] = []
        results['test'] = []    
        stats = {
            'train': {'avg': [], 'std': []},
            'test': {'avg': [], 'std': []},
        }
        scores = self.__evaluate()
        for key, value in scores.items():
            results[key].append(value) 
        results_avg = {}
        for key, value in results.items():
            score_mean = np.around(np.mean(value), decimals=3)
            if score_mean > 5:
                results_avg[key] = 5
            else:
                results_avg[key] = score_mean
            
        '''
        for key, value in results.items():
            score_mean = np.around(np.mean(value), decimals=3)
            score_std = np.around(np.std(value), decimals=3)
            results_avg[key] = (score_mean, score_std)
            stats[key]['avg'].append(score_mean)
            stats[key]['std'].append(score_std)
        print(f"use_metric: davies_bouldin  ,  {results_avg}\ncentroids:\n{self.client_centroids}\n\n")    
        if results_avg['train'][0]>5:
            results_avg['train'][0] = 5
        if results_avg['test'][0]>5:
            results_avg['test'][0] = 5
        '''
        if self.rank == 0:
            self.train_davies_bouldin_list_0.append(results_avg['train'])
            self.test_davies_bouldin_list_0.append(results_avg['test'])
            #print(f"hhhh\n{results_avg['train']}\n{results_avg['train']}")
        elif self.rank == 1:
            self.train_davies_bouldin_list_1.append(results_avg['train'])
            self.test_davies_bouldin_list_1.append(results_avg['test'])
        elif self.rank == 2:
            self.train_davies_bouldin_list_2.append(results_avg['train'])
            self.test_davies_bouldin_list_2.append(results_avg['test'])
        
    def write_kmeans(self):       
        #print(f"rank: {self.rank}\nself.train_davies_bouldin_list_0:\n{self.train_davies_bouldin_list_0}\nself.test_davies_bouldin_list_0\n{self.test_davies_bouldin_list_0}")        
        pa = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_modeling_trainer/K_means_serverless/result"
        if self.rank == 0:
            plt.figure()
            plt.plot(range(len(self.train_davies_bouldin_list_0)), self.train_davies_bouldin_list_0,color='blue', linestyle="solid", label="Train davies_bouldin")
            plt.plot(range(len(self.test_davies_bouldin_list_0)), self.test_davies_bouldin_list_0,color='green', linestyle="solid", label="Test davies_bouldin")
            plt.xlabel('rounds')
            plt.savefig('{}/rank_{}_DBI_n_clusters_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.n_clusters,self.args.lr, self.args.rounds, self.args.epoch))
           
            print(f"rank: {self.rank}\nself.train_davies_bouldin_list_0:\n{self.train_davies_bouldin_list_0[-1]}\nself.test_davies_bouldin_list_0\n{self.test_davies_bouldin_list_0[-1]}")  
            print(f"\nchange:")
            for i in range(len(self.change_list)):
                print(self.change_list[i])
            

        if self.rank == 1:
            plt.figure()
            plt.plot(range(len(self.train_davies_bouldin_list_1)), self.train_davies_bouldin_list_1,color='blue', linestyle="solid", label="Train davies_bouldin")
            plt.plot(range(len(self.test_davies_bouldin_list_1)), self.test_davies_bouldin_list_1,color='green', linestyle="solid", label="Test davies_bouldin")
            plt.xlabel('rounds')
            plt.savefig('{}/rank_{}_DBI_n_clusters_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.n_clusters,self.args.lr, self.args.rounds, self.args.epoch))
            print(f"rank: {self.rank}\nself.train_davies_bouldin_list_1:\n{self.train_davies_bouldin_list_1[-1]}\nself.test_davies_bouldin_list_1\n{self.test_davies_bouldin_list_1[-1]}")  

        if self.rank == 2:
            plt.figure()
            plt.plot(range(len(self.train_davies_bouldin_list_2)), self.train_davies_bouldin_list_2,color='blue', linestyle="solid", label="Train davies_bouldin")
            plt.plot(range(len(self.test_davies_bouldin_list_2)), self.test_davies_bouldin_list_2,color='green', linestyle="solid", label="Test davies_bouldin")
            plt.xlabel('rounds')
            plt.savefig('{}/rank_{}_DBI_n_clusters_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.n_clusters,self.args.lr, self.args.rounds, self.args.epoch))
            print(f"rank: {self.rank}\nself.train_davies_bouldin_list_2:\n{self.train_davies_bouldin_list_2[-1]}\nself.test_davies_bouldin_list_2\n{self.test_davies_bouldin_list_2[-1]}")  

    def write(self,begin_time,load_time):
        
        end_time = t.perf_counter()
        total_time = end_time - self.start_time
        pa = "/home/maxvyang01/SecureDatabase/script/l_result/temp/linear"
        if self.rank == 0:
            fileloader = self.filetime+"Linear__wine_quity__lr_"+ str(self.args.lr) +"__epoch_"+ str(self.args.epoch) +"__rounds_"+ str(self.args.rounds)+"__batch_size__"+str(self.args.batch_size)
            pat = "/home/maxvyang01/SecureDatabase/script/l_result/"+fileloader
            with open(f"{pa}/train_loss_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_loss_list_0))
            with open(f"{pa}/train_RMSE_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_RMSE_list_0))
            with open(f"{pa}/test_set_RMSE_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.test_set_RMSE_list_0))
            with open(f"{pa}/pat.txt", 'w') as train_los_0:
                train_los_0.write(pat)
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\nbegin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

        elif self.rank == 1:
            train_loss_list = self.train_loss_list_1
            train_RMSE_list = self.train_RMSE_list_1
            test_set_RMSE_list = self.test_set_RMSE_list_1
            with open(f"{pa}/train_loss_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_loss_list_1))
            with open(f"{pa}/train_RMSE_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_RMSE_list_1))
            with open(f"{pa}/test_set_RMSE_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.test_set_RMSE_list_1))
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\nbegin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

        elif self.rank == 2:
            train_loss_list = self.train_loss_list_2
            train_RMSE_list = self.train_RMSE_list_2
            test_set_RMSE_list = self.test_set_RMSE_list_2
            with open(f"{pa}/train_loss_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_loss_list_2))
            with open(f"{pa}/train_RMSE_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_RMSE_list_2))
            with open(f"{pa}/test_set_RMSE_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.test_set_RMSE_list_2))
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\nbegin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")
        '''
        #画图
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list)
        plt.xlabel('rounds')
        plt.ylabel('Train loss')
        plt.savefig('{}/rank_{}_train_loss_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))
        
        plt.figure()
        plt.plot(range(len(train_RMSE_list)), train_RMSE_list)
        plt.xlabel('rounds')
        plt.ylabel('train_RMSE_list')
        plt.savefig('{}/rank_{}_train_RMSE_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))

        plt.figure()
        plt.plot(range(len(test_set_RMSE_list)), test_set_RMSE_list)
        plt.xlabel('rounds')
        plt.ylabel('test_set_RMSE_list')
        plt.savefig('{}/rank_{}_test_set_RMSE_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))
        '''    
if __name__ == '__main__':
    arg = args_parser()
    # lr_trainer = LogisticTrainer(args)
    # logistic_trainer.one_round()






