import math
import numpy as np
import random
#from utils_stats import plot_progress
from conf import args_parser
import torch
import sys,os
from .base_trainer import BaseTrainer
import time as t
import datetime
import matplotlib.pyplot as plt
from .base_trainer import BaseTrainer
from transmission.pickle import aggregate_server_pb2, aggregate_server_pb2_grpc
import pickle
from tqdm import tqdm
#sys.path.append("../../../")
import oss2,time
    
class KmeansTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, dataset, test_dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.test_dataset = test_dataset
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        self.rank = args.rank
        self.args = args
        self.epoch_lr = args.kmeans_lr#本地学习率
        self.n_clusters=args.n_clusters
        self.clients_counts = np.zeros(self.n_clusters)
        self.n_dims = n_f
        self.clients_updates = np.zeros((self.n_clusters, self.n_dims))  
        self.client_centroids = np.zeros((self.n_clusters, self.n_dims))
        self.train_davies_bouldin_list_0 = []
        self.train_davies_bouldin_list_1 = []
        self.train_davies_bouldin_list_2 = []  
        self.test_davies_bouldin_list_0 = []
        self.test_davies_bouldin_list_1 = []
        self.test_davies_bouldin_list_2 = []
        self.not_change_epoch = 0
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        self.bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")

    def __randCent(self,k):
        for i in range(k):
            index = int(np.random.uniform(0,self.clients_counts))
            self.client_centroids[i,:] = self.dataSet[index,:]
        return

    def __compute_step_for_client(self,client_data):
        centroids = self.client_centroids
        #检查簇心和数据维度
        #print(f"datat:\n{client_data}\n\ncentroids:\n{centroids}")
        #raise

        #print(f"client_data:\n{client_data}")
        #print(f"centroids:\n{centroids}")
        #print(f"here   np.expand_dims(client_data, axis=1)\n\n{np.expand_dims(client_data, axis=1)}\n\nnp.expand_dims(centroids, axis=0)\n\n{np.expand_dims(centroids, axis=0)}")
        #print(f"here   client_data:\n\n{client_data.shape}\n\centroids:\n\n{centroids.shape}\n\n")
        
        differences = np.expand_dims(client_data, axis=1) - np.expand_dims(centroids, axis=0)
        #print(f"differences shape: {differences.shape}")  #differences shape: (5197, 3, 6)
        local_sq_dist = np.sum(np.square(differences), axis=2)
        #print(f"local_sq_dist shape: {local_sq_dist.shape}")  #local_sq_dist shape: (5197, 3)
        #print("111")

        enc_local_sq_dist_msg = self.client.encry_np(local_sq_dist)
        bottom_to_server_dict = {}
        bottom_to_server_dict["enc_local_sq_dist_msg"] = enc_local_sq_dist_msg
        bottom_to_server_dict["not_change_epoch"] = self.not_change_epoch
        bottom_to_server_dict["args"] = self.args
        bottom_to_server_dict_d = pickle.dumps(bottom_to_server_dict)
        name = "vfl_from_bottom/rank_"+ str(self.args.rank) +".txt"
        self.bucket.put_object(name, bottom_to_server_dict_d)

        name = "to_bottom_grad/rank_"+ str(self.args.rank) +".txt"       
        while self.bucket.object_exists(name) == False:
            time.sleep(0.01)
        get_sq_dist_dict = pickle.loads(self.bucket.get_object(name).read())
        self.bucket.delete_object(name) 
        self.not_change_epoch = get_sq_dist_dict["not_change_epoch"]
        global_sq_dist = self.client.decry_np(get_sq_dist_dict["enc_global_sq_dist_msg"])
        t = int(len(global_sq_dist) / self.args.n_clusters)
        global_sq_dist = np.reshape(np.array(global_sq_dist),(t, self.args.n_clusters))

        #print(f"global_sq_dist shape: {global_sq_dist.shape}")  #global_sq_dist shape: (5197, 3)

        labels = np.argmin(global_sq_dist, axis=1) 

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
                centroid_updates[i] = np.sum(client_data[mask] - centroids[i], axis=0)
            #print(f"i:  {i}   counts:  {counts}")
        #print(f"counts: \n\n{type(counts[0])}\n\n{counts}")
        return centroid_updates, counts


    def __davies_bouldin(self, x, labels, verbose=False):

        NUM_CLUSTERS = self.client_centroids.shape[0]
        distances = np.sqrt(np.sum(np.square(x - self.client_centroids[labels]), axis=1))   

        enc_local_sq_dist_msg = self.client.encry_np(distances)
        bottom_to_server_dict = {}
        bottom_to_server_dict["enc_local_sq_dist_msg"] = enc_local_sq_dist_msg
        bottom_to_server_dict["not_change_epoch"] = self.not_change_epoch
        bottom_to_server_dict["args"] = self.args
        bottom_to_server_dict_d = pickle.dumps(bottom_to_server_dict)
        name = "vfl_from_bottom/rank_"+ str(self.args.rank) +".txt"
        self.bucket.put_object(name, bottom_to_server_dict_d)

        name = "to_bottom_grad/rank_"+ str(self.args.rank) +".txt"      
        while self.bucket.object_exists(name) == False:
            time.sleep(0.01)
        get_sq_dist_dict = pickle.loads(self.bucket.get_object(name).read())
        self.bucket.delete_object(name) 
        self.not_change_epoch = get_sq_dist_dict["not_change_epoch"]
        distances = self.client.decry_np(get_sq_dist_dict["enc_global_sq_dist_msg"])
      

        # centroid distances
        centroid_dist_matrix = np.expand_dims(self.client_centroids, axis=0) - np.expand_dims(self.client_centroids, axis=1)
        centroid_dist_matrix = np.sqrt(np.sum(np.square(centroid_dist_matrix), axis=2))
        centroid_dist_matrix[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
        # print(centroid_dist_matrix)

        # intra cluster dist
        intra_dist = np.zeros(NUM_CLUSTERS)
        distances = np.array(distances)
        for i in range(NUM_CLUSTERS):
            intra_dist[i] = np.mean(distances[i == labels])

        s_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)
        d_i = np.nanmax(s_ij / centroid_dist_matrix, axis=1)
        db_score = np.nanmean(d_i)
        if verbose:
            print("centroid_min_dist", np.amin(centroid_dist_matrix, axis=1))
            print("intra_dist", intra_dist)
        return db_score

        
    def __evaluate(self, splits=['train','test'], use_metric="davies_bouldin", federated=True, verbose=False):
        scores = {}
        x = {}
        x['train'] = self.dataset
        x['test'] = self.test_dataset
        clients_num_d = pickle.dumps(self.args.num_users)
        self.bucket.put_object("num_clients.txt", clients_num_d)        
        for i in range(2):
            split = splits[i]
            differences = np.expand_dims(x[split], axis=1) - np.expand_dims(self.client_centroids, axis=0)
            local_sq_dist = np.sum(np.square(differences), axis=2)

            enc_local_sq_dist_msg = self.client.encry_np(local_sq_dist)
            bottom_to_server_dict = {}
            bottom_to_server_dict["enc_local_sq_dist_msg"] = enc_local_sq_dist_msg
            bottom_to_server_dict["not_change_epoch"] = self.not_change_epoch
            bottom_to_server_dict["args"] = self.args
            bottom_to_server_dict_d = pickle.dumps(bottom_to_server_dict)
            name = "vfl_from_bottom/rank_"+ str(self.args.rank) +".txt"
            self.bucket.put_object(name, bottom_to_server_dict_d)

            name = "to_bottom_grad/rank_"+ str(self.args.rank) +".txt"      
            while self.bucket.object_exists(name) == False:
                time.sleep(0.01)
            get_sq_dist_dict = pickle.loads(self.bucket.get_object(name).read())
            self.bucket.delete_object(name) 
            self.not_change_epoch = get_sq_dist_dict["not_change_epoch"]
            global_sq_dist = self.client.decry_np(get_sq_dist_dict["enc_global_sq_dist_msg"])
            t = int(len(global_sq_dist) / self.args.n_clusters)
            global_sq_dist = np.reshape(np.array(global_sq_dist),(t, self.args.n_clusters))
            labels = np.argmin(global_sq_dist, axis=1) 

            if "davies_bouldin" == use_metric:
                score = self.__davies_bouldin(x[split], labels, verbose)
            scores[split] = score
        return scores


    def one_local_round(self):
        #client_step_centroids = client_step_centroids.detach().numpy()
        '''
        print(f"centroids shape: {self.client_centroids.shape}\ncentroids:      ",end="")
        for i in range(len(self.client_centroids)):
            print(self.client_centroids[i],end="  ")
        print("\n\n\n")
        '''
        change = False
        batch_round = math.ceil(self.dataset.shape[0]/self.args.batch_size)
        for batch_idx in range(batch_round):
            train_data = self.dataset[batch_idx*self.args.batch_size : min(self.dataset.shape[0], (batch_idx+1)*self.args.batch_size), :]
            client_updates_sum, client_counts = self.__compute_step_for_client(train_data)
            #print(f"here!!!    rank_{self.rank}\nclient_updates_sum:\n{client_updates_sum}")
            
            interim_updates = client_updates_sum / np.expand_dims(np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
            if self.epoch_lr is not None:
                interim_updates = self.epoch_lr * interim_updates
            #得到局部簇心的更新信息
            for i in interim_updates:
                for j in i:
                    if j>0.001:
                        change = True
                        break
                if change == True:
                    break
            if change == False:
                self.not_change_epoch += 1
            self.client_centroids = self.client_centroids + interim_updates               
            #clients_client_centroids_expanded = client_step_centroids * np.expand_dims(client_counts, axis=1)
            '''
            print(f"before aggre : ",end="")
            for i in range(len(clients_client_centroids_expanded)):
                print(clients_client_centroids_expanded[i],end="  ")
            print()
            '''
            #self.average_params_kmeans(clients_local_centroids_expanded,client_counts)

            print("centroids:      ",end="")
            for i in range(len(self.client_centroids)):
                print(self.client_centroids[i],end="  ")
            print("\n\n\n")

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
            #print(f"results_avg:{results_avg}")
            self.train_davies_bouldin_list_0.append(results_avg['train'])
            self.test_davies_bouldin_list_0.append(results_avg['test']) 
            #print(f"self.train_davies_bouldin_list_0:{self.train_davies_bouldin_list_0[-1]}    rank: {self.rank}")


        
    def write_kmeans(self):       
        #print(f"rank: {self.rank}\nself.train_davies_bouldin_list_0:\n{self.train_davies_bouldin_list_0}\nself.test_davies_bouldin_list_0\n{self.test_davies_bouldin_list_0}")        
        pa = "ADD_by_yourself/FaaS_FL2023/VFL_faas/model_save/kmeans/result"
        plt.figure()
        plt.plot(range(len(self.train_davies_bouldin_list_0)), self.train_davies_bouldin_list_0,color='blue', linestyle="solid", label="Train davies_bouldin")
        plt.plot(range(len(self.test_davies_bouldin_list_0)), self.test_davies_bouldin_list_0,color='green', linestyle="solid", label="Test davies_bouldin")
        plt.xlabel('epoch')
        plt.savefig('{}/DBI_n_clusters_{}_lr_{}_round_{}.png'.format(pa,self.args.n_clusters,self.args.lr, self.args.epoch))
        
        # print(f"rank: {self.rank}\nself.train_davies_bouldin_list_0:\n{len(self.train_davies_bouldin_list_0)}\nself.test_davies_bouldin_list_0\n{self.test_davies_bouldin_list_0}")  
        # print(f"\nchange:")
        # for i in range(len(self.change_list)):
        #     print(self.change_list[i])
      
    def launch(self):
        epoch = self.args.epoch
        clients_num_d = pickle.dumps(self.args.num_users)
        self.bucket.put_object("num_clients.txt", clients_num_d)     
        for epo in tqdm(range(1, epoch+1)):
            print("epoch: ", epo)
            if self.not_change_epoch >3:
                print("train have finished!\n")
                # self.test()
                break
            else:
                print(f"epoch:{epoch} self.not_change_epoch:{self.not_change_epoch}")
                self.one_local_round()
                # if epo>0 and epo%1==0:
                #     self.test()
        print(f"epoch: {epo}  ,train done!")
        if self.rank == 0:
            self.write_kmeans()   

if __name__ == '__main__':
    arg = args_parser()
    # lr_trainer = LogisticTrainer(args)
    # logistic_trainer.one_round()






