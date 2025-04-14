from transmission.utils import flatten_tensors, unflatten_tensors
from conf import args_parser
import torch.nn
import sys

sys.path.append("../../../")
from data_modeling.client import    Client

from data_modeling.data_loader import MysqlDataSet
import sklearn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class BaseTrainer(object):

    def __init__(self, args, dataset, test_dataset, client_centroids):
        self.args = args

        # initialize settings
        self.sample_num = args.sample_num
        self.n_f = self.args.n_features
        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.train_split = 0.7
        self.shuffle_dataset = True
        self.random_seed = self.args.seed
        if self.args.model == "Linear":
            self.train_sampler, self.test_sampler = self.split_train_test_wine()
        else:    
            self.train_sampler, self.test_sampler = self.split_train_test()
        # initialize model and optimizer
        # self.model = Logistic(n_f=self.n_f)
        # self.optimizer = optim.Adam(self.model.parameters(), lr= self.args.lr)
        # self.criterion = torch.nn.BCELoss()
        self.model = None
        self.optimizer = None
        self.criterion = None

        # initialize the communication params with server
        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]

        self.server_address = args.server_address
        self.client = Client(self.server_address, args.rank, self.sample_num, args.ctx_file)
        print(self.server_address)
        self.com_time = 0
        self.com_cnt = 0
        self.client_centroids = np.random.rand(self.args.n_clusters, self.args.n_dims)
        print("init centroids:      ",end="")
        for i in range(len(self.client_centroids)):
            print(self.client_centroids[i])
        #self.changed = True
        self.not_change_number = 0
        self.change_list = []

    def __dec_init_centroids(self,enc_centroids):
        n_dims = self.args.n_dims
        dec_centroids = self.client.dec_kmeans(enc_centroids, n_dims)
        return dec_centroids

    def update_lr(self,lr_gamma):
        self.args.lr = self.args.lr * lr_gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
        
    def split_train_test(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        indices_t = list(range(len(self.test_dataset)))
        split = int(np.floor(self.train_split * dataset_size))
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_sampler = SubsetRandomSampler(indices)
        test_sampler = SubsetRandomSampler(indices_t)
        return train_sampler, test_sampler
        
    def split_train_test_wine(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.train_split * dataset_size))
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        return train_sampler, test_sampler

    # send model params to server, and get the sum params
    def transmit(self, params_list):
        flat_tensor = flatten_tensors(params_list).detach()
        # get the average params from server
        received_list,com_time = self.client.transmit(flat_tensor)
        received_tensors = torch.tensor(received_list, dtype=flat_tensor.dtype, device=flat_tensor.device)
        self.com_time = self.com_time + com_time
        self.com_cnt = self.com_cnt + 1
        return received_tensors
    
    def transmit_kmeans(self,clients_local_centroids_expanded,client_counts):
        # get the average params from server
        #print(f"client_counts:\n{client_counts}")
        clients_local_centroids_expanded = torch.tensor(clients_local_centroids_expanded)
        clients_local_centroids_expanded = flatten_tensors(clients_local_centroids_expanded).detach()
        latest_centroids = self.client.sum_encrypted_kmeans(self.rank, clients_local_centroids_expanded, client_counts,self.args.n_dims)#com_time
        self.com_cnt = self.com_cnt + 1
        return latest_centroids

    def is_update(self):
        flag, init_params_list,com_time = self.client.transmit([], operator="update_flag")
        self.com_time = self.com_time + com_time
        self.com_cnt = self.com_cnt + 1
        if flag:
            params_list = self.get_params_list()
            init_params_tensor = torch.tensor(init_params_list)
            #print(f"init_params_tensor:\n{init_params_tensor}\n\nparams_list:{params_list}\n\n")#unflatten_tensors(init_params_tensor, params_list):{unflatten_tensors(init_params_tensor, params_list)}")
            for f, t in zip(unflatten_tensors(init_params_tensor, params_list), params_list):
                with torch.no_grad():
                    t.set_(f)
        return flag

    def is_update_kmeans(self):
        latest_centroids, flag, com_time,to_stop = self.client.is_update_kmeans(self.not_change_number, self.args.n_dims)
        self.com_time = self.com_time + com_time
        self.com_cnt = self.com_cnt + 1
        if flag:
            self.client_centroids = latest_centroids
        return flag, to_stop
    
    # from optimizer get the model params,return a list
    def get_params_list(self):
        param_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    p.mul_(self.sample_num)
                    param_list.append(p)

        return param_list

    # update the model params with average params
    def average_params(self):
        #print("1\n")
        params_list = self.get_params_list()
        #print("2\n")
        average_params = self.transmit(params_list)
       # print("3\n")
        # set average params as the new params
        for f, t in zip(unflatten_tensors(average_params, params_list), params_list):
            with torch.no_grad():
                t.set_(f)
        #print("4\n")

    def average_params_kmeans(self,clients_local_centroids_expanded,client_counts):
        latest_centroids = self.transmit_kmeans(clients_local_centroids_expanded,client_counts)
        # set average params as the new params

        change_num = latest_centroids - self.client_centroids
        changed = np.any(np.absolute(change_num) > 0.01)  #0.03 ~200   0.05  ~20
        #self.sum_enc_params.append(updates)
        if changed == True:
            self.not_change_number = 0
            #self.change_list.append(change_num)
        else:
            print("\nnot change!!!\n")
            self.not_change_number = self.not_change_number + 1
            #self.change_list.append(np.zeros(change_num.shape))
            
        self.client_centroids = latest_centroids

    # one communication round
    def one_local_round(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def launch(self):
        for rnd in range(self.args.rounds):
            print("round: ", rnd)
            update_flag = self.is_update()
            print(update_flag)
            if update_flag:
                self.one_local_round()
            else:
                print("not participate in this round")


if __name__ == '__main__':
    arg = args_parser()
    # trainer = BaseTrainer(arg)
    # logistic_trainer.one_round()
