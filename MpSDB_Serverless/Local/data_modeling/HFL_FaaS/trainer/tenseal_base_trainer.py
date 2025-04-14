from conf import args_parser
import torch.nn
import sys
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from data_modeling.client import Client
from data_modeling.data_loader import MysqlDataSet
class BaseTrainer(object):
    def __init__(self, args, dataset, test_dataset, client_centroids):
        self.args = args
        self.sample_num = args.sample_num
        self.n_f = self.args.n_features
        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.train_split = 0.7
        self.shuffle_dataset = True
        self.random_seed = self.args.seed
        if args.model == "Linear":
            self.train_sampler, self.test_sampler = self.split_train_test_wine()
        else:
            self.train_sampler, self.test_sampler = self.split_train_test()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.client = Client(args.rank, self.sample_num, args.ctx_file)
        if args.model == "kmeans":
            self.client_centroids = self.__dec_init_centroids(client_centroids)
        self.not_change_number = 0
        self.change_list = []
    def __dec_init_centroids(self, enc_centroids):
        n_dims = self.args.n_dims
        dec_centroids = self.client.dec_kmeans_tonumpy(enc_centroids, n_dims)
        return dec_centroids
    def update_lr(self, lr_gamma):
        self.args.lr = self.args.lr * lr_gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
    def split_train_test_wine(self):
        dataset_size = min(len(self.dataset), len(self.test_dataset))
        indices = list(range(dataset_size))
        split = int(np.floor(self.train_split * dataset_size))
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        return train_sampler, test_sampler
    def split_train_test(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        indices_t = list(range(len(self.test_dataset)))
        train_sampler = SubsetRandomSampler(indices)
        test_sampler = SubsetRandomSampler(indices_t)
        return train_sampler, test_sampler
    def transmit(self, params_list):
        flat_tensor = self.flatten_tensors(params_list).detach()
        received_list = self.client.transmit(flat_tensor)
        received_tensors = torch.tensor(received_list, dtype=flat_tensor.dtype, device=flat_tensor.device)
        return received_tensors
    def is_update(self):
        flag, init_params_list = self.client.transmit([], operator="update_flag")
        if flag:
            params_list = self.get_params_list()
            init_params_tensor = torch.tensor(init_params_list)
            for f, t in zip(self.unflatten_tensors(init_params_tensor, params_list), params_list):
                with torch.no_grad():
                    t.set_(f)
        return flag
    def get_params_list(self):
        param_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    p.mul_(self.sample_num)
                    param_list.append(p)
        return param_list
    def average_params(self):
        params_list = self.get_params_list()
        average_params = self.transmit(params_list)
        for f, t in zip(self.unflatten_tensors(average_params, params_list), params_list):
            with torch.no_grad():
                t.set_(f)
    def one_local_round(self):
        raise NotImplementedError
    def test(self):
        raise NotImplementedError
    def launch(self):
        for rnd in range(self.args.rounds):
            update_flag = self.is_update()
            if update_flag:
                self.one_local_round()
    @staticmethod
    def flatten_tensors(tensors):
        if len(tensors) == 1:
            return tensors[0].view(-1).clone()
        flat = torch.cat([t.view(-1) for t in tensors], dim=0)
        return flat
    @staticmethod
    def unflatten_tensors(flat, tensors):
        outputs = []
        offset = 0
        for tensor in tensors:
            numel = tensor.numel()
            outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
            offset += numel
        return tuple(outputs)
if __name__ == '__main__':
    arg = args_parser()