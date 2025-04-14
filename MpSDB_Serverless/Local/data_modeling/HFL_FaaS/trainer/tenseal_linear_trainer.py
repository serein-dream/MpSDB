from conf import args_parser
import torch.nn
from data_modeling.model import Linear
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import numpy as np
import tenseal as ts
import os
import oss2
import pickle
import time
import datetime
def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat
def unflatten_tensors(flat, tensors):
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)
class LinearTrainer(BaseTrainer):
    def __init__(self, args, init_enc_client_centroids_d, dataset, dataset_t, rank):
        self.rank = rank
        super().__init__(args, dataset, dataset_t, init_enc_client_centroids_d)
        self.model = Linear(n_f=self.n_f)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.MSELoss()
        self.train_loss_list = []
        self.train_RMSE_list = []
        self.test_set_RMSE_list = []
        self.waiting_for_average_done_time_list = []
    def set_loss_a(self):
        self.test_set_RMSE_list.append(100000)
        self.train_loss_list.append(100000)
        self.train_RMSE_list.append(100000)
    def enc_dec(self, the_params):
        init_params_tensor = self.client.enc_dec_inclient(the_params)
        return init_params_tensor
    def update_params(self, update_params_tensor):
        params_list = self.get_params_list()
        for f, t in zip(unflatten_tensors(update_params_tensor, params_list), params_list):
            with torch.no_grad():
                t.set_(f)
    def get_params_list(self):
        param_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    p.mul_(self.sample_num)
                    param_list.append(p)
        return param_list
    def train_one_epoch(self, data_loader, mse_f):
        accum_loss = 0
        size_set = 0
        mse_loss = 0
        for batch in data_loader:
            train_x, train_y = batch
            train_y = train_y.unsqueeze(dim=1)
            y_pred = self.model(train_x)
            loss = self.criterion(y_pred, train_y)
            accum_loss += loss.item()
            size_set += train_x.size(0)
            mse_loss += mse_f(y_pred, train_y).item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return accum_loss / size_set, np.sqrt(mse_loss / size_set)
    def one_round_serverless(self, rnd):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        mse_f = torch.nn.MSELoss(reduction='sum')
        final_loss, final_rmse = self.train_one_epoch(data_loader, mse_f)
        self.train_loss_list.append(final_loss)
        self.train_RMSE_list.append(final_rmse ** 2)
        self.upload_params(rnd)
        self.wait_for_aggregation()
        self.download_and_update_params()
    def upload_params(self, rnd):
        send_msg = self.send_enc_params()
        send_msg_dumps = pickle.dumps(send_msg)
        name = f"clients_update_params/client_{self.rank}.txt"
        self.bucket.put_object(name, send_msg_dumps)
    def wait_for_aggregation(self):
        waiting_for_average_done_time_s = time.perf_counter()
        while not self.bucket.object_exists('average_done.txt'):
            time.sleep(0.05)
        waiting_for_average_done_time = time.perf_counter() - waiting_for_average_done_time_s
        self.waiting_for_average_done_time_list.append(waiting_for_average_done_time)
    def download_and_update_params(self):
        average_params_s_dumps = self.bucket.get_object('global_params.txt').read()
        average_params_s = pickle.loads(average_params_s_dumps)
        average_params = average_params_s
        self.dec_aggregated_params(average_params)
        self.mark_update_done()
    def mark_update_done(self):
        s_d = pickle.dumps("ok")
        update_done_name = f"OK{self.rank}.txt"
        self.bucket.put_object(update_done_name, s_d)
    def send_enc_params(self):
        params_list = self.get_params_list()
        flat_tensor = flatten_tensors(params_list).detach()
        enc_vector = self.client.enc_tensor(flat_tensor)
        send_msg = enc_vector.serialize()
        return send_msg
    def dec_aggregated_params(self, average_params):
        received_list = self.client.dec_aggregated_params_client(average_params)
        params_list = self.get_params_list()
        flat_tensor = flatten_tensors(params_list).detach()
        received_tensors = torch.tensor(received_list, dtype=flat_tensor.dtype, device=flat_tensor.device)
        self.update_params(received_tensors)
    def test(self):
        data_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.test_sampler)
        accum_mse = 0
        size_set = 0
        mse_f = torch.nn.MSELoss(reduction='sum')
        for batch in data_loader:
            test_x, test_y = batch
            test_y = test_y.unsqueeze(dim=1)
            y_pred = self.model(test_x)
            accum_mse += mse_f(y_pred, test_y).item()
            size_set += test_x.size(0)
        rmse = np.sqrt(accum_mse / size_set)
        self.test_set_RMSE_list.append(rmse ** 2)
        return rmse
    def write_results(self, load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list):
        pa = "/path/to/save/results/"
        filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fileloader = f"{filetime}_Linear_lr_{self.args.lr}_epoch_{self.args.epoch}_rounds_{self.args.rounds}_batch_size_{self.args.batch_size}"
        pat = f"/path/to/save/results/{fileloader}"
        with open(f"{pa}/pat.txt", 'w') as f:
            f.write(pat)
        with open(f"{pa}/train_loss_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.train_loss_list))
        with open(f"{pa}/train_RMSE_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.train_RMSE_list))
        with open(f"{pa}/test_set_RMSE_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.test_set_RMSE_list))
        with open(f"{pa}/time.txt", 'a') as f:
            f.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ntotal time: {total_time}\n  load_data_time: {load_data_time}\n  waiting_for_init_list_time_avg: {sum(waiting_for_init_list_time_list) / len(waiting_for_init_list_time_list)}     {waiting_for_init_list_time_list}\n  one_round_time_list avg: {sum(one_round_time_list) / len(one_round_time_list)}     {one_round_time_list}\n    waiting_for_average_done_time_list_avg: {sum(self.waiting_for_average_done_time_list) / len(self.waiting_for_average_done_time_list)}     {self.waiting_for_average_done_time_list}\n\n\n")
    @property
    def bucket(self):
        auth = oss2.Auth('YOUR_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_SECRET')
        return oss2.Bucket(auth, "ADD_by_yourself", "YOUR_BUCKET_NAME")
if __name__ == '__main__':
    args = args_parser()