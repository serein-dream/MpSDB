from conf import args_parser
import torch.nn
from data_modeling.model import Logistic
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import sys
import tenseal as ts
import os
import pickle
import oss2
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
class LogisticTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataset_t, rank):
        super().__init__(args, dataset, dataset_t, [0])
        self.rank = rank
        self.model = Logistic(n_f=self.n_f)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.MSELoss()
        self.init_metrics()
        self.fileloader = self.generate_fileloader()
    def init_metrics(self):
        self.train_loss_list = [[] for _ in range(3)]
        self.train_accuracy_list = [[] for _ in range(3)]
        self.test_set_accuracy_list = [[] for _ in range(3)]
        self.waiting_for_average_done_time_list = []
    def generate_fileloader(self):
        filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{filetime}_LR__dataset__lr_{self.args.lr}__epoch_{self.args.epoch}__rounds_{self.args.rounds}__batch_size_{self.args.batch_size}"
    def set_loss_a(self):
        for i in range(3):
            if self.rank == i:
                self.test_set_accuracy_list[i].append(100000)
                self.train_loss_list[i].append(100000)
                self.train_accuracy_list[i].append(100000)
    def enc_dec(self, the_params):
        return self.client.enc_dec_inclient(the_params)
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
    def one_round_serverless(self, rnd):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        epoch_results = self.train_epochs(data_loader)
        self.log_epoch_results(epoch_results, rnd)
        self.upload_encrypted_params(rnd)
        self.wait_for_aggregation()
        self.download_and_apply_aggregated_params()
    def train_epochs(self, data_loader):
        accum_loss = 0
        accum_correct = 0
        set_size = 0
        for e in range(self.epoch):
            for batch in data_loader:
                train_x, train_y = batch
                train_y = train_y.unsqueeze(dim=1)
                y_pred = self.model(train_x)
                loss = self.criterion(y_pred, train_y)
                accum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                set_size += train_x.size(0)
                accum_correct += y_pred.ge(0.5).squeeze().eq(train_y.squeeze()).sum().item()
        return accum_loss / self.epoch, accum_correct / set_size
    def log_epoch_results(self, epoch_results, rnd):
        avg_loss, avg_acc = epoch_results
        self.train_loss_list[self.rank].append(avg_loss)
        self.train_accuracy_list[self.rank].append(avg_acc)
    def upload_encrypted_params(self, rnd):
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
    def download_and_apply_aggregated_params(self):
        average_params_s_dumps = self.bucket.get_object('global_params.txt').read()
        average_params_s = pickle.loads(average_params_s_dumps)
        average_params = average_params_s
        self.dec_aggregated_params(average_params)
        self.mark_update_done()
    def mark_update_done(self):
        s_d = pickle.dumps("ok")
        update_done_name = f"OK{self.rank}.txt"
        self.bucket.put_object(update_done_name, s_d)
    def check_glo(self):
        average_params_s_dumps = self.bucket.get_object('global_params0.txt').read()
        average_params_s = pickle.loads(average_params_s_dumps)
        average_params = average_params_s
        self.dec_aggregated_params(average_params)
    def send_enc_params(self):
        params_list = self.get_params_list()
        flat_tensor = flatten_tensors(params_list).detach()
        enc_vector = self.client.enc_tensor(flat_tensor)
        return enc_vector.serialize()
    def dec_aggregated_params(self, average_params):
        received_list = self.client.dec_aggregated_params_client(average_params)
        params_list = self.get_params_list()
        flat_tensor = flatten_tensors(params_list).detach()
        received_tensors = torch.tensor(received_list, dtype=flat_tensor.dtype, device=flat_tensor.device)
        self.update_params(received_tensors)
    def test(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.test_sampler)
        accum_correct = 0
        set_size = 0
        for batch in data_loader:
            test_x, test_y = batch
            test_y = test_y.unsqueeze(dim=1)
            y_pred = self.model(test_x)
            set_size += test_x.size(0)
            accum_correct += y_pred.ge(0.5).squeeze().eq(test_y.squeeze()).sum().item()
        self.test_set_accuracy_list[self.rank].append(accum_correct / set_size)
    def write(self, load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list):
        pa = "/path/to/results/"
        with open(f"{pa}/train_loss_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.train_loss_list[self.rank]))
        with open(f"{pa}/train_accuracy_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.train_accuracy_list[self.rank]))
        with open(f"{pa}/test_set_accuracy_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.test_set_accuracy_list[self.rank]))
        with open(f"{pa}/time.txt", 'a') as f:
            f.write(
                f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\n"
                f"total time: {total_time}\n"
                f"load_data_time: {load_data_time}\n"
                f"waiting_for_init_list_time_avg: {sum(waiting_for_init_list_time_list) / len(waiting_for_init_list_time_list)} {waiting_for_init_list_time_list}\n"
                f"one_round_time_list avg: {sum(one_round_time_list) / len(one_round_time_list)} {one_round_time_list}\n"
                f"waiting_for_average_done_time_list_avg: {sum(self.waiting_for_average_done_time_list) / len(self.waiting_for_average_done_time_list)} {self.waiting_for_average_done_time_list}\n\n\n"
            )
    @property
    def bucket(self):
        auth = oss2.Auth('YOUR_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_SECRET')
        return oss2.Bucket(auth, "ADD_by_yourself", "YOUR_BUCKET_NAME")
if __name__ == '__main__':
    args = args_parser()