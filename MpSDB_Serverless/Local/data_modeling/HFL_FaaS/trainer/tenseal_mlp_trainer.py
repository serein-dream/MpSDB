import random
from transmission.utils import flatten_tensors, unflatten_tensors
from conf import args_parser
import torch.nn
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tenseal as ts
import os
import pickle
import time
import datetime
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class MLPTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataset_t, rank):
        super().__init__(args, dataset, dataset_t, [0])
        self.args = args
        self.model = MLP(dim_in=args.n_features, dim_hidden=args.n_features*5, dim_out=args.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.criterion = torch.nn.BCELoss()
        self.rank = rank
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.test_set_accuracy_list = []
        self.waiting_for_average_done_time_list = []
        self.comp_time = 0
        self.fileloader = self._generate_file_name()
    def _generate_file_name(self):
        filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{filetime}MLP__{self.args.dataset}__lr_{self.args.lr}__epoch_{self.args.epoch}__rounds_{self.args.rounds}__batch_size_{self.args.local_bs}"
    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        return trainloader, validloader
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
        self.model.train()
        local_epoch_loss = []
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        accum_loss, accum_correct, set_size = 0, 0, 0
        for _ in range(self.epoch):
            for batch in data_loader:
                train_x, train_y = batch
                train_y = train_y.unsqueeze(dim=1)
                y_pred = self.model(train_x)
                loss = self.criterion(y_pred, train_y)
                accum_loss += loss.item()
                set_size += train_x.size(0)
                accum_correct += y_pred.ge(0.5).squeeze().eq(train_y.squeeze()).sum().item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._record_metrics(accum_loss, accum_correct, set_size)
        send_msg = self.send_enc_params()
        self._upload_params(send_msg, rnd)
        self._wait_for_aggregation()
        self._update_local_params()
    def _record_metrics(self, accum_loss, accum_correct, set_size):
        loss = accum_loss / self.epoch
        accuracy = accum_correct / set_size
        self.train_loss_list.append(loss)
        self.train_accuracy_list.append(accuracy)
    def _upload_params(self, send_msg, rnd):
        send_msg_dumps = pickle.dumps(send_msg)
        name = f"clients_update_params/client_{self.rank}.txt"
        self.comp_time += self._measure_time(self._upload_to_oss, name, send_msg_dumps)
    def _wait_for_aggregation(self):
        waiting_for_average_done_time_s = time.perf_counter()
        while not self.bucket.object_exists('average_done.txt'):
            time.sleep(0.5)
        average_params_s_dumps = self.bucket.get_object('global_params.txt').read()
        waiting_for_average_done_time_e = time.perf_counter()
        waiting_for_average_done_time = waiting_for_average_done_time_e - waiting_for_average_done_time_s
        self.waiting_for_average_done_time_list.append(waiting_for_average_done_time)
    def _update_local_params(self):
        average_params_s_dumps = self.bucket.get_object('global_params.txt').read()
        average_params_s = pickle.loads(average_params_s_dumps)
        average_params = average_params_s
        self.dec_aggregated_params(average_params)
    def _measure_time(self, func, *args, **kwargs):
        comp_st = time.perf_counter()
        func(*args, **kwargs)
        comp_ed = time.perf_counter()
        return comp_ed - comp_st
    def _upload_to_oss(self, name, send_msg_dumps):
        self.bucket.put_object(name, send_msg_dumps)
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
    def test(self, test_dataset):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = 100 * correct / total
        self.test_set_accuracy_list.append(accuracy)
        return round(accuracy, 2)
    def set_loss_a(self):
        self.test_set_accuracy_list.append(100000)
        self.train_loss_list.append(100000)
        self.train_accuracy_list.append(100000)
    def write(self, load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list):
        pa = "/path/to/result/temp/"
        if self.rank == 0:
            self._write_metrics(pa, load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list)
    def _write_metrics(self, pa, load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list):
        with open(f"{pa}/train_loss_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.train_loss_list))
        with open(f"{pa}/train_accuracy_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.train_accuracy_list))
        with open(f"{pa}/test_accuracy_list_{self.rank}.txt", 'w') as f:
            f.write(str(self.test_set_accuracy_list))
        with open(f"{pa}/time.txt", 'a') as f:
            f.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ntotal time: {total_time}\n  load_data_time: {load_data_time}\n  waiting_for_init_list_time_avg: {sum(waiting_for_init_list_time_list)/len(waiting_for_init_list_time_list)}     {waiting_for_init_list_time_list}\n  one_round_time_list avg: {sum(one_round_time_list)/len(one_round_time_list)}     {one_round_time_list}\n    waiting_for_average_done_time_list_avg: {sum(self.waiting_for_average_done_time_list) / len(self.waiting_for_average_done_time_list)}     {self.waiting_for_average_done_time_list}\n\n\n")
if __name__ == '__main__':
    args = args_parser()