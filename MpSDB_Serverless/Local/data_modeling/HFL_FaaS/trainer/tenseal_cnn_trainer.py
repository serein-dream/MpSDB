import pickle
import torch.nn
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tenseal as ts
import os
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
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), label.clone().detach()
class CNNTrainer(BaseTrainer):
    def __init__(self, args, init_enc_client_centroids_d, dataset, dataset_t, idxs, rank):
        super().__init__(args, dataset, dataset_t, init_enc_client_centroids_d)
        self.args = args
        self.device = 'cuda' if args.gpu else 'cpu'
        self.model = self._select_model()
        self.model.to(self.device)
        self.trainloader, self.validloader = self.train_val_test(dataset, list(idxs))
        self.criterion = torch.nn.NLLLoss().to(self.device)
        self.optimizer = self._select_optimizer()
        self.rank = rank
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.test_set_accuracy_list = []
        self.waiting_for_average_done_time_list = []
        self.fileloader = self._generate_fileloader()
        self.comp_time = 0
    def _select_model(self):
        if self.args.dataset == 'mnist':
            return CNNMnist(args=self.args)
        elif self.args.dataset == 'fmnist':
            return CNNFashion_Mnist(args=self.args)
        elif self.args.dataset == 'cifar':
            return CNNCifar(args=self.args)
    def _select_optimizer(self):
        if self.args.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
    def _generate_fileloader(self):
        filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{filetime}CNN__{self.args.dataset}__lr_{self.args.lr}__epoch_{self.args.epoch}__rounds_{self.args.rounds}__batch_size_{self.args.local_bs}"
    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(1 * len(idxs))]
        idxs_val = idxs[int(0.85 * len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)
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
        comp_st = time.perf_counter()
        self.model.train()
        local_epoch_loss = []
        if rnd in [20, 30]:
            self.update_lr(0.98)
        for e in range(self.args.epoch):
            batch_loss = []
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            local_epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.train_loss_list.append(sum(local_epoch_loss) / len(local_epoch_loss))
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for images, labels in self.validloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        self.train_accuracy_list.append(correct / total)
        send_msg = self.send_enc_params()
        send_msg_dumps = pickle.dumps(send_msg)
        name = f"clients_update_params/client_{self.rank}.txt"
        comp_ed = time.perf_counter()
        self.comp_time += comp_ed - comp_st
        waiting_for_average_done_time_s = time.perf_counter()
        average_done = False
        while not average_done:
            average_done = self.bucket.object_exists('average_done.txt')
            if not average_done:
                time.sleep(0.05)
        average_params_s_dumps = self.bucket.get_object('global_params.txt').read()
        waiting_for_average_done_time_e = time.perf_counter()
        waiting_for_average_done_time = waiting_for_average_done_time_e - waiting_for_average_done_time_s
        self.waiting_for_average_done_time_list.append(waiting_for_average_done_time)
        comp_st = time.perf_counter()
        s_d = pickle.dumps("ok")
        update_done_name = f"OK{self.rank}.txt"
        self.bucket.put_object(update_done_name, s_d)
        average_params_s = pickle.loads(average_params_s_dumps)
        average_params = average_params_s
        self.dec_aggregated_params(average_params)
        comp_ed = time.perf_counter()
        self.comp_time += comp_ed - comp_st
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
    def test(self, test_dataset, epoch):
        torch.save(self.model.state_dict(), f"/path/to/save/cnn_epoch{epoch}_client_{self.rank}.pth")
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        self.test_set_accuracy_list.append(correct / total)
    def set_loss_a(self):
        self.test_set_accuracy_list.append(100000)
        self.train_loss_list.append(100000)
        self.train_accuracy_list.append(100000)
    def write(self, load_data_time, total_time, waiting_for_init_list_time_list, one_round_time_list):
        pa = "/path/to/save/results/"
        if self.rank == 0:
            train_loss_list = self.train_loss_list
            train_accuracy_list = self.train_accuracy_list
            test_accracy_list = self.test_set_accuracy_list
            with open(f"{pa}/train_loss_list_0.txt", 'w') as f:
                f.write(str(train_loss_list))
            with open(f"{pa}/train_accuracy_list_0.txt", 'w') as f:
                f.write(str(train_accuracy_list))
            with open(f"{pa}/test_accuracy_list_0.txt", 'w') as f:
                f.write(str(test_accracy_list))
            with open(f"{pa}/time.txt", 'a') as f:
                f.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ntotal time: {total_time}\n  load_data_time: {load_data_time}\n  waiting_for_init_list_time_avg: {sum(waiting_for_init_list_time_list) / len(waiting_for_init_list_time_list)}     {waiting_for_init_list_time_list}\n  one_round_time_list avg: {sum(one_round_time_list) / len(one_round_time_list)}     {one_round_time_list}\n    waiting_for_average_done_time_list_avg: {sum(self.waiting_for_average_done_time_list) / len(self.waiting_for_average_done_time_list)}     {self.waiting_for_average_done_time_list}\n\n\n")
        elif self.rank == 1:
            with open(f"{pa}/train_loss_list_1.txt", 'w') as f:
                f.write(str(self.train_loss_list))
            with open(f"{pa}/train_accuracy_list_1.txt", 'w') as f:
                f.write(str(self.train_accuracy_list))
            with open(f"{pa}/test_accuracy_list_1.txt", 'w') as f:
                f.write(str(self.test_set_accuracy_list))
            with open(f"{pa}/time.txt", 'a') as f:
                f.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ntotal time: {total_time}\n  load_data_time: {load_data_time}\n  waiting_for_init_list_time_avg: {sum(waiting_for_init_list_time_list) / len(waiting_for_init_list_time_list)}     {waiting_for_init_list_time_list}\n  one_round_time_list avg: {sum(one_round_time_list) / len(one_round_time_list)}     {one_round_time_list}\n    waiting_for_average_done_time_list_avg: {sum(self.waiting_for_average_done_time_list) / len(self.waiting_for_average_done_time_list)}     {self.waiting_for_average_done_time_list}\n\n\n")
        elif self.rank == 2:
            with open(f"{pa}/train_loss_list_2.txt", 'w') as f:
                f.write(str(self.train_loss_list))
            with open(f"{pa}/train_accuracy_list_2.txt", 'w') as f:
                f.write(str(self.train_accuracy_list))
            with open(f"{pa}/test_accuracy_list_2.txt", 'w') as f:
                f.write(str(self.test_set_accuracy_list))
            with open(f"{pa}/time.txt", 'a') as f:
                f.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ntotal time: {total_time}\n  load_data_time: {load_data_time}\n  waiting_for_init_list_time_avg: {sum(waiting_for_init_list_time_list) / len(waiting_for_init_list_time_list)}     {waiting_for_init_list_time_list}\n  one_round_time_list avg: {sum(one_round_time_list) / len(one_round_time_list)}     {one_round_time_list}\n    waiting_for_average_done_time_list_avg: {sum(self.waiting_for_average_done_time_list) / len(self.waiting_for_average_done_time_list)}     {self.waiting_for_average_done_time_list}\n\n\n")
if __name__ == '__main__':
    args = args_parser()