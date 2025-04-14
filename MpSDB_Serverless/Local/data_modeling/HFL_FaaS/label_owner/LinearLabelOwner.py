import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
import numpy as np
import pickle
import torch
from model.utils import get_device
import oss2
import time
class LinearLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):
    
    def __init__(self, model, label_set, args):
        self.device = get_device()
        self.client = Client(server_address=args.server_address, client_rank=args.rank)
        self.top_model = model.to(self.device)
        self.args = args
        self.label_set = label_set
        self.batch_size = args.batch_size
        self.num_iter = int(len(self.label_set) / self.batch_size)
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.total_loss = 0
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.epoch = 1
        self.rnd = 1
        self.auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself<ACCESS_KEY_SECRET>')
        self.bucket = oss2.Bucket(self.auth, "ADD_by_yourself<ENDPOINT>", "ADD_by_yourself<BUCKET_NAME>")
        if len(self.label_set) % self.batch_size != 0:
            self.num_iter += 1
        self.top_compute_time = 0
        self.top_compute_num = 0
        self.top_all_time = 0
    def __get_labels(self, rnd):
        start = (rnd - 1) * self.batch_size
        end = rnd * self.batch_size
        if end > len(self.label_set):
            end = len(self.label_set)
        order_list = list(range(start, end))
        labels = self.label_set[order_list][0]
        return labels
    def __forward_backward_pass(self, middle_output):
        middle_output.retain_grad()
        self.top_model.train()
        pred = self.top_model(middle_output)
        labels = self.__get_labels(self.rnd).to(self.device)
        loss = self.criterion(pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()
        middle_grad = middle_output.grad
        return middle_grad
    def top_forward_backward_pass(self, middle_output):
        middle_grad = self.__forward_backward_pass(middle_output)
        if self.epoch % 1 == 0:
            self.save_model()
        return middle_grad
    def save_model(self):
        torch.save(self.top_model.state_dict(), f"<MODEL_SAVE_PATH>/top_model_epoch{self.epoch}.pth")
    def clear_bucket(self):
        for obj in oss2.ObjectIterator(self.bucket):
            flag = any(keyword in obj.key for keyword in ["ts_ckks_pk.config", "num_clients.txt", "test_model"])
            if not flag:
                self.bucket.delete_object(obj.key)
    def process_middle_output(self, middle_output):
        if "CRY" in self.args.model.upper():
            middle_output = self.client.decry_np(middle_output)
            middle_output = torch.tensor(middle_output, requires_grad=True).view(-1, 1).to(self.device)
        else:
            middle_output = pickle.loads(middle_output)
        return middle_output
    def handle_middle_to_top_dict(self, middle_to_top_dict):
        self.epoch = middle_to_top_dict["epoch"]
        self.rnd = middle_to_top_dict["rnd"]
        middle_output = self.process_middle_output(middle_to_top_dict["middle_output"])
        middle_grad = self.top_forward_backward_pass(middle_output)
        return middle_grad
    def encrypt_middle_grad(self, middle_grad):
        if "CRY" in self.args.model.upper():
            middle_grad = middle_grad.detach().cpu().numpy()
            middle_grad_d = self.client.encry_np(middle_grad)
        else:
            middle_grad_d = pickle.dumps(middle_grad)
        return middle_grad_d
    def run_while(self):
        self.clear_bucket()
        print("Linear Label Owner start")
        while self.epoch < self.args.epoch + 1 or self.rnd < self.args.rnd + 1:
            while not self.bucket.object_exists("vfl_from_middle_to_top.txt"):
                time.sleep(0.01)
            to_get = self.bucket.get_object("vfl_from_middle_to_top.txt").read()
            self.bucket.delete_object("vfl_from_middle_to_top.txt")
            top_st = time.perf_counter()
            st = time.perf_counter()
            middle_to_top_dict = pickle.loads(to_get)
            middle_grad = self.handle_middle_to_top_dict(middle_to_top_dict)
            middle_grad_d = self.encrypt_middle_grad(middle_grad)
            et = time.perf_counter()
            compute_time = et - st
            self.top_compute_num += 1
            self.top_compute_time += compute_time
            self.top_all_time += time.perf_counter() - top_st
            self.bucket.put_object("vfl_from_top_model.txt", middle_grad_d)