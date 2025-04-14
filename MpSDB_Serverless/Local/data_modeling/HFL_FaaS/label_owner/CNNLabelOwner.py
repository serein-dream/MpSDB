import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
import torch.nn as nn
import pickle
import torch
from model.utils import get_device
import oss2
import time
class CNNLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):
    def __init__(self, model, label_list, args):
        self.device = get_device()
        self.top_model = model.to(self.device)
        self.args = args
        self.label_list = label_list
        self.optimizer = self._initialize_optimizer()
        self.criterion = nn.NLLLoss().to(self.device)
        self.batch_size = args.batch_size
        self.num_iter = len(self.label_list)
        self.total_loss = 0
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.epoch = 1
        self.rnd = 1
        self.bucket = self._initialize_oss_bucket()
        self.top_compute_num = 0
        self.top_compute_time = 0
    def _initialize_optimizer(self):
        if self.args.optimizer == 'sgd':
            return optim.SGD(self.top_model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            return optim.Adam(self.top_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
    def _initialize_oss_bucket(self):
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        return oss2.Bucket(auth, "ADD_by_yourself", "123")
    def __get_labels(self, rnd):
        assert rnd < self.num_iter + 1
        return self.label_list[rnd - 1]
    def __fp_bp(self, middle_output):
        middle_output.retain_grad()
        self.top_model.train()
        pred = self.top_model(middle_output)
        labels = self.__get_labels(self.rnd).to(self.device)
        loss = self.criterion(pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()
        if self.rnd == self.args.rnd and self.epoch > 1 and (self.epoch + 1) % 2 == 0:
            self.total_loss = 0
        return middle_output.grad
    def top_fp_bp(self, middle_output):
        middle_grad = self.__fp_bp(middle_output)
        if self.epoch % 1 == 0:
            torch.save(self.top_model.state_dict(), f"/cnn/top_model_epoch{self.epoch}.pth")
        return middle_grad
    def clean_bucket(self):
        for obj in oss2.ObjectIterator(self.bucket):
            flag = any(keyword in obj.key for keyword in ["ts_ckks_pk.config", "num_clients.txt", "ini_middle_model"])
            if not flag:
                self.bucket.delete_object(obj.key)
    def run_while(self):
        self.clean_bucket()
        print("CNN Label Owner start")
        while self.epoch < self.args.epoch + 1 or self.rnd < self.args.rnd + 1:
            start_time = time.time()
            while not self.bucket.object_exists("vfl_from_middle_to_top.txt") and time.time() - start_time < 600:
                time.sleep(0.01)
            to_get = self.bucket.get_object("vfl_from_middle_to_top.txt").read()
            self.bucket.delete_object("vfl_from_middle_to_top.txt")
            middle_to_top_dict = pickle.loads(to_get)
            self.epoch = middle_to_top_dict["epoch"]
            self.rnd = middle_to_top_dict["rnd"]
            middle_grad = self.top_fp_bp(pickle.loads(middle_to_top_dict["middle_output"]))
            middle_grad_d = pickle.dumps(middle_grad)
            self.bucket.put_object("vfl_from_top_model.txt", middle_grad_d)