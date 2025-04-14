import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
import torch.nn as nn
import pickle
import torch
from model.utils import get_device
import oss2
import time
class MLPLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):
    def __init__(self, model, label_set, args):
        self.device = get_device()
        self.top_model = model.to(self.device)
        self.args = args
        self.label_set = label_set
        self.batch_size = args.batch_size
        self.num_iter = len(self.label_set) // self.batch_size
        if len(self.label_set) % self.batch_size != 0:
            self.num_iter += 1
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.criterion = nn.BCELoss().to(self.device)
        self.total_loss = 0
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.epoch = 1
        self.rnd = 1
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself<ACCESS_KEY_SECRET>')
        self.bucket = oss2.Bucket(auth, "<OSS_ENDPOINT>", "ADD_by_yourself<BUCKET_NAME>")
        self.top_compute_num = 0
        self.top_compute_time = 0
    def _get_labels(self, rnd):
        start = (rnd - 1) * self.batch_size
        end = min(rnd * self.batch_size, len(self.label_set))
        order_list = list(range(start, end))
        labels = self.label_set[order_list][0]
        return labels
    def _forward_backward_pass(self, middle_output):
        middle_output.retain_grad()
        self.top_model.train()
        pred = self.top_model(middle_output)
        labels = self._get_labels(self.rnd).to(self.device)
        loss = self.criterion(pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()
        if self.rnd == self.args.rnd and self.epoch > 1:
            self.total_loss = 0
        return middle_output.grad
    def top_forward_backward_pass(self, middle_output):
        middle_grad = self._forward_backward_pass(middle_output)
        if self.epoch % 1 == 0:
            torch.save(self.top_model.state_dict(), f"<MODEL_SAVE_PATH>/top_model_epoch{self.epoch}.pth")
        return middle_grad
    def _clean_bucket(self):
        for obj in oss2.ObjectIterator(self.bucket):
            flag = any(keyword in obj.key for keyword in ["ts_ckks_pk.config", "num_clients.txt", "test_model"])
            if not flag:
                self.bucket.delete_object(obj.key)
    def _process_round(self):
        to_get = self.bucket.get_object("vfl_from_middle_to_top.txt").read()
        self.bucket.delete_object("vfl_from_middle_to_top.txt")
        middle_to_top_dict = pickle.loads(to_get)
        self.epoch = middle_to_top_dict["epoch"]
        self.rnd = middle_to_top_dict["rnd"]
        middle_grad = self.top_forward_backward_pass(pickle.loads(middle_to_top_dict["middle_output"]))
        middle_grad_d = pickle.dumps(middle_grad)
        self.bucket.put_object("vfl_from_top_model.txt", middle_grad_d)
    def run(self):
        self._clean_bucket()
        print("MLP Label Owner start")
        while self.epoch < self.args.epoch + 1 or self.rnd < self.args.rnd + 1:
            while not self.bucket.object_exists("vfl_from_middle_to_top.txt"):
                time.sleep(0.01)
            self._process_round()