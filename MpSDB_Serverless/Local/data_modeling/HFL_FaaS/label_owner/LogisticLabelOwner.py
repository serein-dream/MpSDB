import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
import torch.nn as nn
import pickle
import torch
from model.utils import get_device
import oss2
import time
class LogisticLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):
    def __init__(self, model, label_set, args):
        self.device = get_device()
        self.top_model = model.to(self.device)
        self.client = Client(server_address=args.server_address, client_rank=args.rank)
        self.args = args
        self.label_set = label_set
        self.batch_size = args.batch_size
        self.num_iter = len(self.label_set) // self.batch_size
        if len(self.label_set) % self.batch_size != 0:
            self.num_iter += 1
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=self.args.lr)
        self.criterion = nn.BCELoss().to(self.device)
        self.total_loss = 0
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.epoch = 1
        self.rnd = 1
        self.bucket = self._initialize_oss_bucket()
    def _initialize_oss_bucket(self):
        auth = oss2.Auth('ADD_by_yourself<your-access-key-id>', 'ADD_by_yourself<your-access-key-secret>')
        return oss2.Bucket(auth, "ADD_by_yourself<your-endpoint>", "ADD_by_yourself<your-bucket-name>")
    def _get_labels(self, rnd):
        start = (rnd - 1) * self.batch_size
        end = min(rnd * self.batch_size, len(self.label_set))
        order_list = list(range(start, end))
        labels = self.label_set.__getitem__(order_list)[0]
        return labels
    def _forward_backward_pass(self, middle_output):
        middle_output.retain_grad()
        pred = torch.sigmoid(middle_output).to(self.device)
        labels = self._get_labels(self.rnd).to(self.device)
        loss = self.criterion(pred, labels)
        loss.backward()
        self.total_loss += loss.item()
        middle_grad = middle_output.grad
        return middle_grad
    def top_forward_backward_pass(self, middle_output):
        middle_grad = self._forward_backward_pass(middle_output)
        if self.epoch % 1 == 0:
            self._save_model()
        return middle_grad
    def _save_model(self):
        save_path = f"<your-save-path>/top_model_epoch{self.epoch}.pth"
        torch.save(self.top_model.state_dict(), save_path)
    def _clean_up_oss_space(self):
        for obj in oss2.ObjectIterator(self.bucket):
            flag = any(keyword in obj.key for keyword in ["ts_ckks_pk.config", "num_clients.txt", "test_model"])
            if not flag:
                self.bucket.delete_object(obj.key)
    def _process_middle_output(self, middle_output):
        if "CRY" in self.args.model.upper():
            middle_output = self.client.decry_np(middle_output)
            middle_output = torch.tensor(middle_output, requires_grad=True).view(-1, 1).to(self.device)
        else:
            middle_output = pickle.loads(middle_output)
        return middle_output
    def _upload_middle_gradient(self, middle_grad):
        if "CRY" in self.args.model.upper():
            middle_grad = self.client.encry_np(middle_grad.detach().cpu().numpy())
        else:
            middle_grad = pickle.dumps(middle_grad)
        self.bucket.put_object("vfl_from_top_model.txt", middle_grad)
    def run_while(self):
        self._clean_up_oss_space()
        print("LR Label Owner start")
        while self.epoch < self.args.epoch + 1 or self.rnd < self.args.rnd + 1:
            while not self.bucket.object_exists("vfl_from_middle_to_top.txt"):
                time.sleep(0.01)
            middle_to_top_dict = pickle.loads(self.bucket.get_object("vfl_from_middle_to_top.txt").read())
            middle_output = self._process_middle_output(middle_to_top_dict["middle_output"])
            self.bucket.delete_object("vfl_from_middle_to_top.txt")
            self.epoch = middle_to_top_dict["epoch"]
            self.rnd = middle_to_top_dict["rnd"]
            middle_grad = self.top_forward_backward_pass(middle_output)
            self._upload_middle_gradient(middle_grad)