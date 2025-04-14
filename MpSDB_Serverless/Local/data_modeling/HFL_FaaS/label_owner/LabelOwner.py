import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
import torch
import torch.optim as optim
from model.utils import dragonnet_loss_binarycross
from model.DragonNet import TopModel
import pickle
class LabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):
    def __init__(self, model, label_set, batch_size):
        self.top_model = model
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=0.01)
        self.label_set = label_set
        self.batch_size = batch_size
        self.num_iter = (len(self.label_set) + self.batch_size - 1) // self.batch_size
    def __get_labels(self, rnd):
        start = (rnd - 1) * self.batch_size
        end = min(rnd * self.batch_size, len(self.label_set))
        order_list = list(range(start, end))
        labels = self.label_set[order_list][0]
        return labels
    def __fp_bp(self, middle_output, rnd):
        raise NotImplementedError
    def top_fp_bp(self, request, context):
        middle_output = pickle.loads(request.params_msg)
        rnd = request.round
        middle_grad = self.__fp_bp(middle_output, rnd)
        response = label_owner_pb2.middle_grad(round=rnd, grad_msg=pickle.dumps(middle_grad))
        return response