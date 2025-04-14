import torch.nn

import torch.optim as optim
from client import Client
from torch.utils.data import DataLoader
from model.utils import get_device

class BaseTrainer:
    def __init__(self, n_f, args, client_rank, dataset):
        self.bottom_model = None
        self.optimizer = None
        self.args = args
        self.client_rank = client_rank
        self.client = Client(server_address=self.args.server_address, client_rank=client_rank)
        #if args.model.upper() == "Linear" or args.model.upper() == "LR":
        self.dataset = dataset
        self.device = get_device()

    def transmit(self, bottom_output, epoch, rnd):
        grads = self.client.transmit(bottom_output, epoch, rnd)
        return grads

    def transmit_cry(self, bottom_output, epoch, rnd):
        grads = self.client.transmit_cry(bottom_output, epoch, rnd)
        return grads

    def one_epoch_bottom_fp_bp(self, epoch):
        raise NotImplementedError
