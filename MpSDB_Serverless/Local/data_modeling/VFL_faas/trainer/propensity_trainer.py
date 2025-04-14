import torch.nn
from model.PropensityNet import BottomModel
import torch.optim as optim
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from tqdm import tqdm


class PropNetTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.bottom_model = BottomModel(n_f).to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=1e-3)

    def one_epoch_bottom_fp_bp(self, epoch):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,
                                                  shuffle=False, num_workers=0)
        rnd = 1
        for batch in data_loader:

            idx, train_x = batch
            # train_x = train_x.cpu()
            train_x = train_x.to(self.device)
            # print(self.bottom_model.dense.weight.device)
            bottom_output = self.bottom_model(train_x)
            grads = self.transmit(bottom_output, epoch, rnd)
            self.optimizer.zero_grad()
            # bottom_output.backward(grads.cpu())
            bottom_output.backward(grads)
            self.optimizer.step()
            # print(rnd, idx)
            rnd += 1
        # print("bias",self.bottom_model.dense.bias)

    def launch(self):
        epoch = self.args.epoch

        for epo in tqdm(range(1, epoch+1)):
            self.one_epoch_bottom_fp_bp(epo)

        torch.save(self.bottom_model.state_dict(),
                   "../save/psn/bottom_model_epoch{0}_client_{1}.pth".format(epoch, self.client_rank))
