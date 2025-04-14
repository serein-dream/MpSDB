from conf import args_parser
import torch.nn
from data_modeling.model import Logistic
import sys
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import os
import time as t
import datetime
import matplotlib.pyplot as plt
sys.path.append("../../../")


class LogisticTrainer(BaseTrainer):
    def __init__(self, args, dataset, test_dataset, rank):
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        super().__init__(args, dataset, test_dataset,[0])
        self.model = Logistic(n_f=self.n_f)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.BCELoss()
        self.rank = rank
        self.train_loss_list_1 = []
        self.train_loss_list_2 = []
        self.train_loss_list_0 = []
        self.train_accuracy_list_1 = []
        self.train_accuracy_list_2 = []
        self.train_accuracy_list_0 = []
        self.test_set_accuracy_list_0 = []
        self.test_set_accuracy_list_1 = []
        self.test_set_accuracy_list_2 = []

        self.fileloader = self.filetime+"LR__cancer__lr_"+ str(self.args.lr) +"__epoch_"+ str(self.args.epoch) +"__rounds_"+ str(self.args.rounds)+"__batch_size__"+str(self.args.batch_size)
    def one_local_round(self,rnd):
        # if rnd in [2,4,6,8,10,12]:
        #     self.update_lr(0.9)
        # else:
        #     self.update_lr(0.95)
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        accum_loss = 0
        accum_correct = 0
        set_size = 0
        for e in range(self.epoch):
            for batch in data_loader:
                train_x, train_y = batch
                train_y = train_y.unsqueeze(dim=1)

                y_pred = self.model(train_x)
                loss = self.criterion(y_pred, train_y)
                #print(f"epoch: {self.epoch} \npred:{y_pred}\nlabels:\n{train_y}\nloss:{loss}")
                accum_loss += loss.item()
                set_size += train_x.size(0)
                accum_correct += y_pred.ge(0.5).squeeze().eq(train_y.squeeze()).sum().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print(f"epoch: {e} grad:{train_x.grad}  \n model:\n{self.model.state_dict()}")
                #raise
            if e % 1 == 0:
                print(f"e:{e}")
                #print("loss: ", accum_loss)
                print("train accuracy: ", accum_correct / set_size,"\naccum_correct: ", accum_correct,"\nset_size: ",set_size)
        if self.rank == 0:
            self.train_loss_list_0.append(accum_loss / self.epoch)
            self.train_accuracy_list_0.append(accum_correct / set_size)  
        elif self.rank == 1:
            self.train_loss_list_1.append(accum_loss / self.epoch)
            self.train_accuracy_list_1.append(accum_correct / set_size)    
        elif self.rank == 2:
            self.train_loss_list_2.append(accum_loss / self.epoch)
            self.train_accuracy_list_2.append(accum_correct / set_size)    
        self.average_params()
    def set_loss_a(self):
        if self.rank == 0:
            self.test_set_accuracy_list_0.append(100000)
            self.train_loss_list_0.append(100000)
            self.train_accuracy_list_0.append(100000)
        elif self.rank == 1:
            self.test_set_accuracy_list_1.append(100000)
            self.train_loss_list_1.append(100000)
            self.train_accuracy_list_1.append(100000)
        elif self.rank == 2:
            self.test_set_accuracy_list_2.append(100000)
            self.train_loss_list_2.append(100000)
            self.train_accuracy_list_2.append(100000)       


    def test(self):
        data_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.test_sampler)
        accum_correct = 0
        set_size = 0
        for batch in data_loader:
            test_x, test_y = batch
            test_y = test_y.unsqueeze(dim=1)
            y_pred = self.model(test_x)
            set_size += test_x.size(0)
            accum_correct += y_pred.ge(0.5).squeeze().eq(test_y.squeeze()).sum().item()
        print("test accuracy: ", accum_correct / set_size)
        if self.rank == 0:
            self.test_set_accuracy_list_0.append(accum_correct / set_size)
        elif self.rank == 1:
            self.test_set_accuracy_list_1.append(accum_correct / set_size)
        elif self.rank == 2:
            self.test_set_accuracy_list_2.append(accum_correct / set_size)

    def write(self,load_time):
        end_time = t.perf_counter()
        total_time = end_time - self.start_time
        pa = "/home/maxvyang01/SecureDatabase/script/l_result/temp/LR"
        if self.rank == 0:
            pat = "/home/maxvyang01/SecureDatabase/script/l_result/"+self.fileloader
            train_loss_list = self.train_loss_list_0
            train_accuracy_list = self.train_accuracy_list_0
            test_accracy_list = self.test_set_accuracy_list_0
            with open(f"{pa}/train_loss_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_loss_list_0))
            with open(f"{pa}/train_accuracy_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_accuracy_list_0))
            with open(f"{pa}/test_set_accuracy_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.test_set_accuracy_list_0))
            with open(f"{pa}/pat.txt", 'w') as train_los_0:
                train_los_0.write(pat)
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")
            print(f"\n\ntest: {self.test_set_accuracy_list_1}\n\n")
        elif self.rank == 1:
            train_loss_list = self.train_loss_list_1
            train_accuracy_list = self.train_accuracy_list_1
            test_accracy_list = self.test_set_accuracy_list_1
            with open(f"{pa}/train_loss_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_loss_list_1))
            with open(f"{pa}/train_accuracy_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_accuracy_list_1))
            with open(f"{pa}/test_set_accuracy_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.test_set_accuracy_list_1))
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

        elif self.rank == 2:
            train_loss_list = self.train_loss_list_2
            train_accuracy_list = self.train_accuracy_list_2
            test_accracy_list = self.test_set_accuracy_list_2
            with open(f"{pa}/train_loss_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_loss_list_2))
            with open(f"{pa}/train_accuracy_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_accuracy_list_2))
            with open(f"{pa}/test_set_accuracy_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.test_set_accuracy_list_2))
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")
        ''' 
        #画图
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list)
        plt.xlabel('rounds')
        plt.ylabel('Train loss')
        plt.savefig('{}/rank_{}_train_loss_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))
        
        plt.figure()
        plt.plot(range(len(train_accuracy_list)), train_accuracy_list)
        plt.xlabel('rounds')
        plt.ylabel('train accuracy %')
        plt.savefig('{}/rank_{}_test_loss_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))

        plt.figure()
        plt.plot(range(len(test_accracy_list)), test_accracy_list)
        plt.xlabel('rounds')
        plt.ylabel('Test accracy  %')
        plt.savefig('{}/rank_{}_test_accracy_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))
        ''' 
if __name__ == '__main__':
    args = args_parser()
    # lr_trainer = LogisticTrainer(args)
    # logistic_trainer.one_round()
