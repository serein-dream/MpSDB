from conf import args_parser
import torch.nn
from data_modeling.model import Linear
import sys,os
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import numpy as np
import time as t
import datetime
import matplotlib.pyplot as plt
sys.path.append("../../../")


class LinearTrainer(BaseTrainer):
    def __init__(self, args, dataset, test_dataset ,rank):
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        super().__init__(args, dataset, test_dataset,[0])
        self.model = Linear(n_f=self.n_f)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.MSELoss()
        self.rank = rank
        self.train_loss_list_1 = []
        self.train_loss_list_2 = []
        self.train_loss_list_0 = []
        self.train_RMSE_list_1 = []
        self.train_RMSE_list_2 = []
        self.train_RMSE_list_0 = []
        self.test_set_RMSE_list_0 = []
        self.test_set_RMSE_list_1 = []
        self.test_set_RMSE_list_2 = []

    def set_loss_a(self):
        if self.rank == 0:
            self.test_set_RMSE_list_0.append(100000)
            self.train_loss_list_0.append(100000)
            self.train_RMSE_list_0.append(100000)
        elif self.rank == 1:
            self.test_set_RMSE_list_1.append(100000)
            self.train_loss_list_1.append(100000)
            self.train_RMSE_list_1.append(100000)
        elif self.rank == 2:
            self.test_set_RMSE_list_2.append(100000)
            self.train_loss_list_2.append(100000)
            self.train_RMSE_list_2.append(100000)     
  

    def one_local_round(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        mse_f = torch.nn.MSELoss(reduction='sum')
        for e in range(self.epoch):
            accum_loss = 0
            size_set = 0
            mse_loss = 0
            for batch in data_loader:
                train_x, train_y = batch
                train_y = train_y.unsqueeze(dim=1)
                
                y_pred = self.model(train_x)
                #print("\ntrain_y:\n", train_y,"\ny_pred:\n",y_pred)
                #print(f"train_x:\n{train_x}\ntrain_y:\n{train_y}\ny_pred:\n{y_pred}")
                loss = self.criterion(y_pred, train_y)
                accum_loss += loss.item()
                size_set += train_x.size(0)

                mse_loss += mse_f(y_pred, train_y).item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if e == self.epoch-1:
                print("loss: ", accum_loss / size_set)
                print("RMSE: ", np.sqrt(mse_loss / size_set))
                print(size_set)
        if self.rank == 0:
            self.train_loss_list_0.append(accum_loss / size_set)
            self.train_RMSE_list_0.append(mse_loss / size_set)  
        elif self.rank == 1:
            self.train_loss_list_1.append(accum_loss / size_set)
            self.train_RMSE_list_1.append(mse_loss / size_set)    
        elif self.rank == 2:
            self.train_loss_list_2.append(accum_loss / size_set)
            self.train_RMSE_list_2.append(mse_loss / size_set)    
        self.average_params()

    def test(self):
        data_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.test_sampler)
        accum_mse = 0
        size_set = 0
        mse_f = torch.nn.MSELoss(reduction='sum')
        for batch in data_loader:
            test_x, test_y = batch
            test_y = test_y.unsqueeze(dim=1)
            y_pred = self.model(test_x)
            accum_mse += mse_f(y_pred, test_y).item()
            size_set += test_x.size(0)
            #print("\ntest_y:\n", test_y,"\ny_pred:\n",y_pred)
        print("test-set RMSE: ", np.sqrt(accum_mse / size_set))
        if self.rank == 0:
            self.test_set_RMSE_list_0.append(accum_mse / size_set)
        elif self.rank == 1:
            self.test_set_RMSE_list_1.append(accum_mse / size_set)
        elif self.rank == 2:
            self.test_set_RMSE_list_2.append(accum_mse / size_set) 


    def write(self,begin_time,load_time):
        end_time = t.perf_counter()
        total_time = end_time - self.start_time
        pa = "/home/maxvyang01/SecureDatabase/script/l_result/temp/linear"
        if self.rank == 0:
            fileloader = self.filetime+"Linear__wine_quity__lr_"+ str(self.args.lr) +"__epoch_"+ str(self.args.epoch) +"__rounds_"+ str(self.args.rounds)+"__batch_size__"+str(self.args.batch_size)
            pat = "/home/maxvyang01/SecureDatabase/script/l_result/"+fileloader
            with open(f"{pa}/train_loss_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_loss_list_0))
            with open(f"{pa}/train_RMSE_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_RMSE_list_0))
            with open(f"{pa}/test_set_RMSE_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.test_set_RMSE_list_0))
            with open(f"{pa}/pat.txt", 'w') as train_los_0:
                train_los_0.write(pat)
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\nbegin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

        elif self.rank == 1:
            train_loss_list = self.train_loss_list_1
            train_RMSE_list = self.train_RMSE_list_1
            test_set_RMSE_list = self.test_set_RMSE_list_1
            with open(f"{pa}/train_loss_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_loss_list_1))
            with open(f"{pa}/train_RMSE_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_RMSE_list_1))
            with open(f"{pa}/test_set_RMSE_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.test_set_RMSE_list_1))
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\nbegin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

        elif self.rank == 2:
            train_loss_list = self.train_loss_list_2
            train_RMSE_list = self.train_RMSE_list_2
            test_set_RMSE_list = self.test_set_RMSE_list_2
            with open(f"{pa}/train_loss_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_loss_list_2))
            with open(f"{pa}/train_RMSE_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_RMSE_list_2))
            with open(f"{pa}/test_set_RMSE_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.test_set_RMSE_list_2))
            with open(f"{pa}/time.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\nbegin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")
        '''
        #画图
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list)
        plt.xlabel('rounds')
        plt.ylabel('Train loss')
        plt.savefig('{}/rank_{}_train_loss_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))
        
        plt.figure()
        plt.plot(range(len(train_RMSE_list)), train_RMSE_list)
        plt.xlabel('rounds')
        plt.ylabel('train_RMSE_list')
        plt.savefig('{}/rank_{}_train_RMSE_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))

        plt.figure()
        plt.plot(range(len(test_set_RMSE_list)), test_set_RMSE_list)
        plt.xlabel('rounds')
        plt.ylabel('test_set_RMSE_list')
        plt.savefig('{}/rank_{}_test_set_RMSE_{}_{}_lr_{}_round_{}_local_epoch_{}.png'.format(pa,self.args.rank,self.args.dataset, self.args.model,self.args.lr, self.args.rounds, self.args.epoch))
        '''    
if __name__ == '__main__':
    arg = args_parser()
    # lr_trainer = LogisticTrainer(args)
    # logistic_trainer.one_round()
