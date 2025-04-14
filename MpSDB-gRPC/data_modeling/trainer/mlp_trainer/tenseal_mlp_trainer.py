from conf import args_parser
import torch.nn
import matplotlib.pyplot as plt
from data_modeling.model import MLP
import sys
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, Dataset
import os
import time as t
import datetime
import matplotlib.pyplot as plt
sys.path.append("../../../")

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    

class MLPTrainer(BaseTrainer):
    def __init__(self, args, dataset, test_dataset, rank):
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        _ = 0
        super().__init__(args, dataset, test_dataset, _)
        self.args = args

        self.model = MLP(dim_in=args.n_features, dim_hidden=50,
                                  dim_out=args.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
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
        self.fileloader = self.filetime+"MLP__" + str(self.args.dataset) + "__lr_"+ str(self.args.lr) +"__epoch_"+ str(self.args.epoch) +"__rounds_"+ str(self.args.rounds)+"__batch_size__"+str(self.args.local_bs)

        
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

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.85*len(idxs))]
        idxs_val = idxs[int(0.85*len(idxs)):]

        #print(f"len(trainloader):{len(idxs_train)}")
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        '''
        print(f"len(idxs_train):{len(idxs_train)}\nafter_len(trainloader):{len(trainloader)}")
        
        len(idxs_train):17000
        after_len(trainloader):85
        '''
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        return trainloader, validloader


    def one_local_round(self,rnd):
        self.model.train()
        local_epoch_loss = []
        # if rnd == 20 or rnd == 30:
        #     self.update_lr(0.97)

        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        accum_loss = 0
        accum_correct = 0
        set_size = 0

        for e in range(self.epoch):
            batch_loss = []
            for batch in data_loader:
                train_x, train_y = batch
                
                train_y = train_y.unsqueeze(dim=1)

                y_pred = self.model(train_x)
                loss = self.criterion(y_pred, train_y)
                print(f"epoch: {self.epoch} \npred:{y_pred}\nlabels:\n{train_y}\nloss:{loss}")
                accum_loss += loss.item()
                set_size += train_x.size(0)
                accum_correct += y_pred.ge(0.5).squeeze().eq(train_y.squeeze()).sum().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if batch_idx % 10 == 0:
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         rnd, e, batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                if e % 1 == 0:
                    print("loss: ", accum_loss)
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

    def test(self,test_dataset):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Inference
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels) 
        print("test accuracy: ", 100*correct/total, "%")    
        if self.rank == 0:
            self.test_set_accuracy_list_0.append(correct/total)
        elif self.rank == 1:
            self.test_set_accuracy_list_1.append(correct/total)
        elif self.rank == 2:
            self.test_set_accuracy_list_2.append(correct/total)  


    def write(self,load_time):
        end_time = t.perf_counter()
        total_time = end_time - self.start_time
        pa = "/home/maxvyang01/SecureDatabase/script/l_result/temp/mlp"
        #数据存档
        if self.rank == 0:
            pat = "/home/maxvyang01/SecureDatabase/script/l_result/"+self.fileloader
            train_loss_list = self.train_loss_list_0
            train_accuracy_list = self.train_accuracy_list_0
            test_accracy_list = self.test_set_accuracy_list_0
            with open(f"{pa}/train_loss_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_loss_list_0))
            with open(f"{pa}/train_accuracy_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_accuracy_list_0))
            with open(f"{pa}/test_accuracy_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.test_set_accuracy_list_0))
            with open(f"{pa}/pat.txt", 'w') as train_los_0:
                train_los_0.write(pat)
            with open(f"{pa}/time_rank.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

        elif self.rank == 1:
            train_loss_list = self.train_loss_list_1
            train_accuracy_list = self.train_accuracy_list_1
            test_accracy_list = self.test_set_accuracy_list_1
            with open(f"{pa}/train_loss_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_loss_list_1))
            with open(f"{pa}/train_accuracy_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.train_accuracy_list_1))
            with open(f"{pa}/test_accuracy_list_1.txt", 'w') as train_los_1:
                train_los_1.write(str(self.test_set_accuracy_list_1))
            with open(f"{pa}/time_rank.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")


        elif self.rank == 2:
            train_loss_list = self.train_loss_list_2
            train_accuracy_list = self.train_accuracy_list_2
            test_accracy_list = self.test_set_accuracy_list_2
            with open(f"{pa}/train_loss_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_loss_list_2))
            with open(f"{pa}/train_accuracy_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.train_accuracy_list_2))
            with open(f"{pa}/test_accuracy_list_2.txt", 'w') as train_los_2:
                train_los_2.write(str(self.test_set_accuracy_list_2))
            with open(f"{pa}/time_rank.txt", 'a') as train_los_0:
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
