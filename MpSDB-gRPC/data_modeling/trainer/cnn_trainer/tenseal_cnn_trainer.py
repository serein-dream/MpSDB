import csv
import pathlib
from conf import args_parser
import torch.nn
import matplotlib.pyplot as plt
from data_modeling.model import CNNMnist,CNNFashion_Mnist,CNNCifar,BottomModel,MiddleModel,TopModel
import sys
import torch.optim as optim
from data_modeling.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, Dataset
import os
import time as t
import datetime
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
    

class CNNTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataset_t, idxs, rank):
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        _ = []
        super().__init__(args, dataset, dataset_t,_)
        self.args = args
        if args.gpu_id:
            torch.cuda.set_device(args.gpu_id)
        self.device = 'cuda' if args.gpu else 'cpu'
        if args.dataset == 'mnist':
            self.model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            self.model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            self.model = CNNCifar(args=args)
        '''
        self.bottom_model = BottomModel()
        self.middle_model = MiddleModel()
        self.top_model = TopModel(10)
        self.bottom_model.to(self.device)
        self.middle_model.to(self.device)
        self.top_model.to(self.device)
        '''
        self.model.to(self.device)
        #self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader, self.validloader = self.train_val_test(dataset, list(idxs))
        self.criterion = torch.nn.NLLLoss().to(self.device)

        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
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
        self.fileloader = self.filetime+"CNN__" + str(self.args.dataset) + "__lr_"+ str(self.args.lr) +"__epoch_"+ str(self.args.epoch) +"__rounds_"+ str(self.args.rounds)+"__batch_size__"+str(self.args.local_bs)
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.85*len(idxs))]
        idxs_val = idxs[int(0.85*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        return trainloader, validloader

    def get_data(self,train_dataset,test_dataset):
        trainloader = DataLoader(train_dataset, batch_size=0000,
                                shuffle=False)
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.reshape(images.size(0), -1)
            labels = labels.reshape(labels.size(0), -1)
            print(f" images shape:{images.shape}")
            print(f" labels shape:{labels.shape}")
            train_x = torch.cat((images, labels), dim=1)
            print(f"shape:{train_x.shape}")
            train_x = train_x.numpy()
            result_path = f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/dataset/h_cnn/h_train_cnn_{self.rank}.csv"
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()  
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(train_x)
            print(f"train_x {self.rank}:  {train_x.shape}")

        testloader = DataLoader(test_dataset, batch_size=20000,
                                shuffle=False)
        
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.reshape(images.size(0), -1)
            labels = labels.reshape(labels.size(0), -1)
            train_x = torch.cat((images, labels), dim=1)
            print(f"shape:{train_x.shape}")
            train_x = train_x.numpy()
            result_path = f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/dataset/h_cnn/h_test_cnn_{self.rank}.csv"
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()  
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(train_x)
            print(f"test {self.rank}:  {train_x.shape}")


    def one_local_round(self,rnd):
        self.model.train()
        local_epoch_loss = []
        if rnd == 20 or rnd == 30:
            self.update_lr(0.98)
        for e in range(self.epoch):
            accum_loss = 0
            accum_correct = 0
            set_size = 0
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                '''
                self.bottom_model.zero_grad()
                self.middle_model.zero_grad()
                self.top_model.zero_grad()

                self.bottom_model.train()
                bottom_out = self.bottom_model(images)

                self.middle_model.train()
                middle_out = self.middle_model(bottom_out)

                middle_out.retain_grad()
                self.top_model.train()
                label = self.top_model(middle_out)
                loss = self.criterion(label, labels)
                loss.backward()
                self.optimizer.step()
                middle_grad = middle_out.grad

                bottom_out.retain_grad()
                self.optimizer.zero_grad()
                middle_out.backward(middle_grad)
                self.optimizer.step()
                all_bottom_grads = bottom_out.grad
                print(f"\n\nself.all_bottom_grads:{all_bottom_grads.shape}\n\n")
                
                for param in self.bottom_model.parameters():
                    print(f"model shape:  {param.size()}")
                raise
                '''
                #images.requires_grad=True
                #images.retain_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                #print(f"log_probs： \n{log_probs}  \nlabels：\n{labels}\nloss:{loss}")
                print(f"loss:{loss}")
                loss.backward()
                self.optimizer.step()
                
                #print(f"train_x grad:{images.grad}")
                if batch_idx % 1 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        rnd, e,  len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            local_epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if self.rank == 0:
            self.train_loss_list_0.append(sum(local_epoch_loss)/len(local_epoch_loss))
        elif self.rank == 1:
            self.train_loss_list_1.append(sum(local_epoch_loss)/len(local_epoch_loss))   
        elif self.rank == 2:
            self.train_loss_list_2.append(sum(local_epoch_loss)/len(local_epoch_loss)) 
        if self.rank == 0:
            torch.save(self.model.state_dict(), "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/save_model/cnn_model_epoch{0}.pth".format(rnd))
        #验证集验证
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Inference
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            #loss += batch_loss.item()
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        if self.rank == 0:
            self.train_accuracy_list_0.append(correct/total)
        elif self.rank == 1:
            self.train_accuracy_list_1.append(correct/total)
        elif self.rank == 2:
            self.train_accuracy_list_2.append(correct/total)      
        print("train accuracy: ", 100*correct/total, "%")    
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


    def write(self,load_time):
        end_time = t.perf_counter()
        total_time = end_time - self.start_time
        pa = "/home/maxvyang01/SecureDatabase/script/l_result/temp/cnn"

        #数据存档
        if self.rank == 0:
            train_loss_list = self.train_loss_list_0
            train_accuracy_list = self.train_accuracy_list_0
            test_accracy_list = self.test_set_accuracy_list_0

            pat = "/home/maxvyang01/SecureDatabase/script/l_result/"+self.fileloader
            with open(f"{pa}/train_loss_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_loss_list_0))
            with open(f"{pa}/train_accuracy_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.train_accuracy_list_0))
            with open(f"{pa}/test_accuracy_list_0.txt", 'w') as train_los_0:
                train_los_0.write(str(self.test_set_accuracy_list_0))
            with open(f"{pa}/pat.txt", 'w') as train_los_0:
                train_los_0.write(pat)
            with open(f"{pa}/time_rank.txt", 'a') as train_los_0:
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ncnt: {self.com_cnt}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

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
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ncnt: {self.com_cnt}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")

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
                train_los_0.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}\ncnt: {self.com_cnt}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")
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
