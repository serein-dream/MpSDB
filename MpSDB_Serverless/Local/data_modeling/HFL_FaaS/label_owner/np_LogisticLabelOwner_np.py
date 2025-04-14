import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
import torch.nn as nn
import pickle
import torch
from model.utils import get_device
import matplotlib.pyplot as plt
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.test_model import test_LR

class LogisticLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):

    def __init__(self, model, label_set, args):
        self.device = get_device()
        self.top_model = model
        self.args = args
        self.label_set = label_set
        self.batch_size = args.batch_size
        self.num_iter = int(self.label_set.__len__() / self.batch_size)
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=self.args.lr)
        self.criterion = torch.nn.BCELoss()
        self.total_loss = 0
        self.train_loss_list = []
        self.train_accuracy_list = []
        if self.label_set.__len__() % self.batch_size != 0:
            self.num_iter += 1

    def __get_labels(self, rnd):
        start = (rnd - 1) * self.batch_size
        end = rnd * self.batch_size
        if end > self.label_set.__len__():
            end = self.label_set.__len__()
        order_list = [i for i in range(start, end)]
        labels = self.label_set.__getitem__(order_list)[0]
        return labels

    def __fp_bp(self, middle_output, rnd):
        summed_forward_result_tensor = torch.tensor(middle_output)
        print(type(summed_forward_result_tensor))
        summed_forward_result_tensor.requires_grad=True
        summed_forward_result_tensor.retain_grad()
        pred_y = torch.sigmoid(summed_forward_result_tensor).cuda()
        labels = self.__get_labels(rnd)
        loss = self.criterion(pred_y.cuda(), labels.cuda())
        loss.backward()

        #print(f"epoch: {self.epoch} \npred:{pred_y}\nlabels:\n{labels}\nloss:{loss}")
        y_grad = summed_forward_result_tensor.grad   
        self.total_loss += loss.item()
        if rnd == self.args.rnd and self.epoch>1 and (self.epoch+1) % 2==0:#按数据集batch数
            '''
            _, pred_labels = torch.max(pred, 1)
            pred_labels = pred_labels.view(-1)
            correct = torch.sum(torch.eq(pred_labels, labels)).item()
            accuracy = correct/len(labels)
            '''
            accuracy = test_LR(self.epoch - 1)
            self.train_accuracy_list.append(accuracy)   
            self.train_loss_list.append(self.total_loss)    
            print(f"epoch: {self.epoch}    accuracy:  {accuracy}   loss: {self.total_loss}")
            self.draw()
            self.total_loss = 0
        print(f"epoch: {self.epoch}\n grad:\n{y_grad} ")
        return y_grad.numpy()

    def top_fp_bp(self, request, context):
        middle_output = pickle.loads(request.params_msg)
        #middle_output_np = middle_output.detach().cpu().numpy()
        print("111")

        #middle_output_ts = torch.tensor(middle_output_np, requires_grad=True).to(self.device)

        rnd = request.round
        self.epoch = request.epoch
        # middle_grad = self.__fp_bp(middle_output, rnd)
        middle_grad = self.__fp_bp(middle_output, rnd)

        response = label_owner_pb2.middle_grad(round=rnd,
                                               grad_msg=pickle.dumps(middle_grad))
        
        
        if self.epoch % 2 == 0:
            torch.save(self.top_model.state_dict(), "ADD_by_yourself/FaaS_FL2023/VFL/model_save/LR/top_model_epoch{0}.pth".format(self.epoch))
        '''
            if epoch == 200:
                self.optimizer = optim.SGD(self.top_model.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
                                           weight_decay=0.01)
        '''
        return response

    def draw(self):
        plt.figure()
        plt.plot(range(len(self.train_loss_list)), self.train_loss_list)
        plt.xlabel('rounds')
        plt.ylabel('Train loss')
        plt.savefig('ADD_by_yourself/FaaS_FL2023/VFL/model_save/LR/result/loss_lr{0}.png'.format(self.args.lr))
        plt.close()
        plt.figure()
        plt.plot(range(len(self.train_accuracy_list)), self.train_accuracy_list)
        plt.xlabel('rounds')
        plt.ylabel('train_accuracy')
        plt.savefig('ADD_by_yourself/FaaS_FL2023/VFL/model_save/LR/result/train_accuracy_lr{0}.png'.format(self.args.lr))
        plt.close()
