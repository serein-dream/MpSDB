import csv
import pathlib
import torch.nn
from model.CNN import BottomModel
from .base_trainer import BaseTrainer
from tqdm import tqdm
import pickle
import oss2
import time


class CNNTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, train_list):
        super().__init__(n_f, args, client_rank, 0)
        self.bottom_model = BottomModel(n_f)
        
        path = f"ADD_by_yourself/FaaS_FL2023/VFL/model_save_ok_9_5/cnn/bottom_model_epoch4_client_{str(client_rank)}.pth"
        #self.bottom_model.load_state_dict(torch.load(path))
        self.bottom_model.to(self.device)
        self.train_list = train_list  #938 ([64, 1, 28, 14])
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.bottom_model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.bottom_model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        self.bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
        self.bottom_commu_time = 0
        self.bottom_commu_num = 0
        self.comp_time = 0

    def one_epoch_bottom_fp_bp(self, epoch):
        #data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,shuffle=False, num_workers=0)
        rnd = 1
        print(f"train_list len : {len(self.train_list)}")#torch.Tensor
        #print(f"train_list type : {type(self.train_list[0])}")#torch.Tensor
        div_data_st = time.perf_counter()
        clients_num_d = pickle.dumps(self.args.num_users)
        div_data_ed = time.perf_counter()


        self.bucket.put_object("num_clients.txt", clients_num_d)      
        num = 0
        for batch in self.train_list:
            #start_time = time.time()
            #rnd10_time = 0
            num = num+1
            # print(f"epoch: {epoch}  batch_size:{self.args.batch_size}   batch in one epoch:  {rnd}  bottom_output_{self.args.rank}")
            forwadr_s = time.perf_counter()
            train_x = batch
            #for param in self.bottom_model.parameters():
               #print(f"model shape:  {param.size()}")
            train_x = train_x.to(self.device)
            #train_x.requires_grad=True
            #train_x.retain_grad()
            bottom_output = self.bottom_model(train_x)

            bottom_to_server_dict = {}
            bottom_to_server_dict["bottom_output"] = bottom_output
            bottom_to_server_dict["epoch"] = epoch
            bottom_to_server_dict["rnd"] = rnd
            bottom_to_server_dict["args"] = self.args
            bottom_to_server_dict_d = pickle.dumps(bottom_to_server_dict)
            name = "vfl_from_bottom/rank_"+ str(self.args.rank) +".txt"

            forwadr_e = time.perf_counter()
            
            send_t = time.perf_counter()
            self.bucket.put_object(name, bottom_to_server_dict_d)
            name = "to_bottom_grad/rank_"+ str(self.args.rank) +".txt"
            #time_1 = time.time()
            #rnd10_time += time_1 - start_time         
            while self.bucket.object_exists(name) == False:
                time.sleep(0.05)
            #time_2 = time.time()
            #print("backward...")
            time_2 = time.time()
            to_get = self.bucket.get_object(name).read()
            send_e = time.perf_counter()

            bottom_comm_time = send_e - send_t
            self.bottom_commu_time += bottom_comm_time


            back_s = time.perf_counter()
            grads = pickle.loads(to_get)
            self.bucket.delete_object(name)

            self.optimizer.zero_grad()
            bottom_output.backward(grads)
            self.optimizer.step()
            #print(f"grads:{grads}  train_x : {train_x.grad}")
            #t = self.bottom_model.state_dict()
            #print(t)
            # print(rnd, idx)
            rnd += 1
            if epoch % 1 == 0:
              torch.save(self.bottom_model.state_dict(),
                        "ADD_by_yourself/FaaS_FL2023/VFL_faas/model_save/cnn/bottom_model_epoch{0}_client_{1}.pth".format(epoch, self.client_rank))
              
              
            back_e = time.perf_counter()
            self.bottom_commu_num += 1
            self.comp_time += div_data_st - div_data_ed + back_e - back_s + forwadr_e - forwadr_s
            
            print(f"\n\nnum:{self.bottom_commu_num}\ncompute:{self.comp_time}\n")
            # print(f"rnd:{rnd} time:{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}") 
            #if num == 30:
            #    return
        # print("bias",self.bottom_model.fc1.bias)


    def get_data(self):
        for batch in self.train_list:
            train_x = batch
            train_x = train_x.reshape(train_x.size(0), -1)
            print(f"shape:{train_x.shape}")
            train_x = train_x.numpy()
            result_path = f"ADD_by_yourself/FaaS_FL2023/VFL_faas/trainer/v_cnn/v_train_cnn_{self.client_rank}.csv"
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()  
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(train_x)
            print(f"train_x {self.client_rank}:  ")
            raise


    def launch(self):
        epoch = self.args.epoch
        for epo in tqdm(range(1, epoch+1)):
            self.one_epoch_bottom_fp_bp(epo)


