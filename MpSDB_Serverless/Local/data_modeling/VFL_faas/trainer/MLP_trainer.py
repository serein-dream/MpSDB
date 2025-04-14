import torch.nn
from model.MLP import BottomModel
import torch.optim as optim
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from tqdm import tqdm
import oss2
import time
import pickle

class MLPTrainer(BaseTrainer):
    
        # mnist
        #def __init__(self, n_f, args, client_rank, train_list):
    def __init__(self, n_f, args, client_rank, dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.bottom_model = BottomModel(n_f)
        self.bottom_model.to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=self.args.lr,weight_decay=1e-4 )
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        self.bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
        self.bottom_commu_time = 0
        self.bottom_commu_num = 0
        self.comp_time = 0

    def one_epoch_bottom_fp_bp(self, epoch):
        div_data_st = time.perf_counter()
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,
                                                  shuffle=False, num_workers=0)
        #data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,shuffle=False, num_workers=0)
        rnd = 1
        #print(f"train_list type : {len(self.train_list)}")#torch.Tensor
        clients_num_d = pickle.dumps(self.args.num_users)
        div_data_ed = time.perf_counter()


        self.bucket.put_object("num_clients.txt", clients_num_d)      
        # mnist
        #        for batch in self.train_list:       
        for batch in data_loader:
            start_time = time.time()
            rnd10_time = 0
            forwadr_s = time.perf_counter()
            idx, train_x = batch
            #print(f"train_x shape:  {train_x.shape}")
            train_x = train_x.to(self.device)
            #print(f"train_x shape:{train_x.shape}")
            bottom_output = self.bottom_model(train_x)
            #for param in self.bottom_model.parameters():
                #print(f"model shape:  {param.size()}")
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
            time_1 = time.time()
            rnd10_time += time_1 - start_time         
            while self.bucket.object_exists(name) == False:
                time.sleep(0.05)
            time_2 = time.time()
            to_get = self.bucket.get_object(name).read()
            send_e = time.perf_counter()

            bottom_comm_time = send_e - send_t
            self.bottom_commu_time += bottom_comm_time

            self.bucket.delete_object(name)
            print(f"epoch: {epoch}  rnd:  {rnd}  get middle_output_bp grad")


            back_s = time.perf_counter()
            grads = pickle.loads(to_get)

            self.optimizer.zero_grad()
            bottom_output.backward(grads)
            self.optimizer.step()
            
            if epoch % 1 == 0:
                torch.save(self.bottom_model.state_dict(),
                           "ADD_by_yourself/FaaS_FL2023/VFL_faas/model_save/MLP/bottom_model_epoch{0}_client_{1}.pth".format(epoch, self.client_rank))
                
            back_e = time.perf_counter()
            self.bottom_commu_num += 1
            self.comp_time += div_data_st - div_data_ed + back_e - back_s + forwadr_e - forwadr_s
            #if rnd == 10:
            rnd10_time += time.time() - time_2
            #print(f"self.bottom_model.state_dict(): \n{self.bottom_model.state_dict()}")
            #print(f"rnd10_time: {rnd10_time}")

            rnd += 1

    def launch(self):
        epoch = self.args.epoch
        for epo in tqdm(range(1, epoch+1)):
            self.one_epoch_bottom_fp_bp(epo)
            

