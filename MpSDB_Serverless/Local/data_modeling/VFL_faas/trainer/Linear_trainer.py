import torch.nn
from model.Linear import BottomModel
import torch.optim as optim
from .base_trainer import BaseTrainer
from tqdm import tqdm
import pickle
import oss2
import tenseal as ts
import time

class LinearTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.bottom_model = BottomModel(n_f)
        #self.bottom_model.load_state_dict(torch.load(f'ADD_by_yourself/FaaS_FL2023/VFL/model_save_20240423ok/Linear_cry/bottom_model_epoch70_client_{client_rank}.pth'))
        self.bottom_model.to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=self.args.lr)
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        self.bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
        self.wait_time = 0
        self.bottom_commu_time = 0
        self.bottom_commu_num = 0
        self.comp_time = 0

    def one_epoch_bottom_fp_bp(self, epoch):
        div_data_st = time.perf_counter()
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,
                                                  shuffle=False, num_workers=0)
        #print(f"\n\n data_loader.shape:  {len(data_loader)}\n\n")
        rnd = 1
        #print(f"train_list type : {type(self.train_list[0])}")#torch.Tensor
        clients_num_d = pickle.dumps(self.args.num_users)
        div_data_ed = time.perf_counter()
        
        start_wait_t = time.time()       
        self.bucket.put_object("num_clients.txt", clients_num_d)
        end_wait_t = time.time()   
        self.wait_time = self.wait_time + end_wait_t - start_wait_t   
        print(f"len(data_loader) : {len(data_loader)}")        
        for batch in data_loader:
            idx, train_x = batch
            forwadr_s = time.perf_counter()
            train_x = train_x.to(self.device)
            bottom_output = self.bottom_model(train_x)
            #print(f"rank {self.client_rank} bottom_output:\n{bottom_output}")
            if "CRY" in self.args.model.upper():
                besent_bottom_output = bottom_output
                besent_bottom_output = besent_bottom_output.detach().cpu().numpy()
                besent_bottom_output = self.client.encry_np(besent_bottom_output)
            else:
                besent_bottom_output = bottom_output

            bottom_to_server_dict = {}
            bottom_to_server_dict["bottom_output"] = besent_bottom_output
            bottom_to_server_dict["epoch"] = epoch
            bottom_to_server_dict["rnd"] = rnd
            bottom_to_server_dict["args"] = self.args
            bottom_to_server_dict_d = pickle.dumps(bottom_to_server_dict)  
            forwadr_e = time.perf_counter()
            #time.sleep(2)
            start_wait_t = time.time()   
            name = "vfl_from_bottom/rank_"+ str(self.args.rank) +".txt"
            self.bucket.put_object(name, bottom_to_server_dict_d)

            name = "to_bottom_grad/safe_"+ str(self.args.rank) +".txt"  
            while self.bucket.object_exists(name) == False:
                time.sleep(0.05) 
            self.bucket.delete_object(name)
            name = "to_bottom_grad/rank_"+ str(self.args.rank) +".txt"   
            while self.bucket.object_exists(name) == False:
                time.sleep(0.05)
            end_wait_t = time.time()   
            self.wait_time = self.wait_time + end_wait_t - start_wait_t 

            print(f"epoch: {epoch}  rnd:  {rnd}  get middle_output_bp grad")
            
            if "CRY" in self.args.model.upper():
                
                start_wait_t = time.time()  
                grads = self.bucket.get_object(name).read()               
                self.bucket.delete_object(name)
                end_wait_t = time.time()   
                self.wait_time = self.wait_time + end_wait_t - start_wait_t   

                back_s = time.perf_counter()
                grads = self.client.decry_np(grads)
                # print(f":grads:\n{grads}")
                # print(f"\n\n\n")       
                grads = torch.tensor(grads).to(self.device)
                grads = grads.view(grads.shape[0],1)  
                # print(f":grads:\n{grads}")
                # raise          
            else:
                
                start_wait_t = time.time()   
                grads_S = self.bucket.get_object(name).read()               
                self.bucket.delete_object(name)
                end_wait_t = time.time()   
                self.wait_time = self.wait_time + end_wait_t - start_wait_t  

                back_s = time.perf_counter()

                grads = pickle.loads(grads_S)
                # print(f":grads:\n{grads}")
                # raise          

                
            self.optimizer.zero_grad()
            bottom_output.backward(grads)
            self.optimizer.step()
            # print(rnd, idx)
            rnd += 1
            
            p = self.bottom_model.state_dict()
            print(f"epo:{epoch}  params: \n{p}")
            if epoch % 1 == 0:
                torch.save(self.bottom_model.state_dict(),
                           "ADD_by_yourself/FaaS_FL2023/VFL_faas/model_save/Linear/bottom_model_epoch{0}_client_{1}.pth".format(epoch, self.client_rank,self.args.model))
                
            back_e = time.perf_counter()
            self.comp_time += div_data_st - div_data_ed + back_e - back_s + forwadr_e - forwadr_s
        # print("bias",self.bottom_model.fc1.bias)

    def launch(self):
        epoch = self.args.epoch

        for epo in tqdm(range(1, epoch+1)):
            self.one_epoch_bottom_fp_bp(epo)
        
