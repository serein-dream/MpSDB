import torch.nn
from model.Logistic import BottomModel
import torch.optim as optim
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from tqdm import tqdm
import pickle
import oss2
import time


class LogisticTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.bottom_model = BottomModel(n_f)
        self.bottom_model.to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=self.args.lr)
        #self.test_dataset = test_dataset
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        self.bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")

    def one_epoch_bottom_fp_bp(self, epoch):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,
                                                  shuffle=False, num_workers=0)
        #print(f"\n\n data_loader.shape:  {len(data_loader)}\n\n")
        rnd = 1
        #print(f"train_list type : {type(self.train_list[0])}")#torch.Tensor
        clients_num_d = pickle.dumps(self.args.num_users)
        self.bucket.put_object("num_clients.txt", clients_num_d)      
        for batch in data_loader:
            #start_time = time.time()
            #rnd10_time = 0
            print(f"epoch: {epoch}  rnd:  {rnd}  start")
            idx, train_x = batch
            train_x = train_x.to(self.device)
            bottom_output = self.bottom_model(train_x)
            #bottom_output_np = bottom_output.detach().numpy()
            
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
            name = "vfl_from_bottom/rank_"+ str(self.args.rank) +".txt"
            self.bucket.put_object(name, bottom_to_server_dict_d)

            name = "to_bottom_grad/rank_"+ str(self.args.rank) +".txt"
            #time_1 = time.time()
            #rnd10_time += time_1 - start_time         
            while self.bucket.object_exists(name) == False:
                time.sleep(0.05)
            #time_2 = time.time()
            #grads = torch.tensor(grads).reshape(-1, 1)      
            if "CRY" in self.args.model.upper():
                grads = self.bucket.get_object(name).read()
                grads = self.client.decry_np(grads)
                #print(f"grads:{grads.shape}")
                grads = torch.tensor(grads).to(self.device)
                grads = grads.view(grads.shape[0],1)             
            else:
                grads = pickle.loads(self.bucket.get_object(name).read())
            #print(f"grads shape:{grads.shape}")
            self.bucket.delete_object(name)
            self.optimizer.zero_grad()
            bottom_output.backward(grads)
            #print(f"grads:  {grads}")#000000
            self.optimizer.step()

            # print(rnd, idx)
            rnd += 1
        # print("bias",self.bottom_model.fc1.bias)

    def launch(self):
        epoch = self.args.epoch

        for epo in tqdm(range(1, epoch+1)):
            self.one_epoch_bottom_fp_bp(epo)
            if epo % 1 == 0:
                torch.save(self.bottom_model.state_dict(),
                           "ADD_by_yourself/FaaS_FL2023/VFL_faas/model_save/{2}/bottom_model_epoch{0}_client_{1}.pth".format(epo, self.client_rank,self.args.model))

