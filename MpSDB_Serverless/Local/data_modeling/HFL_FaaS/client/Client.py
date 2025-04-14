import time

import grpc
import numpy as np
import tenseal as ts
import torch
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transmission.utils import flatten_tensors, unflatten_tensors
import pickle
import oss2

class Client:

    def __init__(self, client_rank, sample_num, ctx_file):
        self.client_rank = client_rank
        self.sample_num = sample_num
        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

    def enc_dec_inclient(self, the_params):
        #latest_model_params_s = torch.tensor(the_params).serialize()
        enc_init_params = ts.ckks_vector_from(self.ctx,the_params)
        dec_init_params = enc_init_params.decrypt()
        init_params_tensor = torch.tensor(dec_init_params)#lr_trainer.is_update里的
        return init_params_tensor

    def __list_to_numpy(self, the_list, n_dims):       
        temp1=[]
        temp2=[]
        for i in range(len(the_list)):
            temp2.append(the_list[i])
            if i>0 and (i+1)%n_dims==0:
                np_point = np.array(temp2)
                temp1.append(np_point)
                temp2 = []
        ans_numpy = np.array(temp1)
        return ans_numpy
    
    def dec_kmeans_tonumpy(self, enc_c,n_dims):
        enc_init_centroids = ts.ckks_vector_from(self.ctx,enc_c)
        dec_init_centroids = enc_init_centroids.decrypt()
        dec_init_centroids = self.__list_to_numpy(dec_init_centroids, n_dims)
        return dec_init_centroids
    
    def update_params_serverless(self,args):
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
        params_msg_dumps = bucket.get_object('global_params.txt').read()
        params_msg = pickle.loads(params_msg_dumps)
        enc_init_params = ts.ckks_vector_from(self.ctx,params_msg)
        dec_init_params = enc_init_params.decrypt()
        return dec_init_params
        #return params_msg
        
    def enc_tensor(self,flat_tensor):
        import warnings

        # 忽略特定级别的警告
        warnings.filterwarnings("ignore")

        enc_vector = ts.ckks_vector(self.ctx, flat_tensor)
        return enc_vector

    def dec_aggregated_params_client(self,average_params):
        summed_encrypted_vector = ts.ckks_vector_from(self.ctx, average_params)
        summed_plain_vector = summed_encrypted_vector.decrypt()
        return summed_plain_vector
    
    if __name__ == '__main__':
        serv_address = "127.0.0.1:59000"
    '''
    def __init__(self, server_address, client_rank, sample_num, ctx_file):
        self.server_address = server_address
        self.client_rank = client_rank
        self.sample_num = sample_num
        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = tenseal_aggregate_server_pb2_grpc.AggregationServerServiceStub(channel)

    def __sum_encrypted(self, plain_vector):
        # print(">>> client sum encrypted start")

        # encrypt
        encrypt_start = time.time()
        enc_vector = ts.ckks_vector(self.ctx, plain_vector)
        encrypt_time = time.time() - encrypt_start

        # print("size of msg: {} bytes".format(sys.getsizeof(enc_vector.serialize())))

        # create request
        request_start = time.time()
        request = tenseal_aggregate_server_pb2.local_params(
            client_rank=self.client_rank,
            sample_num=self.sample_num,
            params_msg=enc_vector.serialize()
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        # print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.sum_encrypted(request)
        comm_time = time.time() - comm_start

        # deserialize summed vector from response
        deserialize_start = time.time()
        assert self.client_rank == response.client_rank
        summed_encrypted_vector = ts.ckks_vector_from(self.ctx, response.params_msg)
        deserialize_time = time.time() - deserialize_start

        # decrypt vector
        decrypt_start = time.time()
        summed_plain_vector = summed_encrypted_vector.decrypt()
        decrypt_time = time.time() - decrypt_start

        # print(">>> client sum encrypted end, cost {:.2f} s: encryption {:.2f} s, create request {:.2f} s, "
        #       "comm with server {:.2f} s, deserialize {:.2f} s, decryption {:.2f} s"
        #       .format(time.time() - encrypt_start, encrypt_time, request_time,
        #               comm_time, deserialize_time, decrypt_time))

        return summed_plain_vector

    def __is_update(self):
        request = tenseal_aggregate_server_pb2.update_request(client_rank=self.client_rank,sample_num=self.sample_num)
        response = self.stub.boolean_is_update(request)
        update_flag = response.flag
        enc_init_params = ts.ckks_vector_from(self.ctx,response.params_msg)
        dec_init_params = enc_init_params.decrypt()
        return update_flag, dec_init_params



    def transmit(self, args, params_list, operator="sum"):
        trans_start = time.time()
        # received:list, received tensors convert received to tensors
        received = None
        # print(">>> client transmission cost {:.2f} s".format(time.time() - trans_start))
        if operator == "sum":
            received = self.__sum_encrypted(params_list)
        elif operator == "update_params":
            #received = self.__is_update()
            received = self.update_params_serverless(args)
        return received


if __name__ == '__main__':
    serv_address = "127.0.0.1:59000"
    # ctx_file = "../../transmission/ts_ckks.config"
    # client_rank = 0
    #
    # client = Client(serv_address, client_rank, ctx_file)
'''