import time

import grpc
import numpy as np
import tenseal as ts
import torch
import sys
from transmission.tenseal import tenseal_aggregate_server_pb2, tenseal_aggregate_server_pb2_grpc
from transmission.utils import flatten_tensors, unflatten_tensors
import time
import pickle
import oss2

class Client:

    def __init__(self, server_address, client_rank, sample_num, ctx_file):
        self.server_address = server_address
        print(f"server_address:{server_address}")
        self.client_rank = client_rank
        self.sample_num = sample_num
        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = tenseal_aggregate_server_pb2_grpc.AggregateServerServiceStub(channel)

    def __list_to_numpy(self, the_list,n_dims):       
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
        s = time.perf_counter()
        # print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.sum_encrypted(request)
        com_time = time.perf_counter() - s
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

        return summed_plain_vector,com_time
    
    def sum_encrypted_kmeans(self, rank,clients_local_centroids_expanded,client_counts, n_dims):
        enc_clients_local_centroids_expanded = ts.ckks_vector(self.ctx, clients_local_centroids_expanded)
        #print(f"clients_updates:   {type(clients_updates)}")
        enc_clients_local_centroids_expanded = enc_clients_local_centroids_expanded.serialize()
        client_counts = torch.tensor(client_counts)
        #print(type(client_counts))
        client_counts_d = pickle.dumps(client_counts)
        request = tenseal_aggregate_server_pb2.local_params_kmeans(
            client_rank=rank,
            clients_updates=enc_clients_local_centroids_expanded,
            client_counts=client_counts_d
        )       
        response = self.stub.sum_encrypted_kmeans(request)
        #assert self.client_rank == response.client_rank
        #not_change_number = response.not_change_number
        latest_centroids = response.params_msg#pickle.loads(response.params_msg)
        total_count = pickle.loads(response.total_count)
        encrypt_latest_centroids = ts.ckks_vector_from(self.ctx, latest_centroids)
        latest_centroids = encrypt_latest_centroids.decrypt()
        latest_centroids = self.__list_to_numpy(latest_centroids, n_dims)
        latest_centroids = latest_centroids / np.expand_dims(np.maximum(total_count, np.ones_like(total_count)), axis=1)
        #print(f"latest_centroids:\n   {latest_centroids}")
        return  latest_centroids 

    def __is_update(self):
        print(f"1\nself.client_rank:{self.client_rank}\nself.sample_num:{self.sample_num}")
        request = tenseal_aggregate_server_pb2.update_request(client_rank=self.client_rank,
                                                              sample_num=self.sample_num)
        s = time.perf_counter()
        response = self.stub.boolean_is_update(request)
        com_time = time.perf_counter() - s
        update_flag = response.flag
        enc_init_params = ts.ckks_vector_from(self.ctx,response.global_centroids_msg)#也指cnn参数
        dec_init_params = enc_init_params.decrypt()
        
        return update_flag, dec_init_params,com_time

    def is_update_kmeans(self,not_change_number, n_dim):
        #print(f"1\nself.client_rank:{self.client_rank}\nself.sample_num:{self.sample_num}")
        request = tenseal_aggregate_server_pb2.update_request_kmeans(client_rank=self.client_rank,not_change_number = not_change_number)
        s = time.perf_counter()
        response = self.stub.boolean_is_update_kmeans(request)
        com_time = time.perf_counter() - s
        update_flag = response.flag
        to_stop = response.to_stop
        latest_centroids = response.global_centroids_msg

        #print(f"latest_centroids:\n  {latest_centroids}")
        enc_latest_centroids = ts.ckks_vector_from(self.ctx,latest_centroids)
        dec_latest_centroids = enc_latest_centroids.decrypt()
        dec_latest_centroids = self.__list_to_numpy(dec_latest_centroids, n_dim)
        #print(f"dec_latest_centroids type:   {type(dec_latest_centroids)}")
        #enc_init_params = ts.ckks_vector_from(self.ctx,response.params_msg)
        #dec_init_params = enc_init_params.decrypt()
        #print(f"latest_centroids:\n{latest_centroids}")
        return dec_latest_centroids, update_flag, com_time, to_stop
    
    def dec_kmeans(self,enc_c, n_dims):
        enc_init_centroids = ts.ckks_vector_from(self.ctx,enc_c)
        dec_init_centroids = enc_init_centroids.decrypt()
        dec_init_centroids = self.__list_to_numpy(dec_init_centroids, n_dims)
        return dec_init_centroids


    def transmit(self, params_list, operator="sum"):
        trans_start = time.time()
        # received:list, received tensors convert received to tensors
        received = None
        # print(">>> client transmission cost {:.2f} s".format(time.time() - trans_start))
        if operator == "sum":
            received = self.__sum_encrypted(params_list)
        elif operator == "update_flag":
            received = self.__is_update()
        return received


if __name__ == '__main__':
    serv_address = "127.0.0.1:59000"
    # ctx_file = "../../transmission/ts_ckks.config"
    # client_rank = 0
    #
    # client = Client(serv_address, client_rank, ctx_file)
