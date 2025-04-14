import time
import grpc
import pickle
import transmission.pickle.aggregate_server_pb2 as aggregate_server_pb2
import transmission.pickle.aggregate_server_pb2_grpc as aggregate_server_pb2_grpc
import torch
from transmission.utils import flatten_tensors, unflatten_tensors
import tenseal as ts


class Client:

    def __init__(self, server_address, client_rank):
        self.server_address = server_address
        self.client_rank = client_rank
        ctx_file = "ADD_by_yourself/transmission/ts_ckks.config"
        print(f"ctx:{ctx_file}")
        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = aggregate_server_pb2_grpc.AggregateServerServiceStub(channel)

    def encry_np(self,to_encry_np):
        to_encry_tensor = torch.tensor(to_encry_np)
        to_encry_flatten_tensor = flatten_tensors(to_encry_tensor).detach()
        encry_flatten_tensor = ts.ckks_vector(self.ctx, to_encry_flatten_tensor)
        encry_flatten_tensor_s = encry_flatten_tensor.serialize()
        return encry_flatten_tensor_s

    def decry_np(self, encry_tensor_s):        
        encry_tensor = ts.ckks_vector_from(self.ctx, encry_tensor_s)
        decry_tensor = encry_tensor.decrypt()
        return decry_tensor
    
    def __bottom_dumps(self, plain_vector, epoch, rnd):
        request_start = time.time()
        request = aggregate_server_pb2.bottom_output(
            client_rank=self.client_rank,
            round=rnd,
            epoch=epoch,
            params_msg=pickle.dumps(plain_vector)
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        # print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.middle_fp_bp(request)
        comm_time = time.time() - comm_start

        # load grad tensor

        assert self.client_rank == response.client_rank

        bottom_grad_vector = pickle.loads(response.grad_msg)
        return bottom_grad_vector

    def transmit(self, plain_vector, epoch, rnd):
        trans_start = time.time()
        # received:list, received tensors convert received to tensors
        # print(">>> client transmission cost {:.2f} s".format(time.time() - trans_start))

        received = self.__bottom_dumps(plain_vector, epoch, rnd)

        return received
    def __bottom_cry_s(self, plain_cey_s, epoch, rnd):
        # print(">>> client bottom transmit start")

        # print("size of msg: {} bytes".format(sys.getsizeof(enc_vector.serialize())))

        # create request
        request_start = time.time()
        request = aggregate_server_pb2.bottom_output(
            client_rank=self.client_rank,
            round=rnd,
            epoch=epoch,
            params_msg=plain_cey_s
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        # print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.middle_fp_bp(request)
        comm_time = time.time() - comm_start

        # load grad tensor

        assert self.client_rank == response.client_rank

        bottom_grad_vector = self.decry_np(response.grad_msg)
        #print(type(bottom_grad_vector))
        return bottom_grad_vector

    def transmit_cry(self, plain_vector, epoch, rnd):
        trans_start = time.time()
        # received:list, received tensors convert received to tensors
        # print(">>> client transmission cost {:.2f} s".format(time.time() - trans_start))
        
        received = self.__bottom_cry_s(plain_vector, epoch, rnd)

        return received

if __name__ == '__main__':
    serv_address = "127.0.0.1:59000"
    # ctx_file = "../../transmission/ts_ckks.config"
    # client_rank = 0
    #
    # client = Client(serv_address, client_rank, ctx_file)
