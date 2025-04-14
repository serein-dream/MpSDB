import logging
import transmission.tenseal.tenseal_parse_server_pb2 as tenseal_parse_server_pb2
import transmission.tenseal.tenseal_parse_server_pb2_grpc as tenseal_parse_server_pb2_grpc
from .utils import *
from .parsing import *
import time as t

class ParseServer(tenseal_parse_server_pb2_grpc.ParseServerServiceServicer):

    def __init__(self, address_dict, pk_ctx_file):
        self.address_dict = address_dict
        print(f"\n\n\n{self.address_dict}\n\n\n")

        self.sleep_time = 0.1
        pk_bytes = open(pk_ctx_file, "rb").read()
        self.pk_ctx = ts.context_from(pk_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        

    def parse_request(self, request, context):
        st = t.time()
        print(f"\n\n\nParse start\n\n\n")
        logging.basicConfig(filename='/home/maxvyang01/FaaS_FL2023/SecureDatabase/data_query/parse_server/t.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.info(f"cid: {request.cid} start")
        enc_vector = request_parsing(request, self.pk_ctx, self.address_dict, self.options)
        get_ec_t = t.time()
        key_server_stub = get_key_server_stub(self.address_dict, self.options)
        return_results(key_server_stub, enc_vector, request)
        return_enc_t = t.time()
        logging.info(f"cid: {request.cid} end")
        with open('/home/maxvyang01/FaaS_FL2023/SecureDatabase/data_query/parse_server/test_if_multi.txt','a+') as w:
            w.write(f"st_time_cid{request.cid}:{st}\nget_ec_time_cid{request.cid}:{get_ec_t}\nreturn_enc_time_cid{request.cid}:{return_enc_t}\n\n")
        response = tenseal_parse_server_pb2.google_dot_protobuf_dot_empty__pb2.Empty()
        return response
