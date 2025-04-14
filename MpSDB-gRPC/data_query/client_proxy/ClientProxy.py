from math import fabs
import os
import transmission.tenseal.tenseal_client_proxy_pb2 as tenseal_client_proxy_pb2
import transmission.tenseal.tenseal_client_proxy_pb2_grpc as tenseal_client_proxy_pb2_grpc
import transmission.tenseal.tenseal_parse_server_pb2 as tenseal_parse_server_pb2
import time
import pickle
from data_query.client_proxy.utils import *
import psutil

class ClientProxy(tenseal_client_proxy_pb2_grpc.ClientProxyServiceServicer):
    def __init__(self, parse_server_address, address):
        self.parse_server_address = parse_server_address
        self.address = address
        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        self.sleep_time = 0.1

        self.result_list = []
        print("start")


    def data_query(self, request, context):
        try:
            # 通过读取 /proc 文件系统获取当前进程所在的 CPU 核心
            pid = os.getpid()
            stat_path = f"/proc/{pid}/stat"
            with open(stat_path, 'r') as stat_file:
                stat_content = stat_file.read()
                stat_list = stat_content.split()
                # 第 38 个字段是进程运行的最后一个 CPU 核心的索引
                last_cpu_core = int(stat_list[38])
                # 第 39 个字段是进程运行过的所有 CPU 核心的掩码
                cpu_mask = int(stat_list[39])
                # 将掩码转换为包含 1 的位表示进程运行过的 CPU 核心
                cpu_cores = [i for i, bit in enumerate(bin(cpu_mask)[:1:-1]) if bit == '1']
                #return last_cpu_core
        except Exception as e:
            print(f"Error getting current process CPU core: {e}")
            return None
        # 写入到文件中
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/cpu_clientProxy.txt", "a+") as file:
            file.write(f"cid_: {request.cid} is running on CPU cores: {last_cpu_core}   all:{cpu_cores}\n")   

         #get parse_server stub and send request query to parse_server
        print(f"\n\n\nClientProxy start\n\n\n")
        parse_server_stub = get_parse_server_stub(self.parse_server_address, self.options)
        request_query = tenseal_parse_server_pb2.query_msg_client_proxy(cid=request.cid, qid=request.qid,
                                                                        db_name=request.db_name,
                                                                        column_name=request.column_name, op=request.op,
                                                                        table_name=request.table_name,
                                                                        ip_address=self.address)
        parse_server_stub.parse_request(request_query)

        while not boolean_find_result(request.cid, request.qid, self.result_list):
            time.sleep(self.sleep_time)
        result = get_result(request.cid, request.qid, self.result_list)
        # print(f"clientproxy:\n{result}\n\n\n\n\n\n")
        # raise
        if request.op.upper() == "COUNT":
            result = [round(x) for x in result]

        serialize_msg = pickle.dumps(result)
        response = tenseal_client_proxy_pb2.dec_query_result(dec_result=serialize_msg)

       # a,b,pi,tmp,i = 1,1,0,1,1
        # while (fabs(tmp) >= pow(10,-6)): #计算Pi
        #     pi += tmp
        #     b += 2
        #     a = -a
        #     tmp = a/b
        #     i += 2   
        # serialize_msg = pickle.dumps(1)
        # response = tenseal_client_proxy_pb2.dec_query_result(dec_result=serialize_msg)
        # print("end")
        return response


    def return_dec_query_result(self, request, context):
        cid = request.cid
        qid = request.qid
        serialize_msg = request.dec_result
        result = pickle.loads(serialize_msg)
        result_dict = {'cid': cid, 'qid': qid, 'result': result}
        self.result_list.append(result_dict)

        response = tenseal_client_proxy_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

        return response
