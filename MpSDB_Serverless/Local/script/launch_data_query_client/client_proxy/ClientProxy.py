import json
import transmission.tenseal.tenseal_client_proxy_pb2 as tenseal_client_proxy_pb2
import transmission.tenseal.tenseal_client_proxy_pb2_grpc as tenseal_client_proxy_pb2_grpc
import transmission.tenseal.tenseal_parse_server_pb2 as tenseal_parse_server_pb2
import time as t
import pickle
from data_query.client_proxy.utils import *
import oss2

class ClientProxy(tenseal_client_proxy_pb2_grpc.ClientProxyServiceServicer):
    def __init__(self, parse_server_address, address):
        self.parse_server_address = parse_server_address
        self.address = address
        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        self.sleep_time = 0.1

        self.result_list = []

    def read_ram_user_info(self, file_path):
        with open(file_path, 'r') as file:
            ram_user_info = json.load(file)
        return ram_user_info

    def data_query(self, request, context):
        client_to_parse_list = {}
        client_to_parse_list["cid"] = request.cid
        client_to_parse_list["qid"] = request.qid
        client_to_parse_list["column_name"] = request.column_name
        client_to_parse_list["op"] = request.op
        client_to_parse_list["table_name"] = request.table_name
        
        client_to_parse_list["is_total"] = request.db_name
        client_to_parse_list["db_name"] = "osm_a_"#"conclave_db_3_3_" #"osm_a_"
        client_to_parse_list["ip_address"] = self.address
        client_to_parse_list["client_index_list"] = [0, 1 ,2]
        client_to_parse_list["query_name"] = "normal"
        print(client_to_parse_list)
        client_to_parse_list_d = pickle.dumps(client_to_parse_list)
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        admin_bucket = oss2.Bucket(auth, "ADD_by_yourself", "fc-computes") 


        file_path = 'ADD_by_yourself/FaaS_FL2023_async/faas_fed/set_ram/key/query_user_' + str(client_to_parse_list["cid"]) + '.txt'  
        ram_user_info = self.read_ram_user_info(file_path)
        # 提取子 RAM 用户信息
        new_ram_access_key_id = ram_user_info.get('key-id')
        new_ram_access_key_secret = ram_user_info.get('key-secret')
        user_auth = oss2.Auth(new_ram_access_key_id, new_ram_access_key_secret)
        bucket = oss2.Bucket(auth, "ADD_by_yourself", "clients-up") 
        print(new_ram_access_key_id)

        #清理计算空间
        for obj in oss2.ObjectIterator(admin_bucket, prefix='query_user_' + str(client_to_parse_list["cid"]) + '/'):
            print(f"delete {obj.key}")
            admin_bucket.delete_object(obj.key)

        print('query_user_' + str(client_to_parse_list["cid"]) + '/')
        #清理用户空间
        #t = bucket.get_object("query_user_1/query_client_to_parse.txt").read()
        # t = bucket.get_object("in1.csv").read()
        # print(t)
        # print("\nstart\n")
        # for i in bucket.list_objects("query_user_1/"):
        #     print(i.key)
        for obj in oss2.ObjectIterator(bucket, prefix='query_user_' + str(client_to_parse_list["cid"]) + '/'):
            print(f"delete {obj.key}")
            bucket.delete_object(obj.key)  

        bucket.put_object("query_user_"+str(client_to_parse_list["cid"])+"/query_client_to_parse.txt",client_to_parse_list_d)

        put_time = round(t.time(),5)

        while bucket.object_exists("query_user_" + str(client_to_parse_list["cid"]) + "/key_server_to_client.txt") == False:
            t.sleep(0.001)
        get_time = round(t.time(),5)
        
        print(f"test_time_put_to_parse: {put_time}")
        print(f"test_time_get_from_keyserver: {get_time}")
        
        ans_dict = pickle.loads(bucket.get_object("query_user_" + str(client_to_parse_list["cid"]) + "/key_server_to_client.txt").read())
        if request.cid == ans_dict["cid"] and request.qid == ans_dict["qid"]:
            ans = ans_dict["result"]    
        print(type(ans))  
        response = tenseal_client_proxy_pb2.dec_query_result(dec_result=ans)
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
