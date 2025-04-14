import time
from typing import Set, Dict, Tuple, List

import transmission.tenseal.tenseal_data_server_pb2_grpc as tenseal_data_server_pb2_grpc
import transmission.tenseal.tenseal_data_server_pb2 as tenseal_data_server_pb2
import transmission.tenseal.tenseal_key_server_pb2_grpc as tenseal_key_server_pb2_grpc
import tenseal as ts
from data_query.data_server.conn_mysql import *
import pickle
import grpc
import random
# ip_list = ["rm-bp1q7a1790598nd618o.rwlb.rds.aliyuncs.com",
#             "rm-bp1994805363jn6380o.rwlb.rds.aliyuncs.com",
#             "rm-bp1ys96h6t5lr9fl6uo.rwlb.rds.aliyuncs.com",
#             "rm-bp13ck94u1urshldy4o.rwlb.rds.aliyuncs.com",
#             "rm-bp169d2941snz114wfo.rwlb.rds.aliyuncs.com",
#             "rm-bp1577i0wdv93uhpfyo.rwlb.rds.aliyuncs.com",
#             "rm-bp1n0www1bra9p2t2vo.rwlb.rds.aliyuncs.com",
#             "rm-bp157ziifc8s97vu0wo.rwlb.rds.aliyuncs.com",
#             "rm-bp1o9r196ov3x7ww4wo.rwlb.rds.aliyuncs.com",
#             "rm-bp19be280yt3bh13avo.rwlb.rds.aliyuncs.com"
#             ]

ip_list = [
    "rm-bp173yd59tmo8go7bso.rwlb.rds.aliyuncs.com",
    "rm-bp1f7lt9t18s897j89o.rwlb.rds.aliyuncs.com",
    "rm-bp1672r92kx1977h86o.rwlb.rds.aliyuncs.com",
    "rm-bp1uw8q6x3856v07deo.rwlb.rds.aliyuncs.com",
    "",
    "",
    "",
    "rm-bp18r203w093jq47j5o.rwlb.rds.aliyuncs.com",
    "rm-bp1s13zu3h9o1354e7o.rwlb.rds.aliyuncs.com",
    "rm-bp1exm5746ci6o6r7go.rwlb.rds.aliyuncs.com",
]

class DatabaseServer(tenseal_data_server_pb2_grpc.DatabaseServerServiceServicer):

    def __init__(self, key_server_address, pk_ctx_file, sk_ctx_file, db_name, cfg):
        self.ks_address = key_server_address
        pk_ctx_file = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/transmission/ts_ckks_pk.config"
        sk_ctx_file = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/transmission/ts_ckks.config"
        pk_bytes = open(pk_ctx_file, "rb").read()
        self.pk_ctx = ts.context_from(pk_bytes)
        ctx_byte = open(sk_ctx_file, "rb").read()
        self.sk_ctx = ts.context_from(ctx_byte)
        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        #print(db_name[-1])
        self.db_id = int(db_name[-1])
        if "500w" in db_name or "osm_400w" in db_name or "vertical" in db_name or "lmis" in db_name:
            self.db_ip = ip_list[self.db_id+1]      #可以在oss中读取
        elif "taxi" in db_name:
                self.db_ip = ip_list[self.db_id+6]
        else:
            self.db_ip = ip_list[self.db_id]      #可以在oss中读取
        self.sleep_time = 0.01

        if "imis_400w" in db_name:
            self.ip = ip_list[self.db_id]
            print(f"self.ip:{self.db_id}")

        if "wine" in db_name:
            self.database_name = "wine_v"
            self.table_name = db_name
            print(f"\n\n\nini  table: {self.table_name}\n\n\n")
        elif "osm" in db_name or "lmis" in db_name:
            self.database_name = "osm_db"
            self.table_name = db_name
            if "osm_400w" in db_name:
                self.table_name = "osm_400w"
            else:
                self.table_name = db_name  
        elif "vertical" in db_name:
            self.database_name = "osm_db"
            self.table_name = db_name[:-1]+str(int(db_name[-1])+1)  
        elif "taxi" in db_name:
            self.database_name = "database_1"
            self.table_name = db_name#后面从参数接收
        else:
            self.database_name = db_name
            self.table_name = "table_1"

        if "imis_400w" in db_name:
            if self.db_id == 8:
                self.database_name = "database_1"
            else:
                self.database_name = "osm_db"
            self.table_name = db_name[:-1] + str(int(db_name[-1])+1)
        if "tpch" in db_name:
            self.database_name = "osm_db"
        self.cfg = cfg
        self.n_th_cache: Dict[int,Tuple[int,float]] = {}
        self.hash_cache: Dict[int,Tuple[int,float]] = {}
        print(f"db:  {self.database_name}  ip:{self.db_ip} table_name:{self.table_name}\n")
        # raise

    def query_DWithin(self, request, context):
        st = time.time()
        if "lmis_" in request.table_name:
            self.table_name = request.table_name+str(self.db_id-1)
        else:
            self.table_name = request.table_name+str(self.db_id)
        print(f"self.table_name:{self.table_name}\nself.db_ip:{self.db_ip}\n")
        # sql, k_or_r = generate_sql(request.table_name+str(self.db_id), request)
        sql, k_or_r = generate_sql(self.table_name, request)
  
        st = time.time() 
        query_result = get_query_results(self.database_name, self.db_ip, sql) 
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")   
        if request.column_name == "id":
            query_result = np.array(query_result).ravel()   
            query_result = np.append(query_result,float(k_or_r))
        #print(f"here:\ncnt: {query_result}\n\n")
        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        response = tenseal_data_server_pb2.enc_query_result(enc_result=serialize_msg)
        et = time.time()
        print(f"read and trans pk: {round(et-st,4)}")
        return response

    def query_r_by_k(self, request, context):
        greater = request.greater  #表示随机k取大取小
        sql, _ = generate_sql(self.table_name, request)
        st = time.time()       
        print(f"\n\n{sql}\n\n")
        r = get_r_by_k("osm_db", sql, self.db_ip)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")
        if greater == True:
            p = random.uniform(1.01,1.2)
        else:
            p = random.uniform(0.8,0.99)
        r_p = r * p
        serialize_msg = pickle.dumps(r_p)
        response = tenseal_data_server_pb2.query_r_by_k_out(enc_result=serialize_msg)
        return response


    def query_operation(self, request, context):
        self.table_name = request.table_name
        if "taxi" in self.table_name:
            self.db_ip = ip_list[self.db_id+6]
            self.database_name = "database_1"
        else:
            self.db_ip = ip_list[self.db_id]
            self.database_name = "osm_db"
        # print(f"self.table_name:{self.table_name}\nself.database_name:{self.database_name}\nself.db_ip:{self.db_ip}")
        # raise
        sql, _ = generate_sql(self.table_name, request)
        st = time.time()
        query_result = get_query_results(self.database_name, self.db_ip, sql)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")
        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        response = tenseal_data_server_pb2.enc_query_result(enc_result=serialize_msg)
        return response

    def query_operation_vertical(self, request, context):
        self.table_name = request.table_name
        if "taxi" in self.table_name:
            self.db_ip = ip_list[self.db_id+6]
            self.database_name = "database_1"
        else:
            self.db_ip = ip_list[self.db_id]
            self.database_name = "osm_db"

        table_column_name_list = get_column_name(self.database_name, self.table_name, self.db_ip)
        query_vertical = {}
        for table_column_name in table_column_name_list:
            if table_column_name == "id":
                continue
            request_ = request
            request_.column_name = table_column_name
            print(f"\n\n\nrequest_.op:  {request_.op}\n\n\n")
            request_.op = request.op.replace("_vertical", "")
            if "STD" in request.op.upper():
                 request_.op = "variance"
            print(f"\n\n\nrequest_.op2:  {request_.op}\n\n\n")
            request_.table_name = self.table_name
            sql, _ = generate_sql(self.table_name, request_)
            st = time.time()
            query_result = get_query_results(self.database_name, self.db_ip, sql)
            et = time.time()       
            with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
                f.write(f"{round(et-st, 4)}\n")
            plain_vector = ts.plain_tensor(query_result)
            if "STD" in request.op.upper():
                plain_vector = np.sqrt(plain_vector)
            enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
            serialize_msg = enc_vector.serialize()
            query_vertical[table_column_name] = serialize_msg
        response = tenseal_data_server_pb2.enc_query_result(enc_result=pickle.dumps(query_vertical))
        return response

    def noise_query_operation(self, request, context):
        sql, _ = generate_sql(self.table_name, request)
        cid = request.cid
        qid = request.qid
        channel = grpc.insecure_channel(self.ks_address, options=self.options)
        key_stub = tenseal_key_server_pb2_grpc.KeyServerServiceStub(channel)
        st = time.time()    

        query_result = get_noise_query_results(self.database_name, self.cfg, cid, qid, sql, key_stub)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")
        serialize_msg = pickle.dumps(query_result)

        response = tenseal_data_server_pb2.enc_query_result(enc_result=serialize_msg)

        return response

    def n_th_query_operation(self, request, context):
        cid = request.cid
        qid = request.qid
        n = request.n
        mode = request.mode
        table_name = request.table_name
        column_name = request.column_name
        if mode == "clean":
            self.n_th_cache.clear()
            self.hash_cache.clear()
            print("cid: ", cid, " qid: ", qid, " n: ", n, " mode: ", mode)
            sql = "SELECT {1},COUNT(*) AS i FROM {0} GROUP BY {1} ORDER BY i".format( self.database_name + "." +table_name, column_name)
            print(sql)
            st = time.time() 
            query_result = get_query_results(self.database_name, self.db_ip, sql)
            et = time.time()       
            with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
                f.write(f"{round(et-st, 4)}\n")
            print(query_result)
            for i in range(len(query_result)):
                self.n_th_cache[i] = query_result[i]
                self.hash_cache[hash(query_result[i][0] + 0.01)] = query_result[i]

        available = False

        if n in self.n_th_cache:
            available = True
            query_result = [self.n_th_cache[n][1]]
            hash_value = hash(self.n_th_cache[n][0] + 0.01)
        else:
            hash_value = 0
            query_result = [0]

        if n > len(self.n_th_cache):
            available = False

        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        response = tenseal_data_server_pb2.n_th_query_result(cid=request.cid, qid=request.qid,n = n,hash = hash_value ,result = serialize_msg,available = available)
        return response

    def query_from_buffer(self, request, context):
        hash_ = request.hash
        available = False
        if hash_ in self.hash_cache:
            available = True
            query_result = [self.hash_cache[hash_][1]]
        else:
            query_result = [0]

        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()

        response = tenseal_data_server_pb2.query_result_result(result = serialize_msg,available = available)

        return response

    def query_mode_using_hash(self, request, context):
        hash_ = request.hash
        import pickle
        hash_ = pickle.loads(hash_)

        # init available_list with False
        available_list = [False] * len(hash_)

        query_result = [0] * len(hash_)

        print("query_mode_using_hash: ", hash_)

        for i in range(len(hash_)):
            if hash_[i] in self.hash_cache:
                available_list[i] = True
                query_result[i] = self.hash_cache[hash_[i]][0]

        result_list = []

        print("query_mode_using_hash: ", query_result)

        for i in range(len(query_result)):
            plain_vector = ts.plain_tensor([query_result[i]])
            enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
            serialize_msg = enc_vector.serialize()
            result_list.append(serialize_msg)

        return tenseal_data_server_pb2.query_mode_using_hash_result(mode = pickle.dumps(result_list),available = pickle.dumps(available_list))

    def query_median_posi(self, request, context):
        cid = request.cid
        qid = request.qid
        table_name = request.table_name
        column_name = request.column_name
        median = request.median
        enc_vector = ts.ckks_vector_from(self.sk_ctx, median)
        dec_vector = enc_vector.decrypt()
        median = dec_vector[0]
        avg = request.avg
        enc_vector = ts.ckks_vector_from(self.sk_ctx, avg)
        dec_vector = enc_vector.decrypt()
        avg = dec_vector[0]
        std = request.std
        enc_vector = ts.ckks_vector_from(self.sk_ctx, std)
        dec_vector = enc_vector.decrypt()
        std = dec_vector[0]
        sigma3_left = avg - 3 * std
        sigma3_right = avg + 3 * std

        le_sql = "SELECT COUNT(*) FROM {0} WHERE {1} <= {2}".format(self.database_name + "." +table_name, column_name, median)
        st = time.time() 
        le_result = get_query_results(self.database_name, self.db_ip, le_sql)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")

        g_sql = "SELECT COUNT(*) FROM {0} WHERE {1} > {2}".format(self.database_name + "." +table_name, column_name, median)
        st = time.time() 
        g_result = get_query_results(self.database_name, self.db_ip, g_sql)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")
        print("le_result: ", le_result, " g_result: ", g_result,median)
        le_result = ts.plain_tensor(le_result)
        g_result = ts.plain_tensor(g_result)
        le_enc_vector = ts.ckks_vector(self.pk_ctx, le_result)
        g_enc_vector = ts.ckks_vector(self.pk_ctx, g_result)

        msg = tenseal_data_server_pb2.query_median_posi_result(less_e = le_enc_vector.serialize(), greater = g_enc_vector.serialize())
        return msg

    def get_count(self, request, context):
        table_name = request.table_name
        column_name = request.column_name
        sql = "SELECT COUNT(*) FROM {0}".format(self.database_name + "." +table_name)
        st = time.time() 
        query_result = get_query_results(self.database_name, self.db_ip, sql)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")
        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        response = tenseal_data_server_pb2.enc_query_result(enc_result = serialize_msg)
        return response

    def get_nearest(self, request, context):
        table_name = request.table_name
        column_name = request.column_name
        value = request.value
        enc_vector = ts.ckks_vector_from(self.sk_ctx, value)
        dec_vector = enc_vector.decrypt()
        value = dec_vector[0]
        print(f"NearValue: {value}")
        sql = "SELECT {2} FROM {0} ORDER BY ABS({1} - {2}) LIMIT 3;".format(self.database_name + "." +table_name, value , column_name)
        st = time.time() 
        query_result = get_query_results(self.database_name, self.db_ip, sql)
        et = time.time()       
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
            f.write(f"{round(et-st, 4)}\n")
        print("Query: ",query_result)
        if len(query_result) == 0:
            response = tenseal_data_server_pb2.query_nearest_result(value1 = pickle.dumps([0]), value2 = pickle.dumps([0]), value3 = pickle.dumps([0]), count = 0)
        elif len(query_result) == 1:
            plain_vector1 = ts.plain_tensor([query_result[0]])
            enc_vector1 = ts.ckks_vector(self.pk_ctx, plain_vector1)
            response = tenseal_data_server_pb2.query_nearest_result(value1 = enc_vector1.serialize(), value2 = pickle.dumps([0]), value3 = pickle.dumps([0]), count = 1)
        elif len(query_result) == 2:
            plain_vector1 = ts.plain_tensor([query_result[0]])
            enc_vector1 = ts.ckks_vector(self.pk_ctx, plain_vector1)
            plain_vector2 = ts.plain_tensor([query_result[1]])
            enc_vector2 = ts.ckks_vector(self.pk_ctx, plain_vector2)
            response = tenseal_data_server_pb2.query_nearest_result(value1 = enc_vector1.serialize(), value2 = enc_vector2.serialize(), value3 = pickle.dumps([0]), count = 2)
        else:
            plain_vector1 = ts.plain_tensor([query_result[0]])
            enc_vector1 = ts.ckks_vector(self.pk_ctx, plain_vector1)
            plain_vector2 = ts.plain_tensor([query_result[1]])
            enc_vector2 = ts.ckks_vector(self.pk_ctx, plain_vector2)
            plain_vector3 = ts.plain_tensor([query_result[2]])
            enc_vector3 = ts.ckks_vector(self.pk_ctx, plain_vector3)  #?plain_vector3
            response = tenseal_data_server_pb2.query_nearest_result(value1=enc_vector1.serialize(),
                                                                    value2=enc_vector2.serialize(),
                                                                    value3=enc_vector3.serialize(), count=3)  #?count=3
        return response
