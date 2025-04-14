import transmission.tenseal.tenseal_data_server_pb2_grpc as tenseal_data_server_pb2_grpc
import transmission.tenseal.tenseal_data_server_pb2 as tenseal_data_server_pb2
import transmission.tenseal.tenseal_key_server_pb2_grpc as tenseal_key_server_pb2_grpc
import tenseal as ts
from data_query.data_server.conn_mysql import *
import pickle
import grpc
import random
import copy
from typing import Set, Dict, Tuple, List


# from transmission.psi import decode_ids_from_client, encode_and_hash_local_id_use_sk


class DatabaseServer(tenseal_data_server_pb2_grpc.DatabaseServerServiceServicer):

    def __init__(self, key_server_address, pk_ctx_file, sk_ctx_file, db_name, cfg):
        self.ks_address = key_server_address
        pk_bytes = open(pk_ctx_file, "rb").read()
        self.pk_ctx = ts.context_from(pk_bytes)
        ctx_byte = open(sk_ctx_file, "rb").read()
        self.sk_ctx = ts.context_from(ctx_byte)
        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]

        self.sleep_time = 0.01
        # ID PSi
        self.shuffle_seed = random.random()
        # random.seed(self.shuffle_seed)
        self.global_max_id = None
        self.global_min_id = None

        self.table_name = db_name
        self.database_name = "osm_db"
        self.cfg = cfg
        self.n_th_cache: Dict[int,Tuple[int,float]] = {}
        self.hash_cache: Dict[int,Tuple[int,float]] = {}

        # RSA psi
        self.data_server_status = None
        self.rsa_pk = None
        self.rsa_sk = None
        self.rsa_pk_comm_status = False
        self.client_enc_ids_pk = []
        self.client_ra_list = []
        self.client_enc_ids_comm_status = False
        self.client_dec_ids = []
        self.server_hash_enc_ids = []
        self.client_dec_ids_comm_status = False
        self.server_hash_enc_ids_comm_status = False
        self.psi_result = None

    def reset_rsa_psi_status_per_round(self):
        self.rsa_pk = None
        self.rsa_sk = None
        self.rsa_pk_comm_status = False
        self.client_enc_ids_pk = []
        self.client_ra_list = []
        self.client_enc_ids_comm_status = False
        self.client_dec_ids = []
        self.server_hash_enc_ids = []
        self.client_dec_ids_comm_status = False
        self.server_hash_enc_ids_comm_status = False

    def reset_all_rsa_psi_status(self):
        self.data_server_status = None
        self.rsa_pk = None
        self.rsa_sk = None
        self.rsa_pk_comm_status = False
        self.client_enc_ids_pk = []
        self.client_ra_list = []
        self.client_enc_ids_comm_status = False
        self.client_dec_ids = []
        self.server_hash_enc_ids = []
        self.client_dec_ids_comm_status = False
        self.server_hash_enc_ids_comm_status = False
        self.psi_result = None


    def query_DWithin(self, request):
        sql, k_or_r = generate_sql(self.table_name, request)
        query_result = get_query_results(self.database_name, self.cfg, sql)    
        if request["column_name"] == "id":
            query_result = np.array(query_result).ravel()   
            query_result = np.append(query_result,float(k_or_r))
        with open("ADD_by_yourself/FaaS_FL2023/faas_fed/data_query/data_server/log.txt", "+a") as f:
            f.write(f"query: {self.table_name}:  {query_result}\n\n")
            f.write(f"sql:{sql}\nk_or_r:{k_or_r}")
        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        return serialize_msg

    def query_r_by_k(self, request):
        greater = request["greater"]  #表示随机k取大取小
        sql, _ = generate_sql(self.table_name, request)
        r = get_r_by_k(self.database_name, self.cfg, sql)
        if greater == True:
            p = random.uniform(1.01,1.2)
        else:
            p = random.uniform(0.8,0.99)
        r_p = r * p
        dec_vector = r_p
        return dec_vector


    def query_operation(self, request):
        sql, _ = generate_sql(self.table_name, request)
        query_result = get_query_results(self.database_name, self.cfg, sql)#self.name就是db_name
        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        #response = tenseal_data_server_pb2.enc_query_result(enc_result=serialize_msg)
        return serialize_msg

    def noise_query_operation(self, request):
        sql, _ = generate_sql(request)
        cid = request["cid"]
        qid = request["qid"]
        channel = grpc.insecure_channel(self.ks_address, options=self.options)
        key_stub = tenseal_key_server_pb2_grpc.KeyServerServiceStub(channel)

        query_result = get_noise_query_results(self.database_name, self.cfg, cid, qid, sql, key_stub)
        serialize_msg = pickle.dumps(query_result)

        #response = tenseal_data_server_pb2.enc_query_result(enc_result=serialize_msg)

        return serialize_msg

    # ID Psi unencrypted version
    def get_local_max_min_ids(self, id_list):
        return max(id_list), min(id_list)

    def get_shuffled_id_list(self, id_list):
        """
        :param id_list: data id
        :return: origin id list, new index list
        """
        random.seed(self.shuffle_seed)
        mapping_list = []
        origin_list = copy.deepcopy(id_list)
        random.shuffle(id_list)

        for elem in id_list:
            mapping_list.append(elem - self.global_min_id)

        return origin_list, mapping_list

    def get_id_psi_result(self, intersection_list):
        intersection_result = []
        for elem in intersection_list:
            intersection_result.append(self.global_min_id + elem)

        return intersection_result

    def send_rsa_public_key(self, request, context):
        """
        :param request:
        :param context:
        :return: Process status
        """
        cid = request.cid
        qid = request.qid
        pk_N = request.pk_N
        pk_e = request.pk_e
        # recv_status = False

        if pk_N and pk_e:
            self.rsa_pk = (int(pk_N), pk_e)
            self.rsa_pk_comm_status = True
            print("Public Key received.")
            # print(self.rsa_pk)

        response = tenseal_data_server_pb2.rsa_public_key_response(
            cid=cid,
            qid=qid,
            recv_status=self.rsa_pk_comm_status
        )

        return response

    def send_client_enc_ids(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        cid = request.cid
        qid = request.qid
        client_enc_ids_pk_str = request.client_enc_ids_pk_str

        for enc_id_str in client_enc_ids_pk_str:
            self.client_enc_ids_pk.append(int(enc_id_str))

        self.client_enc_ids_comm_status = True
        response = tenseal_data_server_pb2.send_client_enc_ids_response(
            cid=cid,
            qid=qid,
            recv_status=self.client_enc_ids_comm_status
        )

        return response

    def send_server_enc_ids_and_client_dec_ids(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        cid = request.cid
        qid = request.qid
        client_dec_ids = request.client_dec_ids
        server_hash_enc_ids = request.server_hash_enc_ids

        for dec_id in client_dec_ids:
            self.client_dec_ids.append(int(dec_id))
        self.client_dec_ids_comm_status = True

        # for hash_enc_id in server_hash_enc_ids:
        self.server_hash_enc_ids = server_hash_enc_ids
        self.server_hash_enc_ids_comm_status = True

        response = tenseal_data_server_pb2.send_server_enc_ids_and_client_dec_ids_response(
            cid=cid,
            qid=qid,
            client_dec_ids_recv_status=self.client_dec_ids_comm_status,
            server_hash_enc_ids_recv_status=self.server_hash_enc_ids_comm_status
        )

        return response


    def query_median_posi(self, request):
        cid = request["cid"]
        qid = request["qid"]
        table_name = request["table_name"]
        column_name = request["column_name"]
        median = request["median"]
        enc_vector = ts.ckks_vector_from(self.sk_ctx, median)
        dec_vector = enc_vector.decrypt()
        median = dec_vector[0]
        avg = request["avg"]
        enc_vector = ts.ckks_vector_from(self.sk_ctx, avg)
        dec_vector = enc_vector.decrypt()
        avg = dec_vector[0]
        std = request["std"]
        enc_vector = ts.ckks_vector_from(self.sk_ctx, std)
        dec_vector = enc_vector.decrypt()
        std = dec_vector[0]
        sigma3_left = avg - 3 * std
        sigma3_right = avg + 3 * std

        le_sql = "SELECT COUNT(*) FROM {0} WHERE {1} <= {2}".format(self.database_name + "." +table_name, column_name, median)
        le_result = get_query_results(self.database_name, self.cfg, le_sql)

        g_sql = "SELECT COUNT(*) FROM {0} WHERE {1} > {2}".format(self.database_name + "." +table_name, column_name, median)
        g_result = get_query_results(self.database_name, self.cfg, g_sql)
        print("le_result: ", le_result, " g_result: ", g_result,median)
        le_result = ts.plain_tensor(le_result)
        g_result = ts.plain_tensor(g_result)
        le_enc_vector = ts.ckks_vector(self.pk_ctx, le_result)
        g_enc_vector = ts.ckks_vector(self.pk_ctx, g_result)
        ans = {}
        ans["less_e"] = le_enc_vector.serialize()
        ans["greater"] = g_enc_vector.serialize()
        return ans

    def get_nearest(self, request):
        table_name = request["table_name"]
        column_name = request["column_name"]
        value = request["value"]
        enc_vector = ts.ckks_vector_from(self.sk_ctx, value)
        dec_vector = enc_vector.decrypt()
        value = dec_vector[0]
        print(f"NearValue: {value}")
        sql = "SELECT {2} FROM {0} ORDER BY ABS({1} - {2}) LIMIT 3;".format(self.database_name + "." +table_name, value , column_name)
        query_result = get_query_results(self.database_name, self.cfg, sql)
        print("Query: ",query_result)
        if len(query_result) == 0:
            ans = {}
        elif len(query_result) == 1:
            plain_vector1 = ts.plain_tensor([query_result[0]])
            enc_vector1 = ts.ckks_vector(self.pk_ctx, plain_vector1)
            ans = [enc_vector1.serialize()]
        elif len(query_result) == 2:
            plain_vector1 = ts.plain_tensor([query_result[0]])
            enc_vector1 = ts.ckks_vector(self.pk_ctx, plain_vector1)
            plain_vector2 = ts.plain_tensor([query_result[1]])
            enc_vector2 = ts.ckks_vector(self.pk_ctx, plain_vector2)
            ans = [enc_vector1.serialize(), enc_vector2.serialize()]
        else:
            plain_vector1 = ts.plain_tensor([query_result[0]])
            enc_vector1 = ts.ckks_vector(self.pk_ctx, plain_vector1)
            plain_vector2 = ts.plain_tensor([query_result[1]])
            enc_vector2 = ts.ckks_vector(self.pk_ctx, plain_vector2)
            plain_vector3 = ts.plain_tensor([query_result[2]])
            enc_vector3 = ts.ckks_vector(self.pk_ctx, plain_vector3)
            ans = [enc_vector1.serialize(), enc_vector2.serialize(), enc_vector3.serialize()]
        return ans
    

    def n_th_query_operation(self, request):
        cid = request["cid"]
        qid = request["qid"]
        n = request["n"]
        mode = request["mode"]
        table_name = request["table_name"]
        column_name = request["column_name"]
        if mode == "clean":
            self.n_th_cache.clear()
            self.hash_cache.clear()
            print("cid: ", cid, " qid: ", qid, " n: ", n, " mode: ", mode)
            sql = "SELECT {1},COUNT(*) AS i FROM {0} GROUP BY {1} ORDER BY i".format(self.database_name + "." +table_name, column_name)
            print(sql)
            query_result = get_query_results(self.database_name, self.cfg, sql)
            print(query_result)
            for i in range(len(query_result)):
                self.n_th_cache[i] = query_result[i]
                print(f"query_result: {query_result}\n\n")  #query_result[i] = 1.0  取[0]即1
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
        ans = {}
        ans["hash_value"] = hash_value
        ans["result"] = serialize_msg
        ans["available"] = available
        #response = tenseal_data_server_pb2.n_th_query_result(cid=request.cid, qid=request.qid,n = n,hash = hash_value ,result = serialize_msg,available = available)
        return ans
    
    def query_mode_using_hash(self, hash_):
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
        ans = {}
        ans["mode"] = pickle.dumps(result_list)
        ans["available"] = pickle.dumps(available_list)
        return ans
    
    def query_from_buffer(self, hash_):
        available = False
        if hash_ in self.hash_cache:
            available = True
            query_result = [self.hash_cache[hash_][1]]
        else:
            query_result = [0]

        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        serialize_msg = enc_vector.serialize()
        ans = {}
        ans["result"] = serialize_msg
        ans["available"] = available
        return ans