from conn_mysql import generate_sql, get_noise_query_results, get_query_results, get_r_by_k
import tenseal as ts
from conn_mysql import *
import pickle
import random
import copy
from typing import Set, Dict, Tuple, List
import time

# from transmission.psi import decode_ids_from_client, encode_and_hash_local_id_use_sk


def query_operation(request, bucket, db_name):

    #db_number = int(db_name[-1])
    if "osm" in db_name:
        database_name = "osm_db"
        table_name = db_name
    else:
        database_name = db_name
        table_name = "table_1"

    time_log = {}
    t11 = time.time()
    #pk_bytes = bucket.get_object('key/ts_ckks_pk.config').read()
    pk_bytes = open('/code/pymysql/ts_ckks_pk.config', "rb").read()
    t12 = time.time()
    pk_ctx = ts.context_from(pk_bytes)
    t13 = time.time()
    #ctx_byte = bucket.get_object('key/ts_ckks.config').read()
    ctx_byte = open('/code/pymysql/ts_ckks_pk.config', "rb").read()
    time_log["read_pk"] = t12 - t11
    time_log["read_sk"] = t13 - t12
    print(f"\n\n{time_log}\n\n")
    sk_ctx = ts.context_from(ctx_byte)
    print(time_log)
    time_log = {}
    t1 = time.time()
    sql, _ = generate_sql(table_name, request)
    t2 = time.time()
    time_log["generate_sql"] = t2 - t1
    query_result = get_query_results(database_name, sql)#name就是db_name
    t3 = time.time()
    time_log["query"] = t3 - t2
    plain_vector = ts.plain_tensor(query_result)
    enc_vector = ts.ckks_vector(pk_ctx, plain_vector)
    serialize_msg = enc_vector.serialize()
    t4 = time.time()
    time_log["enc"] = t4 - t3
    print(f"\n\n{time_log}\n\n")
    #response = tenseal_data_server_pb2.enc_query_result(enc_result=serialize_msg)
    return serialize_msg


