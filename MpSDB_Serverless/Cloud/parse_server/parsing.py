import pickle
import tenseal as ts
import time as t
import oss2
from typing import Tuple, Set, List
def put_need_query(bucket, query_db_list, cid, request_d):
    for i in query_db_list:
        bucket.put_object(f"query_user_{cid}/{i}_query_request_by_dataServer.txt", request_d)
def request_parsing(request, pk_ctx, bucket):
    cid = str(request["cid"])
    is_total = request["is_total"].upper()
    if is_total == "TOTAL":
        query_db_list = request["client_index_list"]
        request["ip_address"] = ""
        enc_vector = process_total_request(request, pk_ctx, bucket, query_db_list, cid)
    else:
        request_d = pickle.dumps(request)
        query_db_list.append(int(request["db_name"][-1]))
        put_need_query(bucket, query_db_list, cid, request_d)
        query_result_name = f"query_user_{cid}/query_result/query_result_{query_db_list[0]}.txt"
        wait_for_object(bucket, query_result_name)
        query_result = pickle.loads(bucket.get_object(query_result_name).read())
        enc_vector = ts.ckks_vector_from(pk_ctx, query_result)
        bucket.delete_object(query_result_name)
    return enc_vector
def process_total_request(request, pk_ctx, bucket, query_db_list, cid):
    op = request["op"].upper()
    column_name = request["column_name"]
    if "DWITHIN" in op or "KNN" in op:
        request["query_name"] = "query_DWithin"
        request_d = pickle.dumps(request)
        put_need_query(bucket, query_db_list, cid, request_d)
        if column_name == "COUNT(*)":
            return get_total_sum(cid, pk_ctx, bucket, query_db_list)
        else:
            return get_total_list(cid, pk_ctx, bucket, query_db_list, False, False)
    elif op == "MAX":
        request["op"] = "max"
        request_d = pickle.dumps(request)
        put_need_query(bucket, query_db_list, cid, request_d)
        total_list = get_total_list(cid, pk_ctx, bucket, query_db_list, True)
        return get_total_max(cid, request, pk_ctx, bucket, total_list)
    elif op == "MIN":
        request["op"] = "min"
        request_d = pickle.dumps(request)
        put_need_query(bucket, query_db_list, cid, request_d)
        total_list = get_total_list(cid, pk_ctx, bucket, query_db_list, True)
        return get_total_min(cid, request, pk_ctx, bucket, total_list)
    elif op == "AVG":
        return get_total_avg(cid, request, pk_ctx, bucket, query_db_list)
    elif op == "COUNT":
        request["op"] = "count"
        request_d = pickle.dumps(request)
        put_need_query(bucket, query_db_list, cid, request_d)
        return get_total_sum(cid, pk_ctx, bucket, query_db_list)
    elif "VERTICAL" in op:
        return get_total_vertical(cid, request, pk_ctx, bucket, query_db_list)
    elif op == "VARIANCE":
        return get_total_var(cid, request, pk_ctx, bucket, query_db_list)
    elif op in ["STDDEV", "STD"]:
        return get_total_std(cid, request, pk_ctx, bucket, query_db_list)
    elif op == "VAR_SAMP":
        return get_total_var_samp(cid, request, pk_ctx, bucket, query_db_list)
    elif op == "STDDEV_SAMP":
        return get_total_std_samp(cid, request, pk_ctx, bucket, query_db_list)
    else:
        request_d = pickle.dumps(request)
        put_need_query(bucket, query_db_list, cid, request_d)
        return get_total_sum(cid, pk_ctx, bucket, query_db_list)
def get_total_vertical(cid, request, pk_ctx, bucket, query_db_list):
    request["query_name"] = "query_operation_vertical"
    request_d = pickle.dumps(request)
    put_need_query(bucket, query_db_list, cid, request_d)
    total_list = get_total_list(cid, pk_ctx, bucket, query_db_list, False, False)
    global_vertical = {}
    if "STD" in request["op"].upper():
        total_enc_var_vertical_msg = pickle.dumps(total_list)
        key_server_dict = {"op": "sqrt_enc_vector_vertical", "vector_msg": total_enc_var_vertical_msg, "cid": cid}
        key_server_dict_d = pickle.dumps(key_server_dict)
        bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
        wait_for_object(bucket, f"query_user_{cid}/key_server_sqrt_enc_vector_vertical.txt")
        global_vertical = pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_sqrt_enc_vector_vertical.txt").read())
        bucket.delete_object(f"query_user_{cid}/key_server_sqrt_enc_vector_vertical.txt")
    else:
        for dict in total_list:
            global_vertical.update(dict)
    return global_vertical
def get_total_list(cid, pk_ctx, bucket, query_db_list, if_to_s, if_ckks_to_vector=True):
    total_list = []
    wait_for_all_objects(bucket, cid, query_db_list)
    for i in range(len(query_db_list)):
        query_result_name = f"query_user_{cid}/query_result/query_result_{query_db_list[i]}.txt"
        query_result = pickle.loads(bucket.get_object(query_result_name).read())
        enc_vector = ts.ckks_vector_from(pk_ctx, query_result) if if_ckks_to_vector else query_result
        if if_to_s:
            enc_vector = enc_vector.serialize()
        total_list.append(enc_vector)
        bucket.delete_object(query_result_name)
    return total_list
def wait_for_object(bucket, object_name):
    while not bucket.object_exists(object_name):
        t.sleep(0.01)
def wait_for_all_objects(bucket, cid, query_db_list):
    while True:
        now_query_result_num = sum(1 for _ in oss2.ObjectIterator(bucket, prefix=f"query_user_{cid}/query_result/"))
        if now_query_result_num == len(query_db_list):
            break
        t.sleep(0.01)
def get_total_sum(cid, pk_ctx, bucket, query_db_list):
    total_list = get_total_list(cid, pk_ctx, bucket, query_db_list, False, True)
    return sum(total_list)
def get_total_max(cid, request, pk_ctx, bucket, total_list):
    key_server_dict = {"op": "get_max_boolean_positive", "total_list": total_list, "cid": cid}
    key_server_dict_d = pickle.dumps(key_server_dict)
    bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
    wait_for_object(bucket, f"query_user_{cid}/key_server_max_result.txt")
    max_enc_msg = pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_max_result.txt").read())
    bucket.delete_object(f"query_user_{cid}/key_server_max_result.txt")
    return ts.ckks_vector_from(pk_ctx, max_enc_msg)
def get_total_min(cid, request, pk_ctx, bucket, total_list):
    key_server_dict = {"op": "get_min_boolean_positive", "total_list": total_list, "cid": cid}
    key_server_dict_d = pickle.dumps(key_server_dict)
    bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
    wait_for_object(bucket, f"query_user_{cid}/key_server_min_result.txt")
    min_enc_msg = pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_min_result.txt").read())
    bucket.delete_object(f"query_user_{cid}/key_server_min_result.txt")
    return ts.ckks_vector_from(pk_ctx, min_enc_msg)
def get_total_avg(cid, request, pk_ctx, bucket, query_db_list):
    request["op"] = "sum"
    request_d = pickle.dumps(request)
    put_need_query(bucket, query_db_list, cid, request_d)
    total_enc_sum = get_total_sum(cid, pk_ctx, bucket, query_db_list)
    request["op"] = "count"
    request_d = pickle.dumps(request)
    put_need_query(bucket, query_db_list, cid, request_d)
    total_enc_count = get_total_sum(cid, pk_ctx, bucket, query_db_list)
    total_enc_sum_msg = total_enc_sum.serialize()
    total_enc_count_msg = total_enc_count.serialize()
    key_server_dict = {"op": "div_enc_vector", "dividend_msg": total_enc_sum_msg, "divisor_msg": total_enc_count_msg, "cid": cid}
    key_server_dict_d = pickle.dumps(key_server_dict)
    bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
    wait_for_object(bucket, f"query_user_{cid}/key_server_div_result.txt")
    total_enc_avg_msg = pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_div_result.txt").read())
    bucket.delete_object(f"query_user_{cid}/key_server_div_result.txt")
    return ts.ckks_vector_from(pk_ctx, total_enc_avg_msg)
def get_total_var(cid, request, pk_ctx, bucket, query_db_list):
    total_enc_sum, total_enc_avg, total_enc_count, total_enc_mid_result = get_mid_total_var(cid, request, pk_ctx, bucket, query_db_list)
    total_enc_mid_result_msg = total_enc_mid_result.serialize()
    total_enc_count_msg = total_enc_count.serialize()
    key_server_dict = {"op": "div_enc_vector", "dividend_msg": total_enc_mid_result_msg, "divisor_msg": total_enc_count_msg, "cid": cid}
    key_server_dict_d = pickle.dumps(key_server_dict)
    bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
    wait_for_object(bucket, f"query_user_{cid}/key_server_div_result.txt")
    div_enc_mid_msg = pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_div_result.txt").read())
    bucket.delete_object(f"query_user_{cid}/key_server_div_result.txt")
    div_enc_mid_vector = ts.ckks_vector_from(pk_ctx, div_enc_mid_msg)
    total_enc_avg_2 = total_enc_avg * total_enc_avg
    return div_enc_mid_vector - total_enc_avg_2
def get_total_std(cid, request, pk_ctx, bucket, query_db_list):
    total_enc_var = get_total_var(cid, request, pk_ctx, bucket, query_db_list)
    total_enc_var_msg = total_enc_var.serialize()
    key_server_dict = {"op": "sqrt_enc_vector", "vector_msg": total_enc_var_msg, "cid": cid}
    key_server_dict_d = pickle.dumps(key_server_dict)
    bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
    wait_for_object(bucket, f"query_user_{cid}/key_server_sqrt_result.txt")
    total_enc_std_msg = pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_sqrt_result.txt").read())
    bucket.delete_object(f"query_user_{cid}/key_server_sqrt_result.txt")
    return ts.ckks_vector_from(pk_ctx, total_enc_std_msg)
def get_mid_total_var(cid, request, pk_ctx, bucket, query_db_list):
    request["op"] = "count"
    request_d = pickle.dumps(request)
    put_need_query(bucket, query_db_list, cid, request_d)
    total_enc_count = get_total_sum(cid, pk_ctx, bucket, query_db_list)
    request["op"] = "sum"
    request_d = pickle.dumps(request)
    put_need_query(bucket, query_db_list, cid, request_d)
    total_enc_sum = get_total_sum(cid, pk_ctx, bucket, query_db_list)
    request["op"] = "variance*count+avg*sum"
    request_d = pickle.dumps(request)
    put_need_query(bucket, query_db_list, cid, request_d)
    total_enc_mid_result = get_total_sum(cid, pk_ctx, bucket, query_db_list)
    total_enc_avg = get_total_avg(cid, request, pk_ctx, bucket, query_db_list)
    return total_enc_sum, total_enc_avg, total_enc_count, total_enc_mid_result