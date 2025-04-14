"""
parsing.py responses to parse the request from clients,
and return results
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
import threading
import time

import hydra
from numpy import *
import tenseal as ts
from omegaconf import DictConfig

from .utils import *
import re
import concurrent.futures

# enc_vector_thread = 0
# enc_list_thread = []
    
# parse the request from client_proxy, divided into two schemes:local and total
def request_parsing(request, pk_ctx, address_dict, options):
    
    client_num = int(request.db_name[5:])  

    new_address_dict = {}
    i = 0
    for key in address_dict:
        new_address_dict[key] = address_dict[key] 
        i = i+1
        if i == client_num+1:
            break
    address_dict = new_address_dict
    db_name = "TOTAL"
    if db_name == "TOTAL":
        enc_list = process_total_request(request, pk_ctx, address_dict, options)
        print(f"\n\n\nparsing okk\n\n\n")
        return enc_list

    else:
        db_stub = get_db_stub(request.db_name, address_dict, options)
        enc_vector = process_local_request(request, pk_ctx, db_stub)
        return enc_vector
#   here！！
# process the request in which query data_modeling is only from local database
def get_local_count(request, db_stub,n):
    mode = "encypted"
    if "CLEAN" in request.op:
        mode = "clean"
    n_th_query_msg = tenseal_data_server_pb2.n_th_query_msg(cid=request.cid, qid=request.qid, n = n,mode = mode, table_name=request.table_name,column_name=request.column_name)
    n_th_query_response = db_stub.n_th_query_operation(n_th_query_msg)
    return n_th_query_response

def process_local_request(request, pk_ctx, db_stub):
    query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=request.op,
                                                                   column_name=request.column_name,
                                                                   table_name=request.table_name)
    response = db_stub.query_operation(query_request)
    enc_vector = ts.ckks_vector_from(pk_ctx, response.enc_result)

    return enc_vector


# process the request in which query data_modeling is only from local database and needs noise.
def process_noise_local_request(request, db_stub):
    query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=request.op,
                                                                   column_name=request.column_name,
                                                                   table_name=request.table_name)
    response = db_stub.noise_query_operation(query_request)
    noise_vector = pickle.loads(response.enc_result)

    return noise_vector

#   here！！
# process the request which needs all dataServer participates
def get_n_th_list(request, db_stub_list, pk_ctx, n):
    n_th_list = []
    for stub in db_stub_list:
        n_th_query_response = get_local_count(request, stub, n)
        enc_vector = ts.ckks_vector_from(pk_ctx, n_th_query_response.result)
        hash_code = n_th_query_response.hash
        available = n_th_query_response.available
        print("hash_code:", hash_code, "available:", available)
        if available:
            n_th_list.append((hash_code,enc_vector))
    return n_th_list

from typing import Tuple, Set
from typing import List

#   here！！
def is_n_th_enough(res_list: List[List[Tuple[int,object]]]):
    set_all: List[Set[int]] = []

    data_server_num = len(res_list[0])

    for index in range(data_server_num):
        set_all.append(set())
        for res in res_list:
            if res[index] is not None:
                set_all[index].add(res[index][0])

    temp_and = set_all[0]
    temp_or = set_all[0]

    for set_ in set_all:
        temp_and = temp_and & set_
        temp_or = temp_or | set_

    return temp_and, temp_or

#   here！！
def get_encryped_total_frequency(hash_code, db_stub_list,pk_ctx):
    temp = None
    for stub in db_stub_list:
        query_request = tenseal_data_server_pb2.buffer_query_msg(hash=hash_code)
        response = stub.query_from_buffer(query_request)
        if response.available:
            enc_vector = ts.ckks_vector_from(pk_ctx, response.result)
            if temp is None:
                temp = enc_vector
            else:
                temp = temp + enc_vector

    return temp

#   here！！
def get_the_highest_hash(set_once: Set[int],key_stub,db_stub_list,pk_ctx):

    encryped_value: List[object] = []
    orderd_hash: List[int] = []
    for hash_code in set_once:
        orderd_hash.append(hash_code)
        encryped_value.append(get_encryped_total_frequency(hash_code,db_stub_list,pk_ctx))

    max_enc_vector_list = [encryped_value[0]]
    for enc_vector in encryped_value[1:]:
        sub_diff = max_enc_vector_list[0] - enc_vector
        sub_serialize_msg = sub_diff.serialize()
        request = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)
        response = key_stub.boolean_positive_proxi(request)
        equal = key_stub.boolean_equal_proxi(request)
        comparison_flag = response.bool_msg
        equal_flag = equal.bool_msg
        if not equal_flag:
            if comparison_flag:
                pass
            else:
                max_enc_vector_list = [enc_vector]
        else:
            max_enc_vector_list.append(enc_vector)

    list_hash = []

    for j in range(len(max_enc_vector_list)):
        for i in range(len(encryped_value)):
            if max_enc_vector_list[j] == encryped_value[i]:
                list_hash.append(orderd_hash[i])

    return list_hash

#   here！！
def get_enc_using_hash(hash_code,db_stub_list):
    result_set = set()
    import pickle
    print("hash_code:", hash_code)
    hash_code = pickle.dumps(hash_code)
    for stub in db_stub_list:
        # serialize the vector into bytes
        query_request = tenseal_data_server_pb2.query_mode_using_hash_msg(hash=hash_code)
        response = stub.query_mode_using_hash(query_request)
        import pickle
        available = pickle.loads(response.available)
        mode = pickle.loads(response.mode)
        for i in range(len(available)):
            if available[i]:
                result_set.add(mode[i])
    return result_set

#   here！！
def get_total_var_mode(request, db_stub_list, pk_ctx, key_stub):
    from typing import Set
    set_all: Set[int] = set()
    set_once: Set[int] = set()
    res_list: List[List[Tuple[int,object]]] = []
    i = 0
    k = 2
    out = False
    while True:
        while len(set_all) < k:
            if i == 0:
                request.op = "VAR_MODE_CLEAN"
            else:
                request.op = "VAR_MODE"
            i+= 1
            n_th_list = get_n_th_list(request, db_stub_list, pk_ctx, i-1)

            if len(n_th_list) == 0:
                out = True
                break

            res_list.append(n_th_list)

            set_all,set_once = is_n_th_enough(res_list)
        k+=1

        print(set_all,set_once)
        highest_hash_list = get_the_highest_hash(set_once,key_stub,db_stub_list,pk_ctx)
        if len(highest_hash_list) != k:
            break
        if out: break
    print("highest_hash_list:", highest_hash_list)
    enc_list = get_enc_using_hash(highest_hash_list,db_stub_list)
    res_list = []
    for enc in enc_list:
        res_list.append(ts.ckks_vector_from(pk_ctx,enc))
    # change to plain vector
    print("res_list",res_list)

    res_bytes_list = []
    for res in res_list:
        res_bytes_list.append(res.serialize())

    request_ = tenseal_key_server_pb2.vector(vector_msg=pickle.dumps(res_bytes_list))
    response = key_stub.unpack_enc_vector(request_)
    enc_vector = ts.ckks_vector_from(pk_ctx,response.vector_msg)
    return enc_vector

def get_total_count(request, db_stub_list, pk_ctx, key_stub):
    count = 0
    for stub in db_stub_list:
        query_request = tenseal_data_server_pb2.query_count_msg(table_name=request.table_name,column_name=request.column_name)
        response = stub.get_count(query_request)
        count += ts.ckks_vector_from(pk_ctx, response.enc_result)
    return count


def get_min_indexs(candidate_delta_list, db_stub_list, pk_ctx, key_stub):
    res_index = 0
    for enc_index in range(1,len(candidate_delta_list)):
        res = ts.ckks_vector_from(pk_ctx,key_stub.abs(tenseal_key_server_pb2.vector(vector_msg=candidate_delta_list[res_index].serialize())).vector_msg)
        enc = ts.ckks_vector_from(pk_ctx,key_stub.abs(tenseal_key_server_pb2.vector(vector_msg=candidate_delta_list[enc_index].serialize())).vector_msg)
        sub_diff = res - enc
        sub_diff_rev = enc - res
        sub_serialize_msg = sub_diff.serialize()
        sub_serialize_msg_rev = sub_diff_rev.serialize()
        request = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)
        response = key_stub.boolean_positive_proxi(request)
        comparison_flag = response.bool_msg
        request = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg_rev)
        response = key_stub.boolean_positive_proxi(request)
        comparison_flag_rev = response.bool_msg
        if comparison_flag and not comparison_flag_rev:
            res_index = enc_index
    return res_index


def show_all(candidate_delta_list,key_stub):
    print("showlist: ")
    for i in range(len(candidate_delta_list)):
        print(key_stub.show_raw_vector(tenseal_key_server_pb2.vector(vector_msg=candidate_delta_list[i].serialize())).raw_msg)
    pass


def get_total_var_median(request, db_stub_list, pk_ctx, key_stub, options):

    std_min = 0
    std_max = 1

    std = get_total_std(request, db_stub_list, pk_ctx, key_stub, options)
    avg = get_total_avg(request, db_stub_list, pk_ctx, key_stub, options)
    total = get_total_count(request, db_stub_list, pk_ctx, key_stub, options)

    max_3sigma = avg + 3 * std
    min_3sigma = avg - 3 * std

    is_odd = key_stub.is_odd(tenseal_key_server_pb2.vector(vector_msg=total.serialize())).bool_msg

    candidate_list = []

    i = 0

    if is_odd:
        while True:
            i+=1
            std_mid = (std_max + std_min) / 2
            temp_mid = std_mid * max_3sigma + (1 - std_mid) * min_3sigma
            if i > 100:
                print("break early")
                return temp_mid
            le_list = []
            g_list = []
            for stub in db_stub_list:
                query_request = tenseal_data_server_pb2.query_median_posi_msg(cid = request.cid, qid = request.qid, table_name = request.table_name, column_name = request.column_name, median = temp_mid.serialize(), avg = avg.serialize(), std = std.serialize())
                response = stub.query_median_posi(query_request)
                le = ts.ckks_vector_from(pk_ctx, response.less_e)
                g = ts.ckks_vector_from(pk_ctx, response.greater)
                le_list.append(le)
                g_list.append(g)
            le_sum = sum(le_list)
            g_sum = sum(g_list)
            sub_diff = le_sum - g_sum
            sub_serialize_msg = sub_diff.serialize()
            is_sub_abs_1 = key_stub.is_sub_abs_1(tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)).bool_msg
            if is_sub_abs_1:
                break
            requests = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)
            response = key_stub.boolean_positive(requests)
            comparison_flag = response.bool_msg
            with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/data_query/parse_server/log.txt", "a") as file:
                file.write(f"std_min: {std_min}, std_max: {std_max}, std_mid: {std_mid}, le_sum:{le_sum}, g_sum:{g_sum},is_sub_abs_1:{is_sub_abs_1}, comparison_flag: {comparison_flag}\n")

            if comparison_flag:
                std_max = std_mid
            else:
                std_min = std_mid
    else:
        while True:
            i+=1
            std_mid = (std_max + std_min) / 2
            temp_mid = std_mid * max_3sigma + (1 - std_mid) * min_3sigma
            le_list = []
            g_list = []
            if i > 100:
                print("break early")
                return temp_mid
            for stub in db_stub_list:
                query_request = tenseal_data_server_pb2.query_median_posi_msg(cid=request.cid, qid=request.qid,
                                                                              table_name=request.table_name,
                                                                              column_name=request.column_name,
                                                                              median=temp_mid.serialize(),
                                                                              avg=avg.serialize(), std=std.serialize())
                response = stub.query_median_posi(query_request)
                le = ts.ckks_vector_from(pk_ctx, response.less_e)
                g = ts.ckks_vector_from(pk_ctx, response.greater)
                le_list.append(le)
                g_list.append(g)
            le_sum = sum(le_list)
            g_sum = sum(g_list)
            sub_diff = le_sum - g_sum
            sub_serialize_msg = sub_diff.serialize()
            is_equal = key_stub.boolean_equal_proxi(tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)).bool_msg
            if is_equal:
                break
            requests = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)
            response = key_stub.boolean_positive(requests)
            comparison_flag = response.bool_msg
            print("std_min:", std_min, "std_max:", std_max, "std_mid:", std_mid, "comparison_flag:", comparison_flag)
            if comparison_flag:
                std_max = std_mid
            else:
                std_min = std_mid

    # get the nearest value in each dataServer
    now_mid = std_mid * max_3sigma + (1 - std_mid) * min_3sigma
    for stub in db_stub_list:
        get_neareat_request = tenseal_data_server_pb2.query_nearest_msg(table_name=request.table_name, column_name=request.column_name, value=now_mid.serialize())
        response = stub.get_nearest(get_neareat_request)
        count = response.count
        if count == 0:
            continue
        elif count == 1:
            candidate_list.append(ts.ckks_vector_from(pk_ctx, response.value1))
        elif count == 2:
            candidate_list.append(ts.ckks_vector_from(pk_ctx, response.value1))
            candidate_list.append(ts.ckks_vector_from(pk_ctx, response.value2))
        else:
            candidate_list.append(ts.ckks_vector_from(pk_ctx, response.value1))
            candidate_list.append(ts.ckks_vector_from(pk_ctx, response.value2))
            candidate_list.append(ts.ckks_vector_from(pk_ctx, response.value3))

    if is_odd:
        candidate_delta_list = []
        for candidate in candidate_list:
            le_list = []
            g_list = []
            for stub in db_stub_list:
                query_request = tenseal_data_server_pb2.query_median_posi_msg(cid=request.cid, qid=request.qid,
                                                                              table_name=request.table_name,
                                                                              column_name=request.column_name,
                                                                              median=candidate.serialize(),
                                                                              avg=avg.serialize(), std=std.serialize())
                response = stub.query_median_posi(query_request)
                le = ts.ckks_vector_from(pk_ctx, response.less_e)
                g = ts.ckks_vector_from(pk_ctx, response.greater)
                le_list.append(le)
                g_list.append(g)
            le_sum = sum(le_list)
            g_sum = sum(g_list)
            sub_diff = le_sum - g_sum
            candidate_delta_list.append(sub_diff)

        show_all(candidate_delta_list,key_stub)
        show_all(candidate_list,key_stub)

        index = get_min_indexs(candidate_delta_list,db_stub_list,pk_ctx,key_stub)
        min_ = candidate_delta_list[index]
        can = candidate_delta_list.pop(index)
        index_ = get_min_indexs(candidate_delta_list,db_stub_list,pk_ctx,key_stub)
        if index_ >= index:
            index_ += 1
        candidate_delta_list.insert(index, can)
        min__ = candidate_delta_list[index_]
        show_all([min_,min__],key_stub)
        abs_min_ = ts.ckks_vector_from(pk_ctx,key_stub.abs(tenseal_key_server_pb2.vector(vector_msg=min_.serialize())).vector_msg)
        abs_min__ = ts.ckks_vector_from(pk_ctx,key_stub.abs(tenseal_key_server_pb2.vector(vector_msg=min__.serialize())).vector_msg)
        sub_diff =  abs_min_ - abs_min__
        is_greater = key_stub.boolean_positive_round_proxi(tenseal_key_server_pb2.vector(vector_msg=sub_diff.serialize())).bool_msg
        is_equal = key_stub.boolean_equal_round_proxi(tenseal_key_server_pb2.vector(vector_msg=sub_diff.serialize())).bool_msg
        show_all([abs_min_,abs_min__],key_stub)
        if is_greater and not is_equal:
            print("is_greater")
            print("index:", index, "index_:", index_)
            return candidate_list[index_]
        elif is_equal:
            print("is_equal")
            print("index:", index, "index_:", index_)
            one = candidate_list[index]
            two = candidate_list[index_]
            sub_diff = one - two
            is_equal = key_stub.boolean_equal_proxi(tenseal_key_server_pb2.vector(vector_msg=sub_diff.serialize())).bool_msg
            is_greater = key_stub.boolean_positive_proxi(tenseal_key_server_pb2.vector(vector_msg=sub_diff.serialize())).bool_msg
            ordered_ = []
            if is_equal:
                ordered_ = [one, two]
                return (one + two) * 0.5
            elif is_greater:
                ordered_ = [two, one]
            else:
                ordered_ = [one, two]

            mid_ = (one + two) * 2
            le_list = []
            g_list = []
            for stub in db_stub_list:
                query_request = tenseal_data_server_pb2.query_median_posi_msg(cid=request.cid, qid=request.qid,
                                                                              table_name=request.table_name,
                                                                              column_name=request.column_name,
                                                                              median=mid_.serialize(),
                                                                              avg=avg.serialize(), std=std.serialize())
                response = stub.query_median_posi(query_request)
                le = ts.ckks_vector_from(pk_ctx, response.less_e)
                g = ts.ckks_vector_from(pk_ctx, response.greater)
                le_list.append(le)
                g_list.append(g)
            le_sum = sum(le_list)
            g_sum = sum(g_list)
            lr_sub_diff = le_sum - g_sum

            is_positive = key_stub.boolean_positive_proxi(tenseal_key_server_pb2.vector(vector_msg=lr_sub_diff.serialize())).bool_msg
            if is_positive:
                return ordered_[0]
            else:
                return ordered_[1]

            # delta = candidate_delta_list[index]
            # is_positive = key_stub.boolean_positive_proxi(tenseal_key_server_pb2.vector(vector_msg=delta.serialize())).bool_msg
            # if is_positive:
            #     return ordered_[0]
            # else:
            #     return ordered_[1]
        else:
            print("is_less")
            print("index:", index, "index_:", index_)
            return candidate_list[index]
    else:
        two_list = []
        for candidate in candidate_list:
            le_list = []
            g_list = []
            for stub in db_stub_list:
                query_request = tenseal_data_server_pb2.query_median_posi_msg(cid=request.cid, qid=request.qid,
                                                                                table_name=request.table_name,
                                                                                column_name=request.column_name,
                                                                                median=candidate.serialize(),
                                                                                avg=avg.serialize(), std=std.serialize())
                response = stub.query_median_posi(query_request)
                le = ts.ckks_vector_from(pk_ctx, response.less_e)
                g = ts.ckks_vector_from(pk_ctx, response.greater)
                le_list.append(le)
                g_list.append(g)
            le_sum = sum(le_list)
            g_sum = sum(g_list)
            sub_diff = le_sum - g_sum
            sub_serialize_msg = sub_diff.serialize()
            is_equal = key_stub.is_sub_abs_1(tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)).bool_msg
            if is_equal:
                two_list.append(candidate)
                if len(two_list) == 2:
                    print("two candidate, return")
                    return (two_list[0] + two_list[1]) / 2
    # return std_mid * min_3sigma + (1 - std_mid) * max_3sigma

def if_num_in_r_less_than_k(request, pk_ctx, r, address_dict, key_stub, k_, options):
    enc_vector = 0    
    pattern = r'[()]'
    sql_list = [i for i in re.split(pattern, request.op) if i != '']
    new_op = "DWithin(" + sql_list[1] + "(" + sql_list[2] + "), location, " + str(r) + ")"
    query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=new_op,
                                                                    column_name="COUNT(*)",
                                                                    table_name=request.table_name)
    # for stub in db_stub_list:
    #     response = stub.query_DWithin(query_request)
    #     enc_vector += ts.ckks_vector_from(pk_ctx, response.enc_result)       
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        threads = []
        result_enc_list_thread = [] 
        #并发
        for key in address_dict:
            if "DATA" in key.upper():
                address = address_dict[key]    
                future = executor.submit(query_DWithin_MultiThread, address, query_request, options, pk_ctx, True)
                threads.append(future)   
        for future in concurrent.futures.as_completed(threads):  # 并发执行
            result_enc_list_thread.append(future.result())   

    enc_vector = sum(result_enc_list_thread) 
    request = tenseal_key_server_pb2.is_less_than_k_in(enc_result=enc_vector.serialize(), k = k_)
    sign = key_stub.is_less_than_k(request)
    return sign     


def get_r_of_k(request, address_dict, is_right, options, pk_ctx):
    op = request.op
    if is_right == False:    
        op = op[:-2] + str(int(op[-2])//len(address_dict)) + op[-1]
    #print(f"get_r_of_k:\nis_right:{is_right}\nop:{op}")
    query_request = tenseal_data_server_pb2.query_r_by_k_in(cid=request.cid, qid=request.qid, op=op,
                                                                    column_name="id",
                                                                    table_name=request.table_name,
                                                                    greater = is_right)
    r_list = []
    # for stub in db_stub_list:
    #     response = stub.query_r_by_k(query_request)
    #     r_list.append(pickle.loads(response.enc_result))

    # global enc_list_thread
    # threads = []
    # #并发
    # enc_list_thread = []
    # for key in address_dict:
    #     if "DATA" in key.upper():
    #         address = address_dict[key]    
    #         thread = threading.Thread(target=query_r_by_k_MultiThread, args=(address, query_request, options))
    #         threads.append(thread)          
    #         thread.start()    
    # for thread in threads:
    #     thread.join()   

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        threads = []
        result_enc_list_thread = [] 
        #并发
        for key in address_dict:
            if "DATA" in key.upper():
                address = address_dict[key]    
                future = executor.submit(query_r_by_k_MultiThread, address, query_request, options)
                threads.append(future)   
        for future in concurrent.futures.as_completed(threads):  # 并发执行
            result_enc_list_thread.append(future.result())      

    if is_right == False:  
        r = min(result_enc_list_thread)
    else:
        r = max(result_enc_list_thread)
    return r


def query_DWithin_MultiThread(address, query_request, options, pk_ctx, if_tockk):
    stub = get_one_db_stub(address, options)
    response = stub.query_DWithin(query_request)
    # global enc_list_thread
    # if if_tockk:
    #     enc_list_thread.append(ts.ckks_vector_from(pk_ctx, response.enc_result))
    # else:
    #     enc_list_thread.append(response.enc_result)
    if if_tockk:
        return ts.ckks_vector_from(pk_ctx, response.enc_result)
    else:
        return response.enc_result

def query_r_by_k_MultiThread(address, query_request, options):
    stub = get_one_db_stub(address, options)
    response = stub.query_r_by_k(query_request)
    # global enc_list_thread
    # enc_list_thread.append(pickle.loads(response.enc_result))
    return pickle.loads(response.enc_result)

    
def process_total_request(request, pk_ctx, address_dict, options):
    db_stub_list = get_all_db_stub(address_dict, options)
    op = request.op.upper()        
    column_name = request.column_name
    # sum value of column from all dataServer, op: count or sum

    # if op == "COUNT":
    #     key_stub = get_keyserver_stub(address_dict, options)
    #     count_vector = get_total_count(request, db_stub_list, key_stub)
    #     count_palin_vector = ts.plain_tensor(count_vector)
    #     count_enc_vector = ts.ckks_vector(pk_ctx, count_palin_vector)
    #     return count_enc_vector
    # max value of column from all dataServer
    # global enc_list_thread
    if "DWITHIN" in op or "KNN" in op: 
        if column_name == "COUNT(*)":#返回单值
            query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=request.op,
                                                                            column_name=request.column_name,
                                                                            table_name=request.table_name)

            #并发：         
            # for stub in db_stub_list:
            #     response = stub.query_DWithin(query_request)
            #     enc_vector += ts.ckks_vector_from(pk_ctx, response.enc_result)
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                threads = []
                result_enc_list_thread = [] 
                #并发
                for key in address_dict:
                    if "DATA" in key.upper():
                        address = address_dict[key]    
                        future = executor.submit(query_DWithin_MultiThread, address, query_request, options, pk_ctx, True)
                        threads.append(future)   
                for future in concurrent.futures.as_completed(threads):  # 并发执行
                    result_enc_list_thread.append(future.result())                 
            
            return sum(result_enc_list_thread) 
        else:      
            column_name = request.column_name
            #total_list = get_total_list(request, db_stub_list, pk_ctx)
            query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=request.op,
                                                                            column_name=request.column_name,
                                                                            table_name=request.table_name)
            st = time.time()
            # threads = []
            # #并发
            # enc_list_thread = []
            # for key in address_dict:
            #     if "DATA" in key.upper():
            #         address = address_dict[key]    
            #         thread = threading.Thread(target=query_DWithin_MultiThread, args=(address, query_request, options, pk_ctx, False))
            #         threads.append(thread)          
            #         thread.start()    
            # for thread in threads:
            #     thread.join()     

            # # for stub in db_stub_list:
            # #     response = stub.query_DWithin(query_request)
            # #     enc_vector_list.append(response.enc_result)
            # # sum_enc_vector = sum(total_list)
                
            # et = time.time()
            # print(f"wait_for dataserver: {round(et-st,4)}")
            # return enc_list_thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                threads = []
                result_enc_list_thread = [] 
                #并发
                for key in address_dict:
                    if "DATA" in key.upper():
                        address = address_dict[key]    
                        future = executor.submit(query_DWithin_MultiThread, address, query_request, options, pk_ctx, False)
                        threads.append(future)   
                for future in concurrent.futures.as_completed(threads):  # 并发执行
                    result_enc_list_thread.append(future.result())                 
            et = time.time()
            print(f"wait_for dataserver: {round(et-st,4)}")
            return result_enc_list_thread
    
    elif "KN_B_N" in op:  
        key_server_stub = get_key_server_stub(address_dict, options)            
        left = get_r_of_k(request, address_dict, False, options, pk_ctx)
        right = get_r_of_k(request, address_dict, True, options, pk_ctx)
        deviation = 1e-6
        mid = (left + right) / 2
        while (left + deviation <= right):
            print(f"left:{left} right:{right} mid:{mid}")
            mid = (left + right) / 2
            k_ = int(request.op[-2])
            response = if_num_in_r_less_than_k(request, pk_ctx, mid, address_dict, key_server_stub, k_, options)
            sign = response.is_less
            print(f"sign:{sign}")
            if sign>0:
                left = mid
            elif sign<0:
                right = mid
            else:
                break
        pattern = r'[()]'
        sql_list = [i for i in re.split(pattern, request.op) if i != '']
        new_op = "DWithin(" + sql_list[1] + "(" + sql_list[2] + "), location, " + str(mid) + ")"            
        query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=new_op,
                                                                        column_name=request.column_name,
                                                                        table_name=request.table_name)
    #   for stub in db_stub_list:
    #       response = stub.query_DWithin(query_request)
    #       enc_vector_list.append(response.enc_result)
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            threads = []
            result_enc_list_thread = [] 
            #并发
            for key in address_dict:
                if "DATA" in key.upper():
                    address = address_dict[key]    
                    future = executor.submit(query_DWithin_MultiThread, address, query_request, options, pk_ctx, False)
                    threads.append(future)   
            for future in concurrent.futures.as_completed(threads):  # 并发执行
                result_enc_list_thread.append(future.result())   

        return result_enc_list_thread
                   
    elif op == "MAX":
        key_stub = get_key_server_stub(address_dict, options)
        request.op = "max"
        total_list = get_total_list_multiThread(request, address_dict, pk_ctx, options)
        max_enc_vector = get_total_max(request, total_list, pk_ctx, key_stub, options)
        return max_enc_vector
    # min value of column from all dataServer
    elif op == "MIN":
        key_stub = get_key_server_stub(address_dict, options)  
        request.op = "min"      
        total_list = get_total_list_multiThread(request, address_dict, pk_ctx, options)
        min_enc_vector = get_total_min(request, total_list, pk_ctx, key_stub, options)
        return min_enc_vector
    # average value of column from all dataServer
    elif op == "AVG":
        key_stub = get_key_server_stub(address_dict, options)
        avg_enc_vector = get_total_avg(request, address_dict, pk_ctx, key_stub, options)
        return avg_enc_vector
    elif "VERTICAL" in op:
        key_stub = get_key_server_stub(address_dict, options)
        avg_enc_vector = get_total_avg_vertical(request, address_dict, pk_ctx, key_stub, options)
        return avg_enc_vector
    elif op == "VARIANCE":
        key_stub = get_key_server_stub(address_dict, options)
        # var_vector = get_total_var_dp(request, address_dict, key_stub)
        # var_palin_vector = ts.plain_tensor(var_vector)
        # var_enc_vector = ts.ckks_vector(pk_ctx, var_palin_vector)
        var_enc_vector = get_total_var(request, address_dict, pk_ctx, key_stub, options)
        return var_enc_vector
    elif op in ["STDDEV", "STD"]:
        key_stub = get_key_server_stub(address_dict, options)
        std_enc_vector = get_total_std(request, address_dict, pk_ctx, key_stub, options)
        return std_enc_vector
    elif op == "VAR_SAMP":
        key_stub = get_key_server_stub(address_dict, options)
        var_samp_enc_vector = get_total_var_samp(request, address_dict, pk_ctx, key_stub)
        return var_samp_enc_vector
    elif op == "STDDEV_SAMP":
        key_stub = get_key_server_stub(address_dict, options)
        std_samp_enc_vector = get_total_std_samp(request, address_dict, pk_ctx, key_stub)
        return std_samp_enc_vector
    
    #begin

    elif op == "VAR_MODE":
        key_stub = get_key_server_stub(address_dict, options)
        var_mode_enc_vector = get_total_var_mode(request, address_dict, pk_ctx, key_stub)
        return var_mode_enc_vector
    elif op == "VAR_MEDIAN":
        key_stub = get_key_server_stub(address_dict, options)
        var_median_enc_vector = get_total_var_median(request, address_dict, pk_ctx, key_stub)
        return var_median_enc_vector  
     #end 
    else:
        sum_enc_vector = get_total_sum(request, address_dict, pk_ctx, options)
        return sum_enc_vector



def query_tongji_MultiThread(address, query_request, options, pk_ctx):
    stub = get_one_db_stub(address, options)
    response = stub.query_operation(query_request)
    enc_vector = ts.ckks_vector_from(pk_ctx, response.enc_result)
    return enc_vector
    
# get all query result(encrypted vector) from data_server,stored in a list
def get_total_list_multiThread(request, address_dict, pk_ctx, options):
    query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=request.op,
                                                                   column_name=request.column_name,
                                                                   table_name=request.table_name)
    total_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        threads = []
        result_enc_list_thread = [] 
        #并发
        for key in address_dict:
            if "DATA" in key.upper():
                address = address_dict[key]    
                future = executor.submit(query_tongji_MultiThread, address, query_request, options, pk_ctx)
                threads.append(future)   
        for future in concurrent.futures.as_completed(threads):  # 并发执行
            result_enc_list_thread.append(future.result())         

    return result_enc_list_thread



# get all query result with noise from data_server,stored in a list
def get_noise_total_list(request, db_stub_list):
    total_list = []
    for stub in db_stub_list:
        noise_vector = process_noise_local_request(request, stub)
        total_list.extend(noise_vector)
    return total_list


# sum all noise query results, get the plain sum
def get_noise_total_sum(request, db_stub_list):
    noise_total_list = get_noise_total_list(request, db_stub_list)
    print(noise_total_list)
    sum_noise_vector = sum(noise_total_list)
    return sum_noise_vector


# sum all query results, get the encrypted sum
def get_total_sum(request, db_stub_list, pk_ctx, options):
    total_list = get_total_list_multiThread(request, db_stub_list, pk_ctx, options)
    sum_enc_vector = sum(total_list)
    return sum_enc_vector


# get the count over the total query result list with noise
def get_total_count_dp(request, db_stub_list, key_stub):
    sum_noise_vector = []
    generate_noise_request = tenseal_key_server_pb2.generate_noise_request(cid=request.cid, qid=request.qid,
                                                                           type="float")
    key_stub.generate_noise(generate_noise_request)
    noise_total_list = get_noise_total_list(request, db_stub_list)
    sum_noise = sum(noise_total_list)
    sum_noise_vector.append(sum_noise)
    return sum_noise_vector


# get the max encrypted vector over the total query result list
def get_total_max(request, total_list, pk_ctx, key_stub): # TODO： 有可能出现两个一样的情况

    max_enc_vector = total_list[0]
    for enc_vector in total_list[1:]:
        sub_diff = max_enc_vector - enc_vector
        sub_serialize_msg = sub_diff.serialize()
        request = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)
        response = key_stub.boolean_positive(request)
        comparison_flag = response.bool_msg
        if not comparison_flag:
            max_enc_vector = enc_vector
    return max_enc_vector


# get the min encrypted vector over the total query result list
def get_total_min(request, total_list, pk_ctx, key_stub):
    request.op = "min"

    min_enc_vector = total_list[0]
    for enc_vector in total_list[1:]:
        sub_diff = min_enc_vector - enc_vector
        sub_serialize_msg = sub_diff.serialize()
        request = tenseal_key_server_pb2.vector(vector_msg=sub_serialize_msg)
        response = key_stub.boolean_positive(request)
        comparison_flag = response.bool_msg
        if comparison_flag:
            min_enc_vector = enc_vector
    return min_enc_vector


# get the average encrypted vector over the total query result list
def get_total_avg_dp(request, db_stub_list, pk_ctx, key_stub):
    # key_server generates noises foe the qid-th request
    generate_noise_request = tenseal_key_server_pb2.generate_noise_request(cid=request.cid, qid=request.qid,
                                                                           type="float")
    key_stub.generate_noise(generate_noise_request)
    # get the encrypted total sum
    request.op = "sum"
    total_list = get_total_list(request, db_stub_list, pk_ctx)
    sum_enc_vector = sum(total_list)

    # get the plain total count
    request.op = "count"
    # noise_count_sum = get_noise_total_sum(request, db_stub_list)
    # noise_count_sum = round(noise_count_sum)
    noise_count_sum = get_total_count_dp(request, db_stub_list, key_stub)
    noise_count_sum = round(noise_count_sum[0])

    avg_enc_vector = 1 / noise_count_sum * sum_enc_vector

    return avg_enc_vector


def get_total_avg(request, db_stub_list, pk_ctx, key_stub, options):
    # get the encrypted total sum
    request.op = "sum"
    total_enc_sum = get_total_sum(request, db_stub_list, pk_ctx, options)

    # get the encrypted total count
    request.op = "count"
    total_enc_count = get_total_sum(request, db_stub_list, pk_ctx, options)

    # get the average(sum/count) by calling the division interface provided by key_server
    total_enc_sum_msg = total_enc_sum.serialize()
    total_enc_count_msg = total_enc_count.serialize()
    div_request = tenseal_key_server_pb2.div_vectors(dividend_msg=total_enc_sum_msg, divisor_msg=total_enc_count_msg)
    div_response = key_stub.div_enc_vector(div_request)
    total_enc_avg_msg = div_response.vector_msg

    avg_enc_vector = ts.ckks_vector_from(pk_ctx, total_enc_avg_msg)
    return avg_enc_vector



def query_tongji_vertical_MultiThread(address, query_request, options, pk_ctx):
    stub = get_one_db_stub(address, options)
    response = stub.query_operation_vertical(query_request)
    data_query_vertical = pickle.loads(response.enc_result)
    return data_query_vertical
    

def get_total_avg_vertical(request, address_dict, pk_ctx, key_stub, options):
    # get the encrypted total sum
    global_vertical = {}
    query_request = tenseal_data_server_pb2.query_msg_parse_server(cid=request.cid, qid=request.qid, op=request.op,
                                                                   column_name=request.column_name,
                                                                   table_name=request.table_name)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        threads = []
        total_list = [] 
        #并发
        for key in address_dict:
            if "DATA" in key.upper():
                address = address_dict[key]    
                future = executor.submit(query_tongji_vertical_MultiThread, address, query_request, options, pk_ctx)
                threads.append(future)   
        for future in concurrent.futures.as_completed(threads):  # 并发执行
            total_list.append(future.result())         

    if "STD" in request.op.upper():
        print("to sqrt")    
        total_enc_var_vertical_msg = pickle.dumps(total_list)
        sqrt_request = tenseal_key_server_pb2.vector(vector_msg=total_enc_var_vertical_msg)
        sqrt_response = key_stub.sqrt_enc_vector_vertical(sqrt_request)
        global_vertical = pickle.loads(sqrt_response.vector_msg)
    else:
        for dict in total_list:
            for key,value in dict.items():
                global_vertical[key] = value
    # print(f"\n\n\nquery data:{db_stub}  okk\n\n\n")

    return global_vertical

# # get middle values for the total SD_Sample
def get_total_var_dp(request, db_stub_list, key_stub):
    generate_noise_request = tenseal_key_server_pb2.generate_noise_request(cid=request.cid, qid=request.qid,
                                                                           type="float")
    key_stub.generate_noise(generate_noise_request)
    request.op = "variance*count+avg*sum"
    noise_total_square_list = get_noise_total_list(request, db_stub_list)
    request.op = "sum"
    noise_total_sum_list = get_noise_total_list(request, db_stub_list)
    request.op = "count"
    noise_total_count_list = get_noise_total_list(request, db_stub_list)

    total_square = sum(noise_total_square_list)
    total_sum = sum(noise_total_sum_list)
    total_count = sum(noise_total_count_list)

    combined_var = [1 / total_count * (total_square - 1 / total_count * total_sum * total_sum)]

    return combined_var


# get the variance over the total query result list
# def get_total_var(request, db_stub_list, pk_ctx, key_stub):
#     # for each group,
#     request.op = "sum"
#     sum_total_list = get_total_list(request, db_stub_list, pk_ctx)
#     request.op = "count"
#     count_total_list = get_total_list(request, db_stub_list, pk_ctx)
#     total_count = get_total_count_dp(request, db_stub_list, key_stub)
#     total_count = round(total_count[0])
#     request.op = "variance"
#     var_total_list = get_total_list(request, db_stub_list, pk_ctx)
#     request.op = "avg"
#     avg_total_list = get_total_list(request, db_stub_list, pk_ctx)
#
#     square_x = multiply(var_total_list, count_total_list) + multiply(avg_total_list, sum_total_list)
#
#     total_sum = sum(sum_total_list)
#     total_square = sum(square_x)
#
#     combined_var = 1 / total_count * (total_square - 1 / total_count * total_sum * total_sum)
#
#     return combined_var

def get_mid_total_var(request, db_stub_list, pk_ctx, key_stub, options):
    # get the average,count,and variance*count+avg*sum(mid_result) over the total
    request.op = "avg"
    total_enc_avg = get_total_avg(request, db_stub_list, pk_ctx, key_stub, options)
    request.op = "count"
    total_enc_count = get_total_sum(request, db_stub_list, pk_ctx, options)
    request.op = "sum"
    total_enc_sum = get_total_sum(request, db_stub_list, pk_ctx, options)
    request.op = "variance*count+avg*sum"
    total_enc_mid_result = get_total_sum(request, db_stub_list, pk_ctx, options)

    return total_enc_sum, total_enc_avg, total_enc_count, total_enc_mid_result


def get_total_var(request, db_stub_list, pk_ctx, key_stub, options):
    total_enc_sum, total_enc_avg, total_enc_count, total_enc_mid_result = get_mid_total_var(request, db_stub_list,
                                                                                            pk_ctx, key_stub, options)
    # 4.222  9.0  222.0  24.667
    # get the division result:total_mid_result/total_count
    total_enc_mid_result_msg = total_enc_mid_result.serialize()
    total_enc_count_msg = total_enc_count.serialize()
    div_request = tenseal_key_server_pb2.div_vectors(dividend_msg=total_enc_mid_result_msg,
                                                     divisor_msg=total_enc_count_msg)
    div_response = key_stub.div_enc_vector(div_request)
    div_enc_mid_msg = div_response.vector_msg
    div_enc_mid_vector = ts.ckks_vector_from(pk_ctx, div_enc_mid_msg)

    # combined_var = div_mid_enc_vector - avg*avg
    total_enc_var = div_enc_mid_vector - total_enc_avg * total_enc_avg

    return total_enc_var


def get_total_var_samp(request, db_stub_list, pk_ctx, key_stub):
    total_enc_sum, total_enc_avg, total_enc_count, total_enc_mid_result = get_mid_total_var(request, db_stub_list,
                                                                                            pk_ctx, key_stub)

    total_enc_mid_result_msg = total_enc_mid_result.serialize()
    divisor_vector_1 = total_enc_count - 1
    divisor_msg_1 = divisor_vector_1.serialize()
    div_request = tenseal_key_server_pb2.div_vectors(dividend_msg=total_enc_mid_result_msg,
                                                     divisor_msg=divisor_msg_1)
    div_response_1 = key_stub.div_enc_vector(div_request)
    div_enc_mid_msg_1 = div_response_1.vector_msg
    enc_mid_vector_1 = ts.ckks_vector_from(pk_ctx, div_enc_mid_msg_1)

    dividend_vector = total_enc_sum * total_enc_avg
    dividend_msg = dividend_vector.serialize()
    divisor_vector_2 = total_enc_count - 1
    divisor_msg = divisor_vector_2.serialize()
    div_request = tenseal_key_server_pb2.div_vectors(dividend_msg=dividend_msg, divisor_msg=divisor_msg)
    div_response_2 = key_stub.div_enc_vector(div_request)
    div_enc_mid_msg_2 = div_response_2.vector_msg
    enc_mid_vector_2 = ts.ckks_vector_from(pk_ctx, div_enc_mid_msg_2)

    total_enc_var_samp = enc_mid_vector_1 - enc_mid_vector_2

    return total_enc_var_samp
#enc_mid_vector_1   27.75   enc_mid_vector_2  20.056


def get_total_std(request, db_stub_list, pk_ctx, key_stub, options):
    total_enc_var = get_total_var(request, db_stub_list, pk_ctx, key_stub, options)
    total_enc_var_msg = total_enc_var.serialize()
    sqrt_request = tenseal_key_server_pb2.vector(vector_msg=total_enc_var_msg)
    sqrt_response = key_stub.sqrt_enc_vector(sqrt_request)
    total_enc_std_msg = sqrt_response.vector_msg

    total_enc_std = ts.ckks_vector_from(pk_ctx, total_enc_std_msg)

    return total_enc_std


def get_total_std_samp(request, db_stub_list, pk_ctx, key_stub):
    total_enc_var_samp = get_total_var_samp(request, db_stub_list, pk_ctx, key_stub)

    total_enc_var_samp_msg = total_enc_var_samp.serialize()
    sqrt_request = tenseal_key_server_pb2.vector(vector_msg=total_enc_var_samp_msg)
    sqrt_response = key_stub.sqrt_enc_vector(sqrt_request)
    total_enc_std_samp_msg = sqrt_response.vector_msg

    total_enc_std_samp = ts.ckks_vector_from(pk_ctx, total_enc_std_samp_msg)

    return total_enc_std_samp
