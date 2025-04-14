import numpy as np


def generate_noise_list(db_num, noise_type):
    sensitivity = 1
    epsilon = 1e-7
    noise_type = noise_type.upper()
    # generate laplace noise list
    noise_list = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=db_num - 1)

    if noise_type == "INT":
        noise_list = noise_list.astype(int)

    noise_list = noise_list.tolist()
    last_noise = 0 - sum(noise_list)
    noise_list.append(last_noise)

    return noise_list


def get_noise(cid, qid, db_name, total_noise_list):
    result = []
    for noise_dict in total_noise_list:
        if noise_dict['cid'] == cid and noise_dict['qid'] == qid:
            noise_list = noise_dict['noise_list']
            result.append(noise_list[db_name - 1])
            return result


def remove_noise_list(cid, qid, total_noise_list):
    for noise_dict in total_noise_list:
        if noise_dict['cid'] == cid and noise_dict['qid'] == qid:
            total_noise_list.remove(noise_dict)

import pickle
from typing import List
import tenseal as ts
from utils import generate_noise_list, remove_noise_list
import numpy as np

class KeyServer():
    def __init__(self, sk_ctx_bytes):
        self.db_num = 3
        self.sk_ctx = ts.context_from(sk_ctx_bytes)
        self.total_noise_list = []  #41
        self.sleep_time = 0.1
        # ID Psi
        self.cid_list = []
        self.max_id_list = []
        self.min_id_list = []
        #
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.completed = False
         

    def reset_status(self):
        self.n_sum_request = 0
        self.n_sum_response = 0

    def reset_id_psi_status(self):
        self.cid_list = []
        self.max_id_list = []
        self.min_id_list = []
        self.completed = False
        self.n_sum_request = 0
        self.n_sum_response = 0

    def return_num_dec(self, sss):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, sss)
        dec_vector = enc_vector.decrypt()
        return dec_vector


    def return_enc_query_result(self, request):
        # remove the noise list by cid and qid
        remove_noise_list(request["cid"], request["qid"], self.total_noise_list)

        # receive and decrypt the results from parse_server
        enc_serialize_msg = request["enc_vector"]
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        # make request and send it to client_proxy
        dec_serialize_msg = pickle.dumps(dec_vector)
        return dec_serialize_msg

    def return_enc_query_result_vertical(self, request):
        # remove the noise list by cid and qid
        print(f"noise_list:\n{self.total_noise_list}")
        remove_noise_list(request["cid"], request["qid"], self.total_noise_list)
        global_vertical = pickle.loads(request["enc_vector"])
        dec_global_vertical = {}
        print(f"\n\nbegin:\n")
        for key,value in global_vertical.items():
            enc_vector = ts.ckks_vector_from(self.sk_ctx, value)
            dec_value = enc_vector.decrypt()
            dec_global_vertical[key] = dec_value
            print(f"\nkey: {key}\nvalue:{dec_value}\n")
        # make request and send it to client_proxy
        dec_serialize_msg = pickle.dumps(dec_global_vertical)
        return dec_serialize_msg


    def return_enc_range_query_result(self, request):
        # remove the noise list by cid and qid
        print(f"noise_list:\n{self.total_noise_list}")
        remove_noise_list(request["cid"], request["qid"], self.total_noise_list)

        # receive and decrypt the results from parse_server
        ip_address = request["ip_address"]
        enc_serialize_msg = request["enc_vector"]
        enc_vector_list = pickle.loads(enc_serialize_msg)
        client_num = len(enc_vector_list)-2
        op_upper = enc_vector_list[client_num]
        column_upper = enc_vector_list[client_num+1]
        if "DWITHIN" in op_upper or "KN_B_N" in op_upper :
            ans_list = []
            for i in range(client_num):
                enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_vector_list[i])
                dec_vector = enc_vector.decrypt() 
                dec_vector = dec_vector[:-1]
                for j in range(len(dec_vector)):
                    ans_list.append(dec_vector[j])
            dec_vector = ans_list
        else:
            k = 0
            for i in range(client_num):
                enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_vector_list[i])
                new_dec_vector = enc_vector.decrypt()               
                k = round(new_dec_vector[-1]) 
                new_dec_vector = new_dec_vector[:-1]
                if i==0:
                    dec_vector = new_dec_vector
                else:
                    dec_vector = np.concatenate((dec_vector,new_dec_vector), axis=0)
            desired_shape = (-1, 2)
            dec_vector = dec_vector.reshape(desired_shape)
            sorted_indices = np.argsort(dec_vector[:, 1])
            dec_vector = dec_vector[sorted_indices]
            dec_vector = dec_vector.ravel()
            print(f"dec_vector:\n{dec_vector}")
            dec_vector = dec_vector[0::2]
            dec_vector = dec_vector[:k]         
        # make request and send it to client_proxy
        print(f"dec_vector:\n{dec_vector}")
        dec_serialize_msg = pickle.dumps(dec_vector)
        return dec_serialize_msg

    def boolean_positive_two(self, enc_1, enc_2):
        enc_vector_1 = ts.ckks_vector_from(self.sk_ctx, enc_1)
        dec_vector_1 = enc_vector_1.decrypt()
        enc_vector_2 = ts.ckks_vector_from(self.sk_ctx, enc_2)
        dec_vector_2 = enc_vector_2.decrypt()
        flag = False
        if dec_vector_1  > dec_vector_2:
            flag = True
        return flag
        

    def generate_noise(self, cid, qid, noise_type):
        qid = qid
        cid = cid
        noise_type = noise_type
        noise_list = generate_noise_list(self.db_num, noise_type)
        noise_dict = {'cid': cid, 'qid': qid, 'noise_list': noise_list}
        self.total_noise_list.append(noise_dict)
        return self.total_noise_list
  

    def sqrt_enc_vector(self, to_sqrt_enc_vector):
        enc_serialize_msg = to_sqrt_enc_vector
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        sqrt_dec_vector = np.sqrt(dec_vector)

        sqrt_plain_vector = ts.plain_tensor(sqrt_dec_vector)
        sqrt_enc_vector = ts.ckks_vector(self.sk_ctx, sqrt_plain_vector)
        sqrt_serialized_msg = sqrt_enc_vector.serialize()

        return sqrt_serialized_msg
    def sqrt_enc_vertical_dict(self, dict_list):
        global_vertical = {}
        for dict in dict_list:
            for key,value in dict.items():
                enc_vector = ts.ckks_vector_from(self.sk_ctx, value)
                dec_vector = enc_vector.decrypt()
                sqrt_dec_vector = np.sqrt(dec_vector)
                sqrt_plain_vector = ts.plain_tensor(sqrt_dec_vector)
                sqrt_enc_vector = ts.ckks_vector(self.sk_ctx, sqrt_plain_vector)
                global_vertical[key]  = sqrt_enc_vector.serialize()

        return global_vertical

    def s_to_ckk_list(self,enc_s_list):

        enc_list:List[object] = []
        for enc_s in enc_s_list:
            enc_list.append(ts.ckks_vector_from(self.sk_ctx, enc_s))
        return enc_list

    def div_enc_vector(self,dividend_enc_msg, divisor_enc_msg):
        # get the dividend vector
        dividend_enc_vector = ts.ckks_vector_from(self.sk_ctx, dividend_enc_msg)
        dividend_dec_vector = dividend_enc_vector.decrypt()
        # get the divisor vector
        divisor_enc_vector = ts.ckks_vector_from(self.sk_ctx, divisor_enc_msg)
        divisor_dec_vector = divisor_enc_vector.decrypt()
        # get the division(dividend/divisor)
        div_dec_vector = np.divide(dividend_dec_vector, divisor_dec_vector)
        # make response and return
        div_plain_vector = ts.plain_tensor(div_dec_vector)
        div_enc_vector = ts.ckks_vector(self.sk_ctx, div_plain_vector)
        div_enc_vector_serialized_msg = div_enc_vector.serialize()
        return div_enc_vector_serialized_msg

    def unpack_enc_vector(self, enc_serialize_msg):
        enc_vector = pickle.loads(enc_serialize_msg)
        unpacked_enc_vector = []
        for i in range(len(enc_vector)):
            unpacked_enc_vector.append(ts.ckks_vector_from(self.sk_ctx, enc_vector[i]).decrypt()[0])
        plain_vector = ts.plain_tensor(unpacked_enc_vector)
        enc_vector = ts.ckks_vector(self.sk_ctx, plain_vector)
        sqrt_serialized_msg = enc_vector.serialize()
        return sqrt_serialized_msg
    
    
    def boolean_positive_proxi(self, enc_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        if dec_vector[0] >= 0 or abs(dec_vector[0]) <= 1e-5: # close to zero
            flag = True
        return flag

    
    def boolean_equal_proxi_and_positive(self, sub_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, sub_serialize_msg)
        dec_vector = enc_vector.decrypt()
        boolean_equal_proxi = False
        print(dec_vector[0])
        if abs(dec_vector[0]) <= 1e-8:
            boolean_equal_proxi = True
        if dec_vector[0] > 0:
            boolean_positive = True
        ans = {}
        ans["boolean_equal_proxi"] = boolean_equal_proxi
        ans["boolean_positive"] = boolean_positive
        return ans
    

    def boolean_equal_round_proxi(self, enc_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        if round(abs(dec_vector[0])) == 0:
            flag = True
        return flag

    #包含boolean_positive
    def is_sub_abs_1(self, sub_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, sub_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        if round(abs(dec_vector[0])) == 1:
            flag = True
        return flag

    def boolean_positive_round_proxi(self, enc_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        if round(dec_vector[0]) >= 0:
            flag = True
        return flag
    

    def get_global_max_min_id(self, request, context):
        cid = request.cid
        qid = request.qid

        if cid not in self.cid_list:
            self.cid_list.append(cid)
            self.max_id_list.append(request.max_id)
            self.min_id_list.append(request.min_id)
            self.n_sum_request += 1
        else:
            raise ValueError("Already requested.")

        while (self.n_sum_request % self.db_num != 0):
            time.sleep(self.sleep_time)

        global_max_id = max(self.max_id_list)
        global_min_id = min(self.min_id_list)

        response = tenseal_key_server_pb2.max_min_ids(
            cid=cid,
            qid=qid,
            global_max_id=global_max_id,
            global_min_id=global_min_id
        )
        self.n_sum_response += 1

        while (self.n_sum_response % self.db_num != 0):
            time.sleep(self.sleep_time)

        self.reset_id_psi_status()

        return response

    def boolean_positive(self, enc_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        # with open("ADD_by_yourself/FaaS_FL2023/faas_fed/data_query/key_server/log.txt", "+a") as f:
        #     f.write(f"dec_vector:  {dec_vector}\n")
        if dec_vector[0] > 0:
            flag = True
        return flag
    

    def abs_2(self, vector_msg_list):
        enc_vector1 = ts.ckks_vector_from(self.sk_ctx, vector_msg_list[0])
        enc_vector2 = ts.ckks_vector_from(self.sk_ctx, vector_msg_list[1])
        dec_vector1 = enc_vector1.decrypt()
        dec_vector2 = enc_vector2.decrypt()
        res1 = [abs(dec_vector1[0])]
        res2 = [abs(dec_vector2[0])]
        plain_vector1 = ts.plain_tensor(res1)
        plain_vector2 = ts.plain_tensor(res2)
        enc_vector = ts.ckks_vector(self.sk_ctx, plain_vector1)
        enc_vector = ts.ckks_vector(self.sk_ctx, plain_vector2)
        sqrt_serialized_msg1 = enc_vector1.serialize()
        sqrt_serialized_msg2 = enc_vector2.serialize()
        ans = []
        ans.append(sqrt_serialized_msg1)
        ans.append(sqrt_serialized_msg2)
        return ans


    def boolean_equal_proxi(self, enc_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        if abs(dec_vector[0]) <= 1e-8:
            flag = True
        return flag

    def is_odd(self, enc_serialize_msg):
        enc_vector = ts.ckks_vector_from(self.sk_ctx, enc_serialize_msg)
        dec_vector = enc_vector.decrypt()
        flag = False
        if round(abs(dec_vector[0])) % 2 == 1:
            flag = True
        return flag
    


import pickle
import oss2, json
from KeyServer import KeyServer
import time as t

def handler(event, context): 
    print(f"test_time_key_start: {round(t.time(),5)}")
    creds = context.credentials
    auth=oss2.StsAuth(
        creds.access_key_id,
        creds.access_key_secret,
        creds.security_token)
    oss_raw_data = json.loads(event)
    #print(oss_raw_data)
    # Get oss event related parameters passed by oss trigger 
    oss_info_map = oss_raw_data['events'][0]['oss']
    # Get oss bucket name
    bucket_name = oss_info_map['bucket']['name']
    # Set oss service endpoint
    endpoint = 'oss-' +  oss_raw_data['events'][0]['region'] + '-internal.aliyuncs.com'
    # Initiate oss client
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    object_name = oss_info_map['object']['key']
    query_now = False

    #sk_ctx_bytes = bucket.get_object('key/ts_ckks.config').read()
    sk_ctx_bytes = open('/code/bin/ts_ckks.config', "rb").read()
    key_server = KeyServer(sk_ctx_bytes)

    key_server_dict = pickle.loads(bucket.get_object(object_name).read())
    op = key_server_dict["op"]
    cid = str(key_server_dict["cid"])
    to_client_bucket = oss2.Bucket(auth, endpoint, "clients-up") 
    if op == "get_max_boolean_positive":
        total_enc_list = key_server_dict["total_list"]
        max_enc_vector = total_enc_list[0]
        for enc_vector in total_enc_list[1:]:
            comparison_flag = key_server.boolean_positive_two(max_enc_vector, enc_vector)
            if not comparison_flag:
                max_enc_vector = enc_vector
        ans = max_enc_vector
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_max_result.txt",ans)

    elif op == "get_min_boolean_positive":
        total_enc_list = key_server_dict["total_list"]
        min_enc_vector = total_enc_list[0]
        for enc_vector in total_enc_list[1:]:
            comparison_flag = key_server.boolean_positive_two(min_enc_vector, enc_vector)
            if comparison_flag:
                min_enc_vector = enc_vector
        ans = min_enc_vector
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_min_result.txt",ans)
        
        
    elif  op == "div_enc_vector":   
        print(f"\n div_enc_vector \n")             
        total_enc_sum_msg = key_server_dict["dividend_msg"]
        total_enc_count_msg = key_server_dict["divisor_msg"]
        ans = key_server.div_enc_vector(total_enc_sum_msg, total_enc_count_msg)#dividend_msg=total_enc_sum_msg, divisor_msg=total_enc_count_msg
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_div_result.txt",ans)

    elif  op == "sqrt_enc_vector":
        print(f"\n sqrt_enc_vector \n")   
        to_sqrt_enc_vector = key_server_dict["vector_msg"]
        ans = key_server.sqrt_enc_vector(to_sqrt_enc_vector)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_sqrt_result.txt",ans)     

    elif  op == "sqrt_enc_vector_vertical":
        print(f"\n sqrt_enc_vector_vertical \n")   
        dict_list = pickle.loads(key_server_dict["vector_msg"])   
        global_vertical = key_server.sqrt_enc_vertical_dict(dict_list)
        ans = pickle.dumps(global_vertical)
        bucket.put_object("query_user_" + cid + "/key_server_sqrt_enc_vector_vertical.txt",ans)            

    elif  op == "generate_noise":
        print(f"\n generate_noise \n")   
        total_noise_list = key_server.generate_noise(key_server_dict["cid"], key_server_dict["qid"], key_server_dict["type"])
        total_noise_list_s = pickle.dump(total_noise_list)
        bucket.put_object("query_user_" + cid + "/total_noise_list.txt", total_noise_list_s)
        bucket.put_object("query_user_" + cid + "/key_server_noise_ok_result.txt","ok!")
        
    elif  op == "is_odd":
        print(f"\n id_odd(奇数) \n") 
        total_enc_count = key_server_dict["vector_msg"]  
        ans = key_server.is_odd(total_enc_count)
        ans_s = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_is_odd.txt",ans_s)
        
    elif  op == "is_sub_abs_1__boolean_positive":
        print(f"\n is_sub_abs_1 \n") 
        sub_serialize_msg = key_server_dict["vector_msg"]  
        ans = {}
        ans["is_sub_abs_1"] = key_server.is_sub_abs_1(sub_serialize_msg)
        ans["boolean_positive"] = key_server.boolean_positive(sub_serialize_msg)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_is_sub_abs_1__boolean_positive.txt",ans)
        
    elif  op == "is_sub_abs_1":
        print(f"\n is_sub_abs_1 \n") 
        sub_serialize_msg = key_server_dict["vector_msg"]  
        ans = key_server.is_sub_abs_1(sub_serialize_msg)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_is_sub_abs_1.txt",ans)
        
    elif  op == "boolean_equal_proxi":
        print(f"\n boolean_equal__positive_proxi \n") 
        sub_serialize_msg = key_server_dict["vector_msg"]  
        ans = key_server.boolean_equal_proxi(sub_serialize_msg)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_boolean_equal_proxi.txt",ans)
        
    elif  op == "boolean_equal_proxi__boolean_positive":
        print(f"\n boolean_equal__positive_proxi \n") 
        sub_serialize_msg = key_server_dict["vector_msg"]  
        ans = {}
        ans["is_equal"] = key_server.boolean_equal_proxi(sub_serialize_msg)
        ans["comparison_flag"] = key_server.boolean_positive(sub_serialize_msg)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_boolean_equal_proxi__boolean_positive.txt",ans)
        
    elif  op == "abs_2":
        print(f"\n abs_2 \n") 
        vector_msg_list = key_server_dict["vector_msg"]  
        ans = key_server.abs_2(vector_msg_list)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_abs_2.txt",ans)
        
    elif  op == "2_boolean_positive_proxi":
        print(f"\n 2_boolean_positive_proxi \n") 
        ans = []
        vector_msg_list = key_server_dict["vector_msg"]  
        sub_diff0 = vector_msg_list[0]  
        ans.append(key_server.boolean_positive_proxi(sub_diff0))
        sub_diff1 = vector_msg_list[1]  
        ans.append(key_server.boolean_positive_proxi(sub_diff1))
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_2_boolean_positive_proxi.txt",ans)
        
    elif  op == "boolean_positive_proxi":
        print(f"\n boolean_positive_proxi \n") 
        vector_msg_list = key_server_dict["vector_msg"] 
        ans = key_server.boolean_positive_proxi(vector_msg_list)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_boolean_positive_proxi.txt",ans)
        
    elif  op == "boolean_positive_round_proxi__boolean_equal_round_proxi":
        print(f"\n boolean_positive_equal_round_proxi \n") 
        ans = []
        sub_diff_s = key_server_dict["vector_msg"]  
        ans.append(key_server.boolean_positive_round_proxi(sub_diff_s))
        ans.append(key_server.boolean_equal_round_proxi(sub_diff_s))
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_boolean_positive_round_proxi__boolean_equal_round_proxi.txt",ans)
        
    elif  op == "boolean_equal_proxi__boolean_positive_proxi":
        print(f"\n boolean_equal_proxi__boolean_positive_proxi \n") 
        ans = []
        sub_diff_s = key_server_dict["vector_msg"]  
        ans.append(key_server.boolean_equal_proxi(sub_diff_s))
        ans.append(key_server.boolean_positive_proxi(sub_diff_s))
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "/key_server_boolean_equal_proxi__boolean_positive_proxi.txt",ans)

        
    elif  op == "get_encryped_value_list_max_and_equal":
        print(f"\n get_encryped_value_list_max_and_equal \n") 
        ans = []
        encryped_value = key_server.s_to_ckk_list(key_server_dict["vector_msg"])  
        max_enc_vector_list = [encryped_value[0]]
        temp_max = encryped_value[0]
        for enc_vector in encryped_value[1:]:
            sub_diff = temp_max - enc_vector
            sub_serialize_msg = sub_diff.serialize()
            comparison_flag = key_server.boolean_positive_proxi(sub_serialize_msg)
            equal_flag = key_server.boolean_equal_proxi(sub_serialize_msg)
            max_enc_vector_list = [encryped_value[0].serialize()]
            if not equal_flag:
                if comparison_flag:
                    pass
                else:
                    temp_max = enc_vector
                    max_enc_vector_list = [enc_vector.serialize()]
            else:
                max_enc_vector_list.append(enc_vector.serialize())
        max_enc_vector_list_s = pickle.dumps(max_enc_vector_list)  
        print(f"\n\nlen(max_enc_vector_list)：{len(max_enc_vector_list)}\n\n")              
        bucket.put_object("query_user_" + cid + "/key_server_get_encryped_value_list_max_and_equal.txt",max_enc_vector_list_s)
        
    elif  op == "more_get_encryped_value_list_max_and_equal":
        print(f"\n get_encryped_value_list_max_and_equal \n") 
        ans = []
        encryped_value = key_server.s_to_ckk_list(key_server_dict["vector_msg"])  
        max_enc_vector_list = [encryped_value[0]]
        i = 1
        for enc_vector in encryped_value[1:]:
            sub_diff = max_enc_vector_list[0] - enc_vector
            sub_serialize_msg = sub_diff.serialize()
            comparison_flag = key_server.boolean_positive_proxi(sub_serialize_msg)
            equal_flag = key_server.boolean_equal_proxi(sub_serialize_msg)
            print(f"hrrrr------{i}-----equal_flag: {equal_flag}-----comparison_flag:  {comparison_flag}---max_enc_vector_list[0]: {key_server.return_num_dec(max_enc_vector_list[0].serialize())}")
            if not equal_flag:
                if comparison_flag:
                    print("```````pass!")
                    pass
                else:
                    print("```````change!")
                    max_enc_vector_list = [enc_vector]
            else:
                max_enc_vector_list.append(enc_vector)
            i = i+1
        orderd_hash = key_server_dict["temp_vector_msg"]
        list_hash = []
        for j in range(len(max_enc_vector_list)):
            for i in range(len(encryped_value)):
        max_enc_vector_list_s = pickle.dumps(list_hash)  
        bucket.put_object("query_user_" + cid + "/key_server_get_encryped_value_list_max_and_equal.txt",max_enc_vector_list_s)
        
    elif  op == "unpack_enc_vector":
        print(f"\n unpack_enc_vector \n") 
        res_bytes_list_s = key_server_dict["vector_msg"]  
        ans = key_server.unpack_enc_vector(res_bytes_list_s)
        ans = pickle.dumps(ans)
        bucket.put_object("query_user_" + cid + "key_server_unpack_enc_vector.txt",ans)
    
    elif  op == "return_range_query_results":
        print(f"\n return_results \n")   
        ans_s = key_server.return_enc_range_query_result(key_server_dict)
        ans_to_client = {}
        ans_to_client["cid"] = key_server_dict["cid"]
        ans_to_client["qid"] = key_server_dict["qid"]
        ans_to_client["result"] = ans_s
        key_server_dict_d = pickle.dumps(ans_to_client)
        to_client_bucket.put_object("query_user_" + cid + "/key_server_to_client.txt", key_server_dict_d) 

    elif  op == "return_results":
        print(f"\n\n\n\n return_results \n")   
        ans_s = key_server.return_enc_query_result(key_server_dict)
        ans_to_client = {}
        ans_to_client["cid"] = key_server_dict["cid"]
        ans_to_client["qid"] = key_server_dict["qid"]
        ans_to_client["result"] = ans_s
        key_server_dict_d = pickle.dumps(ans_to_client)
        ans_name = "query_user_" + cid + "/key_server_to_client.txt"
        to_client_bucket.put_object(ans_name, key_server_dict_d) 
        
        print(f"test_time_to_client: {round(t.time(),5)}")
        print(f"ans_name: {ans_name}")

    elif  op == "return_results_vertical":
        print(f"\n\n\n\n return_results_vertical \n")   
        ans_s = key_server.return_enc_query_result_vertical(key_server_dict)
        ans_to_client = {}
        ans_to_client["cid"] = key_server_dict["cid"]
        ans_to_client["qid"] = key_server_dict["qid"]
        ans_to_client["result"] = ans_s
        key_server_dict_d = pickle.dumps(ans_to_client)
        ans_name = "query_user_" + cid + "/key_server_to_client.txt"
        to_client_bucket.put_object(ans_name, key_server_dict_d) 
        
        print(f"test_time_to_client: {round(t.time(),5)}")
        print(f"ans_name: {ans_name}")

    bucket.delete_object(object_name)

