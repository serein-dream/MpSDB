import pickle
from typing import List
import tenseal as ts
from utils import generate_noise_list, remove_noise_list
import numpy as np
class KeyServer():
    def __init__(self, sk_ctx_bytes):
        self.db_num = 3
        self.sk_ctx = ts.context_from(sk_ctx_bytes)
        self.total_noise_list = []
        self.sleep_time = 0.1
        self.cid_list = []
        self.max_id_list = []
        self.min_id_list = []
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
    def decrypt_vector(self, enc_vector):
        return ts.ckks_vector_from(self.sk_ctx, enc_vector).decrypt()
    def return_enc_query_result(self, request):
        remove_noise_list(request["cid"], request["qid"], self.total_noise_list)
        dec_vector = self.decrypt_vector(request["enc_vector"])
        return pickle.dumps(dec_vector)
    def return_enc_query_result_vertical(self, request):
        remove_noise_list(request["cid"], request["qid"], self.total_noise_list)
        global_vertical = pickle.loads(request["enc_vector"])
        dec_global_vertical = {key: self.decrypt_vector(value) for key, value in global_vertical.items()}
        return pickle.dumps(dec_global_vertical)
    def return_enc_range_query_result(self, request):
        remove_noise_list(request["cid"], request["qid"], self.total_noise_list)
        enc_vector_list = pickle.loads(request["enc_vector"])
        client_num = len(enc_vector_list) - 2
        op_upper = enc_vector_list[client_num]
        column_upper = enc_vector_list[client_num + 1]
        if "DWITHIN" in op_upper or "KN_B_N" in op_upper:
            dec_vector = [self.decrypt_vector(enc_vector_list[i])[:-1] for i in range(client_num)]
            dec_vector = [item for sublist in dec_vector for item in sublist]
        else:
            k = 0
            dec_vector = np.array([])
            for i in range(client_num):
                dec_vector_i = self.decrypt_vector(enc_vector_list[i])[:-1]
                k = round(dec_vector_i[-1])
                dec_vector = np.concatenate((dec_vector, dec_vector_i), axis=0)
            dec_vector = dec_vector.reshape((-1, 2))
            dec_vector = dec_vector[np.argsort(dec_vector[:, 1])]
            dec_vector = dec_vector.ravel()[0::2][:k]
        return pickle.dumps(dec_vector)
    def boolean_positive_two(self, enc_1, enc_2):
        return self.decrypt_vector(enc_1) > self.decrypt_vector(enc_2)
    def generate_noise(self, cid, qid, noise_type):
        noise_list = generate_noise_list(self.db_num, noise_type)
        self.total_noise_list.append({'cid': cid, 'qid': qid, 'noise_list': noise_list})
        return self.total_noise_list
    def sqrt_enc_vector(self, to_sqrt_enc_vector):
        dec_vector = self.decrypt_vector(to_sqrt_enc_vector)
        sqrt_dec_vector = np.sqrt(dec_vector)
        sqrt_plain_vector = ts.plain_tensor(sqrt_dec_vector)
        sqrt_enc_vector = ts.ckks_vector(self.sk_ctx, sqrt_plain_vector)
        return sqrt_enc_vector.serialize()
    def sqrt_enc_vertical_dict(self, dict_list):
        return {key: self.sqrt_enc_vector(value) for dict in dict_list for key, value in dict.items()}
    def s_to_ckk_list(self, enc_s_list):
        return [ts.ckks_vector_from(self.sk_ctx, enc_s) for enc_s in enc_s_list]
    def div_enc_vector(self, dividend_enc_msg, divisor_enc_msg):
        dividend_dec_vector = self.decrypt_vector(dividend_enc_msg)
        divisor_dec_vector = self.decrypt_vector(divisor_enc_msg)
        div_dec_vector = np.divide(dividend_dec_vector, divisor_dec_vector)
        div_plain_vector = ts.plain_tensor(div_dec_vector)
        div_enc_vector = ts.ckks_vector(self.sk_ctx, div_plain_vector)
        return div_enc_vector.serialize()
    def unpack_enc_vector(self, enc_serialize_msg):
        enc_vector = pickle.loads(enc_serialize_msg)
        unpacked_enc_vector = [self.decrypt_vector(vec)[0] for vec in enc_vector]
        plain_vector = ts.plain_tensor(unpacked_enc_vector)
        enc_vector = ts.ckks_vector(self.sk_ctx, plain_vector)
        return enc_vector.serialize()
    def boolean_positive_proxi(self, enc_serialize_msg):
        dec_vector = self.decrypt_vector(enc_serialize_msg)
        return dec_vector[0] >= 0 or abs(dec_vector[0]) <= 1e-5
    def boolean_equal_proxi_and_positive(self, sub_serialize_msg):
        dec_vector = self.decrypt_vector(sub_serialize_msg)
        boolean_equal_proxi = abs(dec_vector[0]) <= 1e-8
        boolean_positive = dec_vector[0] > 0
        return {"boolean_equal_proxi": boolean_equal_proxi, "boolean_positive": boolean_positive}
    def boolean_equal_round_proxi(self, enc_serialize_msg):
        dec_vector = self.decrypt_vector(enc_serialize_msg)
        return round(abs(dec_vector[0])) == 0
    def is_sub_abs_1(self, sub_serialize_msg):
        dec_vector = self.decrypt_vector(sub_serialize_msg)
        return round(abs(dec_vector[0])) == 1
    def boolean_positive_round_proxi(self, enc_serialize_msg):
        dec_vector = self.decrypt_vector(enc_serialize_msg)
        return round(dec_vector[0]) >= 0
    def boolean_positive(self, enc_serialize_msg):
        dec_vector = self.decrypt_vector(enc_serialize_msg)
        return dec_vector[0] > 0
    def abs_2(self, vector_msg_list):
        dec_vector1 = self.decrypt_vector(vector_msg_list[0])
        dec_vector2 = self.decrypt_vector(vector_msg_list[1])
        res1 = [abs(dec_vector1[0])]
        res2 = [abs(dec_vector2[0])]
        plain_vector1 = ts.plain_tensor(res1)
        plain_vector2 = ts.plain_tensor(res2)
        enc_vector1 = ts.ckks_vector(self.sk_ctx, plain_vector1)
        enc_vector2 = ts.ckks_vector(self.sk_ctx, plain_vector2)
        return [enc_vector1.serialize(), enc_vector2.serialize()]
    def boolean_equal_proxi(self, enc_serialize_msg):
        dec_vector = self.decrypt_vector(enc_serialize_msg)
        return abs(dec_vector[0]) <= 1e-8
    def is_odd(self, enc_serialize_msg):
        dec_vector = self.decrypt_vector(enc_serialize_msg)
        return round(abs(dec_vector[0])) % 2 == 1