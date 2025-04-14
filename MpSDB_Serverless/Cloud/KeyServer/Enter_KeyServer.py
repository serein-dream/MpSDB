import pickle
import oss2
import json
import time as t
from KeyServer import KeyServer
def handler(event, context):
    creds = context.credentials
    auth = oss2.StsAuth(
        creds.access_key_id,
        creds.access_key_secret,
        creds.security_token
    )
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = f'oss-{oss_raw_data["events"][0]["region"]}-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    object_name = oss_info_map['object']['key']
    sk_ctx_bytes = open('/code/bin/ts_ckks.config', "rb").read()
    key_server = KeyServer(sk_ctx_bytes)
    key_server_dict = pickle.loads(bucket.get_object(object_name).read())
    op = key_server_dict["op"]
    cid = str(key_server_dict["cid"])
    to_client_bucket = oss2.Bucket(auth, endpoint, "clients-up")
    def put_result(bucket, cid, result, filename):
        bucket.put_object(f"query_user_{cid}/{filename}", pickle.dumps(result))
    if op == "get_max_boolean_positive":
        total_enc_list = key_server_dict["total_list"]
        max_enc_vector = max(total_enc_list, key=lambda x: key_server.boolean_positive_two(x, x))
        put_result(bucket, cid, max_enc_vector, "key_server_max_result.txt")
    elif op == "get_min_boolean_positive":
        total_enc_list = key_server_dict["total_list"]
        min_enc_vector = min(total_enc_list, key=lambda x: key_server.boolean_positive_two(x, x))
        put_result(bucket, cid, min_enc_vector, "key_server_min_result.txt")
    elif op == "div_enc_vector":
        ans = key_server.div_enc_vector(key_server_dict["dividend_msg"], key_server_dict["divisor_msg"])
        put_result(bucket, cid, ans, "key_server_div_result.txt")
    elif op == "sqrt_enc_vector":
        ans = key_server.sqrt_enc_vector(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_sqrt_result.txt")
    elif op == "sqrt_enc_vector_vertical":
        dict_list = pickle.loads(key_server_dict["vector_msg"])
        ans = key_server.sqrt_enc_vertical_dict(dict_list)
        put_result(bucket, cid, ans, "key_server_sqrt_enc_vector_vertical.txt")
    elif op == "generate_noise":
        total_noise_list = key_server.generate_noise(key_server_dict["cid"], key_server_dict["qid"], key_server_dict["type"])
        put_result(bucket, cid, total_noise_list, "total_noise_list.txt")
        bucket.put_object(f"query_user_{cid}/key_server_noise_ok_result.txt", "ok!")
    elif op == "is_odd":
        ans = key_server.is_odd(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_is_odd.txt")
    elif op == "is_sub_abs_1__boolean_positive":
        sub_serialize_msg = key_server_dict["vector_msg"]
        ans = {
            "is_sub_abs_1": key_server.is_sub_abs_1(sub_serialize_msg),
            "boolean_positive": key_server.boolean_positive(sub_serialize_msg)
        }
        put_result(bucket, cid, ans, "key_server_is_sub_abs_1__boolean_positive.txt")
    elif op == "is_sub_abs_1":
        ans = key_server.is_sub_abs_1(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_is_sub_abs_1.txt")
    elif op == "boolean_equal_proxi":
        ans = key_server.boolean_equal_proxi(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_boolean_equal_proxi.txt")
    elif op == "boolean_equal_proxi__boolean_positive":
        sub_serialize_msg = key_server_dict["vector_msg"]
        ans = {
            "is_equal": key_server.boolean_equal_proxi(sub_serialize_msg),
            "comparison_flag": key_server.boolean_positive(sub_serialize_msg)
        }
        put_result(bucket, cid, ans, "key_server_boolean_equal_proxi__boolean_positive.txt")
    elif op == "abs_2":
        ans = key_server.abs_2(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_abs_2.txt")
    elif op == "2_boolean_positive_proxi":
        ans = [key_server.boolean_positive_proxi(v) for v in key_server_dict["vector_msg"]]
        put_result(bucket, cid, ans, "key_server_2_boolean_positive_proxi.txt")
    elif op == "boolean_positive_proxi":
        ans = key_server.boolean_positive_proxi(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_boolean_positive_proxi.txt")
    elif op == "boolean_positive_round_proxi__boolean_equal_round_proxi":
        sub_diff_s = key_server_dict["vector_msg"]
        ans = [
            key_server.boolean_positive_round_proxi(sub_diff_s),
            key_server.boolean_equal_round_proxi(sub_diff_s)
        ]
        put_result(bucket, cid, ans, "key_server_boolean_positive_round_proxi__boolean_equal_round_proxi.txt")
    elif op == "boolean_equal_proxi__boolean_positive_proxi":
        sub_diff_s = key_server_dict["vector_msg"]
        ans = [
            key_server.boolean_equal_proxi(sub_diff_s),
            key_server.boolean_positive_proxi(sub_diff_s)
        ]
        put_result(bucket, cid, ans, "key_server_boolean_equal_proxi__boolean_positive_proxi.txt")
    elif op == "get_encryped_value_list_max_and_equal":
        encryped_value = key_server.s_to_ckk_list(key_server_dict["vector_msg"])
        max_enc_vector_list = [encryped_value[0]]
        temp_max = encryped_value[0]
        for enc_vector in encryped_value[1:]:
            sub_diff = temp_max - enc_vector
            sub_serialize_msg = sub_diff.serialize()
            comparison_flag = key_server.boolean_positive_proxi(sub_serialize_msg)
            equal_flag = key_server.boolean_equal_proxi(sub_serialize_msg)
            if not equal_flag:
                if comparison_flag:
                    continue
                else:
                    temp_max = enc_vector
                    max_enc_vector_list = [enc_vector]
            else:
                max_enc_vector_list.append(enc_vector)
        put_result(bucket, cid, max_enc_vector_list, "key_server_get_encryped_value_list_max_and_equal.txt")
    elif op == "more_get_encryped_value_list_max_and_equal":
        encryped_value = key_server.s_to_ckk_list(key_server_dict["vector_msg"])
        max_enc_vector_list = [encryped_value[0]]
        for enc_vector in encryped_value[1:]:
            sub_diff = max_enc_vector_list[0] - enc_vector
            sub_serialize_msg = sub_diff.serialize()
            comparison_flag = key_server.boolean_positive_proxi(sub_serialize_msg)
            equal_flag = key_server.boolean_equal_proxi(sub_serialize_msg)
            if not equal_flag:
                if comparison_flag:
                    continue
                else:
                    max_enc_vector_list = [enc_vector]
            else:
                max_enc_vector_list.append(enc_vector)
        orderd_hash = key_server_dict["temp_vector_msg"]
        list_hash = [orderd_hash[i] for i, v in enumerate(encryped_value) if v in max_enc_vector_list]
        put_result(bucket, cid, list_hash, "key_server_get_encryped_value_list_max_and_equal.txt")
    elif op == "unpack_enc_vector":
        ans = key_server.unpack_enc_vector(key_server_dict["vector_msg"])
        put_result(bucket, cid, ans, "key_server_unpack_enc_vector.txt")
    elif op == "return_range_query_results":
        ans_s = key_server.return_enc_range_query_result(key_server_dict)
        ans_to_client = {
            "cid": key_server_dict["cid"],
            "qid": key_server_dict["qid"],
            "result": ans_s
        }
        put_result(to_client_bucket, cid, ans_to_client, "key_server_to_client.txt")
    elif op == "return_results":
        ans_s = key_server.return_enc_query_result(key_server_dict)
        ans_to_client = {
            "cid": key_server_dict["cid"],
            "qid": key_server_dict["qid"],
            "result": ans_s
        }
        ans_name = f"query_user_{cid}/key_server_to_client.txt"
        put_result(to_client_bucket, cid, ans_to_client, ans_name)
    elif op == "return_results_vertical":
        ans_s = key_server.return_enc_query_result_vertical(key_server_dict)
        ans_to_client = {
            "cid": key_server_dict["cid"],
            "qid": key_server_dict["qid"],
            "result": ans_s
        }
        ans_name = f"query_user_{cid}/key_server_to_client.txt"
        put_result(to_client_bucket, cid, ans_to_client, ans_name)
    bucket.delete_object(object_name)