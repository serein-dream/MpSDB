import json
from DataServer import DatabaseServer
import time
import oss2
import pickle
import json
from DataServer import DatabaseServer
import time
import oss2
import pickle
def handler(event, context):
    time_log = {}
    t1 = time.time()
    creds = context.credentials
    auth = oss2.StsAuth(creds.access_key_id, creds.access_key_secret, creds.security_token)
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = f'oss-{oss_raw_data["events"][0]["region"]}-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    object_name = oss_info_map['object']['key']
    num_id = extract_num_id(object_name)
    request_dict = pickle.loads(bucket.get_object(object_name).read())
    cid = str(request_dict["cid"])
    db_name = determine_db_name(request_dict, num_id)
    database_server = DatabaseServer(db_name, int(num_id))
    t2 = time.time()
    time_log["ini"] = t2 - t1
    enc_result = execute_query(bucket, database_server, request_dict)
    t3 = time.time()
    time_log["function"] = t3 - t2
    oss_result_name = generate_oss_result_name(cid, num_id)
    enc_result_s = pickle.dumps(enc_result)
    bucket.put_object(oss_result_name, enc_result_s)
    bucket.delete_object(object_name)
    t4 = time.time()
    time_log["upload and delete"] = t4 - t3
def extract_num_id(object_name):
    start_index = object_name.find("/") + 1
    end_index = object_name.find("_", start_index)
    return object_name[start_index:end_index]
def determine_db_name(request_dict, num_id):
    db_name = request_dict["db_name"] + num_id
    if "taxi_" in request_dict["db_name"] or "tpch" in request_dict["db_name"]:
        db_name = request_dict["db_name"]
    return db_name
def execute_query(bucket, database_server, request_dict):
    query_name = request_dict["query_name"]
    if query_name == "normal":
        return database_server.query_operation(request_dict)
    elif query_name == "query_operation_vertical":
        return database_server.query_operation_vertical(request_dict)
    elif query_name == "median":
        return database_server.query_median_posi(request_dict)
    elif query_name == "get_nearest":
        return database_server.get_nearest(request_dict)
    elif query_name == "n_th_query_operation":
        return database_server.n_th_query_operation(request_dict)
    elif query_name == "query_mode_using_hash":
        return database_server.query_mode_using_hash(request_dict["hash_code"])
    elif query_name == "query_DWithin":
        enc_result, _ = database_server.query_DWithin(request_dict)
        return enc_result
    elif query_name == "query_r_by_k":
        return database_server.query_r_by_k(request_dict)
    elif query_name == "for_query_from_buffer":
        enc_result = []
        hash_code_list = request_dict["hash_code_list"]
        for hash_code in hash_code_list:
            enc_result.append(database_server.query_from_buffer(hash_code))
        return enc_result
    elif query_name == "for_query_from_buffer_once":
        return database_server.query_from_buffer(request_dict["hash_code_list"])
    elif query_name == "noise_signal_query":
        request_dict = pickle.loads(bucket.get_object("noise_query_request.txt").read())
        return database_server.noise_query_operation(request_dict, bucket)
    else:
        raise ValueError(f"Unknown query name: {query_name}")
def generate_oss_result_name(cid, num_id):
    return f"query_user_{cid}/query_result/query_result_{num_id}.txt"