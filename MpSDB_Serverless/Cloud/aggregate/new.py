import oss2, json
import pickle
import time
import tenseal as ts
def load_selection(bucket):
    selection_dumps = bucket.get_object('selection.txt').read()
    To_average_list = pickle.loads(selection_dumps)[1]
    this_rnd = pickle.loads(selection_dumps)[0][0]
    return To_average_list, this_rnd
def check_objects_exist(bucket, To_average_list):
    for i in range(len(To_average_list)):
        if To_average_list[i] > 0:
            name = f"clients_update_params/client_{i}.txt"
            if not bucket.object_exists(name):
                return False
    return True
def collect_parameters(bucket, To_average_list, pk_ctx):
    params_list = []
    for i in range(len(To_average_list)):
        if To_average_list[i] > 0:
            name = f"clients_update_params/client_{i}.txt"
            enc_params_msg_dumps = bucket.get_object(name).read()
            enc_params_msg = pickle.loads(enc_params_msg_dumps)
            enc_params_vector = ts.ckks_vector_from(pk_ctx, enc_params_msg)
            params_list.append(enc_params_vector)
    return params_list
def collect_accum_counts(bucket, To_average_list):
    accum_count_list = []
    for i in range(len(To_average_list)):
        if To_average_list[i] > 0:
            acc_name = f"accum_count_this_client_{i}.txt"
            accum_count_this_client_dumps = bucket.get_object(acc_name).read()
            ac = pickle.loads(accum_count_this_client_dumps)
            accum_count_list.append(ac)
    return accum_count_list
def aggregate_parameters(params_list, accum_count_list):
    accum_count = sum(accum_count_list)
    sum_enc_params = sum(params_list)
    latest_enc_params = (1 / accum_count) * sum_enc_params
    latest_enc_params = latest_enc_params.serialize()
    return latest_enc_params, accum_count
def update_global_params(bucket, latest_enc_params, this_rnd):
    new_global_parmas_dumps = pickle.dumps(latest_enc_params)
    bucket.put_object('global_params.txt', new_global_parmas_dumps)
    bucket.put_object('average_done.txt', pickle.dumps('s'))
def handler(event, context):
    creds = context.credentials
    auth = oss2.StsAuth(creds.access_key_id, creds.access_key_secret, creds.security_token)
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = f'oss-{oss_raw_data["events"][0]["region"]}-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    pk_ctx_bytes = open('/code/tenseal/ts_ckks_pk.config', "rb").read()
    pk_ctx = ts.context_from(pk_ctx_bytes)
    To_average_list, this_rnd = load_selection(bucket)
    if not check_objects_exist(bucket, To_average_list):
        return
    params_list = collect_parameters(bucket, To_average_list, pk_ctx)
    accum_count_list = collect_accum_counts(bucket, To_average_list)
    latest_enc_params, accum_count = aggregate_parameters(params_list, accum_count_list)
    if not bucket.object_exists('average_done.txt'):
        update_global_params(bucket, latest_enc_params, this_rnd)