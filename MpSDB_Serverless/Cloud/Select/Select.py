# -*- coding: utf-8 -*-
import oss2, json
import time
import pickle
import numpy as np
def handler(event, context):
    sel_time_s = time.perf_counter()
    in_com_time = 0
    real_comp = 0
    creds = context.credentials
    auth = oss2.StsAuth(creds.access_key_id, creds.access_key_secret, creds.security_token)
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = f'oss-{oss_raw_data["events"][0]["region"]}-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    clients_list, in_com_time, real_comp = load_clients_list(bucket, in_com_time, real_comp)
    last_epoch_partin, in_com_time, real_comp = load_last_epoch_partin(bucket, in_com_time, real_comp)
    wait_for_clients_update(bucket, last_epoch_partin, in_com_time)
    clean_up_objects(bucket, len(clients_list), in_com_time)
    update_clients_list, in_com_time, real_comp = select_clients(clients_list, in_com_time, real_comp)
    update_shared_variables(bucket, len(clients_list), update_clients_list, in_com_time, real_comp)
    record_selection(bucket, update_clients_list, in_com_time, real_comp)
    mark_initialization_complete(bucket, in_com_time)
    record_selection_time(bucket, real_comp, sel_time_s, time.perf_counter())
def load_clients_list(bucket, in_com_time, real_comp):
    in_com_s1 = time.perf_counter()
    clients_inf_dumps = bucket.get_object('clients_list_dumps.txt').read()
    in_com_e1 = time.perf_counter()
    in_com_time += in_com_e1 - in_com_s1
    real_comp_st = time.perf_counter()
    clients_list = pickle.loads(clients_inf_dumps)
    real_comp_ed = time.perf_counter()
    real_comp += real_comp_ed - real_comp_st
    return clients_list, in_com_time, real_comp
def load_last_epoch_partin(bucket, in_com_time, real_comp):
    last_epoch_partin = None
    if bucket.object_exists('average_done.txt'):
        in_com_s2 = time.perf_counter()
        last_epoch_partin_dumps = bucket.get_object('if_average.txt').read()
        in_com_e2 = time.perf_counter()
        in_com_time += in_com_e2 - in_com_s2
        real_comp_st = time.perf_counter()
        last_epoch_partin = pickle.loads(last_epoch_partin_dumps)
        real_comp_ed = time.perf_counter()
        real_comp += real_comp_ed - real_comp_st
    return last_epoch_partin, in_com_time, real_comp
def wait_for_clients_update(bucket, last_epoch_partin, in_com_time):
    if last_epoch_partin is not None:
        start_t = time.perf_counter()
        all_client_update_itself = False
        while not all_client_update_itself:
            for i, value in enumerate(last_epoch_partin):
                if value > 0:
                    ok_name = f"OK{i}.txt"
                    if not bucket.object_exists(ok_name):
                        break
                    else:
                        last_epoch_partin[i] = 0
                        bucket.delete_object(ok_name)
            okk = sum(last_epoch_partin)
            if okk == 0:
                all_client_update_itself = True
            else:
                now_t = time.perf_counter()
                if now_t - start_t > 40.0:
                    all_client_update_itself = True
                time.sleep(0.05)
def clean_up_objects(bucket, all_clients_num, in_com_time):
    in_com_s3 = time.perf_counter()
    bucket.delete_object('average_done.txt')
    if bucket.object_exists('doing_aggre.txt'):
        bucket.delete_object('doing_aggre.txt')
    for i in range(all_clients_num):
        msg_name = f"clients_update_params/client_{i}.txt"
        if bucket.object_exists(msg_name):
            bucket.delete_object(msg_name)
        msg_name = f"OK{i}.txt"
        if bucket.object_exists(msg_name):
            bucket.delete_object(msg_name)
    in_com_e3 = time.perf_counter()
    in_com_time += in_com_e3 - in_com_s3
def select_clients(clients_list, in_com_time, real_comp):
    real_comp_st = time.perf_counter()
    update_size = np.random.randint(low=2, high=len(clients_list) + 1)
    update_clients_list = np.random.choice(clients_list, update_size, replace=False).tolist()
    real_comp_ed = time.perf_counter()
    real_comp += real_comp_ed - real_comp_st
    return update_clients_list, in_com_time, real_comp
def update_shared_variables(bucket, all_clients_num, update_clients_list, in_com_time, real_comp):
    real_comp_st = time.perf_counter()
    count_dict = [0] * all_clients_num
    count_dict_dumps = pickle.dumps(count_dict)
    accum_count_list = []
    accum_count_list_dumps = pickle.dumps(accum_count_list)
    real_comp_ed = time.perf_counter()
    real_comp += real_comp_ed - real_comp_st
    bucket.put_object('count_dict.txt', count_dict_dumps)
    bucket.put_object('accum_count_list.txt', accum_count_list_dumps)
def record_selection(bucket, update_clients_list, in_com_time, real_comp):
    in_com_s4 = time.perf_counter()
    if bucket.object_exists('selection.txt'):
        last_epoch_inf_dumps = bucket.get_object('selection.txt').read()
        last_epoch_inf = pickle.loads(last_epoch_inf_dumps)
        this_rnd = last_epoch_inf[0]
        this_rnd[0] += 1
        temp_sel = [1] * len(update_clients_list)
        last_epoch_inf[0] = this_rnd
        last_epoch_inf[1] = temp_sel
        current_epoch_inf_dumps = pickle.dumps(last_epoch_inf)
    else:
        this_rnd = [0] * len(update_clients_list)
        temp_sel = [1] * len(update_clients_list)
        current_epoch_inf = [this_rnd, temp_sel]
        current_epoch_inf_dumps = pickle.dumps(current_epoch_inf)
    in_com_e4 = time.perf_counter()
    in_com_time += in_com_e4 - in_com_s4
    To_average_list_dumps = pickle.dumps(temp_sel)
    bucket.put_object('if_average.txt', To_average_list_dumps)
    bucket.put_object('selection.txt', current_epoch_inf_dumps)
def mark_initialization_complete(bucket, in_com_time):
    in_com_s5 = time.perf_counter()
    bucket.put_object('init_list_done.txt', pickle.dumps("done"))
    in_com_e5 = time.perf_counter()
    in_com_time += in_com_e5 - in_com_s5
def record_selection_time(bucket, real_comp, sel_time_s, sel_time_e):
    sel_time = sel_time_e - sel_time_s - in_com_time
    if bucket.object_exists('hfl_sel_time_record.txt'):
        time_record_dumps = bucket.get_object('hfl_sel_time_record.txt').read()
        time_record = pickle.loads(time_record_dumps)
        time_record.append(real_comp)
    else:
        time_record = [real_comp]
    time_record_dumps = pickle.dumps(time_record)
    bucket.put_object('hfl_sel_time_record.txt', time_record_dumps)