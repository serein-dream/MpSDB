import oss2, json
import pickle
import tenseal as ts
import time,datetime
import random

def handler(event, context):
    run_start = time.perf_counter()
    in_com_time = 0
    real_comp = 0
    read_t = 0
    creds = context.credentials
    auth=oss2.StsAuth(creds.access_key_id, creds.access_key_secret, creds.security_token)
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = 'oss-' +  oss_raw_data['events'][0]['region'] + '-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    object_name = oss_info_map['object']['key']
    this_rank = ord(object_name[22]) - ord('0')

    s_d = pickle.dumps('s')
    print(f"\nby:  {object_name}\n")
    read_ed = time.perf_counter()
    read_t += read_ed - run_start

    #判断是否聚合
    in_com_s = time.perf_counter()
    selection_dumps = bucket.get_object('selection.txt').read()#读取参与聚合序列
    in_com_e = time.perf_counter()
    in_com_time += in_com_e - in_com_s

    real_comp_st = time.perf_counter()
    To_average_list = pickle.loads(selection_dumps)[1]
    this_rnd = pickle.loads(selection_dumps)[0][0]
  
    do_average = True
    num = 0
    for obj in oss2.ObjectIterator(bucket, prefix='clients_update_params/'):
        num = num+1
    print(f"num:{num}")
    if num<len(To_average_list):
        do_average = False
    print(f"num:{num}  do_average:{do_average}")
    
    real_comp_ed = time.perf_counter()
    real_comp += real_comp_ed - real_comp_st

    if do_average == False:
        return
    if do_average == True:
        #加载密钥
        read_st = time.perf_counter()
        pk_ctx_bytes = open('/code/tenseal/ts_ckks_pk.config', "rb").read()
        pk_ctx = ts.context_from(pk_ctx_bytes)
        reline_keys = pk_ctx.relin_keys()
        read_ed = time.perf_counter()
        read_t += read_ed - read_st
        time_auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        time_bucket = oss2.Bucket(time_auth, "ADD_by_yourself", "test-tpc-h")
        if time_bucket.object_exists('hfl_aggre_read_time_record.txt')==True:
            time_record_dumps = time_bucket.get_object('hfl_aggre_read_time_record.txt').read()
            time_record = pickle.loads(time_record_dumps)
            time_record.append(read_t)
        else:
            time_record = [read_t]
        time_record_dumps = pickle.dumps(time_record)
        time_bucket.put_object('hfl_aggre_read_time_record.txt', time_record_dumps)
        print("up read\n\n\n\n")

        #开始聚合
            
        #聚合操作1:收集参数
        params_list = []
        for i in range(len(To_average_list)):
            if To_average_list[i]>0:
                name = "clients_update_params/client_" + str(i) + ".txt"
                in_com_s2 = time.perf_counter()
                enc_params_msg_dumps = bucket.get_object(name).read()
                in_com_e2 = time.perf_counter()
                in_com_time += in_com_e2 - in_com_s2


                real_comp_st = time.perf_counter()
                enc_params_msg = pickle.loads(enc_params_msg_dumps)
                enc_params_vector = ts.ckks_vector_from(pk_ctx, enc_params_msg)
                params_list.append(enc_params_vector)
                # print(f"add params {i}\n")    
                real_comp_ed = time.perf_counter()
                real_comp += real_comp_ed - real_comp_st

        accum_count_list = []
        for i in range(len(To_average_list)):
            if To_average_list[i]>0 :
                acc_name = "accum_count_this_client_" + str(i) + ".txt"

                in_com_s3 = time.perf_counter()
                accum_count_this_client_dumps = bucket.get_object(acc_name).read()
                in_com_e3 = time.perf_counter()
                in_com_time += in_com_e3 - in_com_s3

                
                real_comp_st = time.perf_counter()
                ac = pickle.loads(accum_count_this_client_dumps)
                accum_count_list.append(ac)
                print(f"add accum_count {i}\n")
                real_comp_ed = time.perf_counter()
                real_comp += real_comp_ed - real_comp_st

        #聚合操作2：聚合
        #count_dict_dumps = bucket.get_object('count_dict.txt').read()
        #count_dict = pickle.loads(count_dict_dumps)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}accum_count_list:   {accum_count_list}")
        pk_ctx.relin_keys()
        #sum_count = sum(count_dict)
        real_comp_st = time.perf_counter()
        accum_count = sum(accum_count_list)
        sum_enc_params = sum(params_list)
        latest_enc_params = 1 / accum_count * sum_enc_params  
        latest_enc_params = latest_enc_params.serialize()
        real_comp_ed = time.perf_counter()
        real_comp += real_comp_ed - real_comp_st

        in_com_s4 = time.perf_counter()
        if bucket.object_exists('init_list_done.txt'):
            bucket.delete_object('init_list_done.txt')
        in_com_e4 = time.perf_counter()
        in_com_time += in_com_e4 - in_com_s4
        

        if bucket.object_exists('average_done.txt')==False:
            
            #更新OSS中的global_params    避免会触发sel立马清掉标志
            new_global_parmas_dumps = pickle.dumps(latest_enc_params)

            in_com_s5 = time.perf_counter()
            bucket.put_object('global_params.txt', new_global_parmas_dumps)
            bucket.put_object('average_done.txt', s_d)
            in_com_e5 = time.perf_counter()
            in_com_time += in_com_e5 - in_com_s5


            #统计运行时间
            auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
            bucket = oss2.Bucket(auth, "ADD_by_yourself", "test-tpc-h")
            run_end = time.perf_counter()
            run_time = run_end - run_start -in_com_time
            if bucket.object_exists('hfl_aggre_time_record.txt')==True:
                time_record_dumps = bucket.get_object('hfl_aggre_time_record.txt').read()
                time_record = pickle.loads(time_record_dumps)
                time_record.append(real_comp)
            else:
                time_record = [real_comp]
            time_record_dumps = pickle.dumps(time_record)
            bucket.put_object('hfl_aggre_time_record.txt', time_record_dumps)

