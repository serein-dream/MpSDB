import oss2, json
import pickle
import time,datetime
import random
import numpy as np
import tenseal as ts

def handler(event, context):
    creds = context.credentials
    auth=oss2.StsAuth(creds.access_key_id, creds.access_key_secret, creds.security_token)
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = 'oss-' +  oss_raw_data['events'][0]['region'] + '-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    

    object_name = oss_info_map['object']['key']
    tt = random.randint(1, 10)
    print(f"\nby:  {object_name}\n")
    lr = 0.5
    adaptive_lr = None
    momentum_rate = 0.5


    time.sleep(tt*0.05)
    selection_dumps = bucket.get_object('selection.txt').read()
    To_average_list = pickle.loads(selection_dumps)[1]
    this_rnd = pickle.loads(selection_dumps)[0][0]
    do_average = True
    
    print(f"\nrnd:  {this_rnd}    To_average_list:  {To_average_list}\n")
    
    s_d = pickle.dumps('s')
    while bucket.object_exists("reading.txt") == True:
        time.sleep(0.05)
        print("waiting for read...\n")
        if bucket.object_exists('doing_aggre.txt')==True:
            return
    bucket.put_object("reading.txt", s_d)


    if bucket.object_exists('doing_aggre.txt')==True:
        print("删除reading.txt")
        if bucket.object_exists("reading.txt"):
            bucket.delete_object("reading.txt")
        return        
    else:
        print("reading...\n")

        for i in range(len(To_average_list)):
            if To_average_list[i]>0 :
                msg_name = "clients_k_update_params_" + str(i) + ".txt"
                if bucket.object_exists(msg_name) == False:
                    do_average = False
        if do_average == False:
            if bucket.object_exists("reading.txt"):
                bucket.delete_object("reading.txt")
            return        

        print(f"\ndo_average?  {do_average}\nbucket.object_exists('doing_aggre.txt'):   {bucket.object_exists('doing_aggre.txt')}\n")

        if do_average == True and bucket.object_exists('doing_aggre.txt')==False and bucket.object_exists('average_done.txt')==False:
            bucket.put_object('doing_aggre.txt', s_d)

            t = datetime.datetime.now().strftime(f'%Y-%m-%d %H:%M:%S{r".%f"}')
            print(f"have put 'doing_aggre.txt' time:  {t}")

            time.sleep(0.1)
            if bucket.object_exists("reading.txt"):
                bucket.delete_object("reading.txt")
            
                t = datetime.datetime.now().strftime(f'%Y-%m-%d %H:%M:%S{r".%f"}')
                print(f"delete_object 'reading.txt':  {t}")

            print("\n\ndo\n\n")
 
            #加载密钥
            pk_ctx_bytes = open('/code/tenseal/ts_ckks_pk.config', "rb").read()
            pk_ctx = ts.context_from(pk_ctx_bytes)
            reline_keys = pk_ctx.relin_keys()
           
            #开始聚合
                
            #聚合操作1:收集参数

            num = 0
            for i in range(len(To_average_list)):
                if To_average_list[i]>0:
                    name = "clients_k_update_params_" + str(i) + ".txt"
                    clients_update_params_s = bucket.get_object(name).read()
                    enc_clients_updates_numpy = ts.ckks_vector_from(pk_ctx, clients_update_params_s)
                    if num == 0:
                        centroids_sum = enc_clients_updates_numpy
                    else:
                        centroids_sum += enc_clients_updates_numpy
                    num += 1
                    print(f"add params {i}\n")

            num = 0
            for i in range(len(To_average_list)):
                if To_average_list[i]>0 :
                    acc_name = "client_counts_" + str(i) + ".txt"
                    accum_count_this_client_dumps = bucket.get_object(acc_name).read()
                    ac = pickle.loads(accum_count_this_client_dumps)
                    if num == 0:
                        counts = ac
                    else:
                        counts += ac
                    num += 1
                    print(f"add accum_count {i}\n  the num:  {ac}\n  total num: {counts}\n")
            
            for i in range(len(counts)):
                if counts[i] == 0:
                    counts[i] = 1
            print(f"counts:  {counts}")
            #聚合操作2：聚合
            new_enc_centroids = centroids_sum 
            ans = {}
            ans["new_enc_centroids_s"] = new_enc_centroids.serialize()
            ans["total_counts"] = counts
            ans_s = pickle.dumps(ans)
            if bucket.object_exists('init_list_done.txt'):
                bucket.delete_object('init_list_done.txt')

            if bucket.object_exists('average_done.txt')==False:
                
                #更新OSS中的global_params    避免会触发sel立马清掉标志
                bucket.put_object('global_params.txt', ans_s)

                bucket.put_object('average_done.txt', s_d)
