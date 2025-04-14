import oss2, json
import pickle
import os,sys
import time
sys.path.insert(1,(os.path.dirname(os.path.abspath(__file__))))
from model import CNN_MiddleModel
from model import MLP_MiddleModel
import torch
import tenseal as ts

def get_device():
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def concat_bottom_func(bottom_output_dict, num_clients):
    bottom_output_list = []
    for i in range(num_clients):        
        bottom_output_list.append(bottom_output_dict[i])
        print(f"bottom_output_dict[{i}] shape:  {bottom_output_dict[i].shape}")
    concat_bottom = torch.cat(bottom_output_list, dim=-1)
    return concat_bottom, bottom_output_list

def add_bottom(bottom_output_dict, num_clients):
    concat_bottom = bottom_output_dict[0]
    for i in range(1, num_clients):
        concat_bottom = concat_bottom + bottom_output_dict[i]
    return concat_bottom

def split_bottom_grads(bottom_grads, bottom_output_list):
    shape_list = []
    for tensor in bottom_output_list:
        if len(tensor.shape) == 1:
            shape_list.append(tensor.shape[0])
        else:
            shape_list.append(tensor.shape[-1])
    return list(torch.split(bottom_grads, shape_list, dim=-1))

def handler(event, context):  
    ini_time = 0
    communicate_time = 0
    rnd1_time = 0   
    real_comp = 0
    real_comp_list = []
    time1 = time.perf_counter()

    creds = context.credentials
    auth=oss2.StsAuth(creds.access_key_id, creds.access_key_secret, creds.security_token)
    oss_raw_data = json.loads(event)
    oss_info_map = oss_raw_data['events'][0]['oss']
    bucket_name = oss_info_map['bucket']['name']
    endpoint = 'oss-' +  oss_raw_data['events'][0]['region'] + '-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    time2 = time.perf_counter()
    ini_time = time2 - time1



    time3 = time.perf_counter()
    num_clients_S = bucket.get_object("num_clients.txt").read()
    time4 = time.perf_counter()
    communicate_time = communicate_time + time4 - time3
    
    comp_s = time.perf_counter()
    num_clients = pickle.loads(num_clients_S)

    print(f"num_clients : {num_clients}")
    #判断是否到齐（可浪费并行
    now_msg_num = 0
    time5 = time.perf_counter()
    for obj in oss2.ObjectIterator(bucket, prefix='vfl_from_bottom/'):
        now_msg_num = now_msg_num + 1
    time6 = time.perf_counter()
    communicate_time = communicate_time + time6 - time5
    
    print(f"now_msg_num: {now_msg_num}\n num_clients:{num_clients}   ")
    if now_msg_num < num_clients:
        return  

    comp_e = time.perf_counter()
    real_comp = real_comp + comp_e - comp_s
    real_comp_list.append(comp_e - comp_s)


    #到齐的话：
    name = "test_v_kmeans/db_score.txt"      
    if bucket.object_exists(name):
        bucket.delete_object(name)
    time7 = time.perf_counter()
    pk_ctx_bytes = open('/code/tenseal/ts_ckks_pk.config', "rb").read()
    pk_ctx = ts.context_from(pk_ctx_bytes)
    context_bytes = open('/code/tenseal/ts_ckks.config', "rb").read()
    S_ctx = ts.context_from(context_bytes)
    reline_keys = pk_ctx.relin_keys()
    time8 = time.perf_counter()
    ini_time = ini_time + time8 - time7

    for i in range(num_clients):
        name = "to_bottom_grad/rank_"+ str(i) +".txt" 
        time11 = time.perf_counter()
        if bucket.object_exists(name):
            bucket.delete_object(name)
        time12 = time.perf_counter()
        communicate_time = communicate_time + time12 - time11    



    time9 = time.perf_counter()
    request_dict_s = bucket.get_object("vfl_from_bottom/rank_0.txt").read()
    for i in range(num_clients):
        name = "to_bottom_grad/safe_"+ str(i) +".txt" 
        bucket.put_object(name,num_clients_S)
    time10 = time.perf_counter()
    communicate_time = communicate_time + time10 - time9

    comp_s = time.perf_counter()
    request_dict = pickle.loads(request_dict_s)
    args = request_dict["args"]
    comp_e = time.perf_counter()
    real_comp = real_comp + comp_e - comp_s
    real_comp_list.append(comp_e - comp_s)

    if args.model.upper()=="KMEANS":
        bottom_to_server_dict = request_dict
        not_change_epoch_min= bottom_to_server_dict['not_change_epoch']
        enc_global_sq_dist = ts.ckks_vector_from(pk_ctx, bottom_to_server_dict['enc_local_sq_dist_msg'])
        for i in range(0,num_clients):
            name = "vfl_from_bottom/rank_"+ str(i) +".txt"
            if i==0:
                bucket.delete_object(name)
                continue
            else:
                bottom_to_server_dict = pickle.loads(bucket.get_object(name).read())
                if not_change_epoch_min > bottom_to_server_dict['not_change_epoch']:
                    not_change_epoch_min = bottom_to_server_dict['not_change_epoch']
                enc_global_sq_dist += ts.ckks_vector_from(pk_ctx, bottom_to_server_dict['enc_local_sq_dist_msg'])
                bucket.delete_object(name)
        to_client_dict = {}
        to_client_dict['enc_global_sq_dist_msg'] = enc_global_sq_dist.serialize()
        to_client_dict['not_change_epoch'] = not_change_epoch_min
        to_client_dict_d = pickle.dumps(to_client_dict)
        for i in range(num_clients):
            name = "to_bottom_grad/rank_"+ str(i) +".txt"    
            bucket.put_object(name, to_client_dict_d)        
        

    else:    
        comp_s = time.perf_counter()
        rnd = request_dict["rnd"]
        epoch = request_dict["epoch"]
        bottom_output_dict = {}
        if "CRY" in args.model.upper():
            bottom_output_dict[0] = ts.ckks_vector_from(pk_ctx, request_dict["bottom_output"])  
        comp_e = time.perf_counter()
        real_comp = real_comp + comp_e - comp_s
        real_comp_list.append(comp_e - comp_s)

        for i in range(0,num_clients):
            name = "vfl_from_bottom/rank_"+ str(i) +".txt"

            time11 = time.perf_counter()
            request_dict_s = bucket.get_object(name).read()
            time12 = time.perf_counter()
            communicate_time = communicate_time + time12 - time11

            comp_s = time.perf_counter()
            request_dict = pickle.loads(request_dict_s)
            if "CRY" in args.model.upper():
                bottom_vector = ts.ckks_vector_from(pk_ctx, request_dict["bottom_output"])
                bottom_output_dict[i] = bottom_vector
                # print(f"bottom_vector:{i}  {type(bottom_vector)}")
            else:
                bottom_vector = request_dict["bottom_output"]
                bottom_output_dict[i] = bottom_vector
                # print(f"add : {i}")
            comp_e = time.perf_counter()
            real_comp = real_comp + comp_e - comp_s
            real_comp_list.append(comp_e - comp_s)

            time11 = time.perf_counter()
            bucket.delete_object(name)
            time12 = time.perf_counter()
            communicate_time = communicate_time + time12 - time11

            
        if args.model.upper()=="CNN" or args.model.upper()=="MLP":
            device = get_device()
            if args.model.upper()=="CNN":
                middle_model = CNN_MiddleModel()
            else:
                middle_model = MLP_MiddleModel()
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(middle_model.parameters(), lr=args.lr,
                                        momentum=0.5)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(middle_model.parameters(), lr=args.lr,
                                        weight_decay=1e-4)
            
            comp_s = time.perf_counter()
            if rnd>1 or epoch>1:
                model_state_dict = pickle.loads(bucket.get_object("middle_model_state_dict.txt").read())
                comp_s = time.perf_counter()
                middle_model.load_state_dict(model_state_dict)
            middle_model.to(device)

            concat_bottom, bottom_output_list = concat_bottom_func(bottom_output_dict, num_clients)
            middle_model.train()
            middle_output = middle_model(concat_bottom)
        elif "LINEAR" in args.model.upper() or "LR" in args.model.upper():
            comp_s = time.perf_counter()
            middle_output = add_bottom(bottom_output_dict, num_clients)
            
        comp_e = time.perf_counter()
        real_comp = real_comp + comp_e - comp_s
        real_comp_list.append(comp_e - comp_s)
        #鉴于冷启动时间>=label owner时间  这里等待
        middle_to_top_dict = {}
        middle_to_top_dict["rnd"] = rnd
        middle_to_top_dict["epoch"] = epoch
        if "CRY" in args.model.upper():
            middle_to_top_dict["middle_output"] = middle_output.serialize()
        else:
            print(F"middle_output:{type(middle_output)}")
            middle_to_top_dict["middle_output"] = pickle.dumps(middle_output)
        middle_to_top_dict_d = pickle.dumps(middle_to_top_dict)
                    

        middle_com_time_start = time.perf_counter()
        time11 = time.perf_counter()
        bucket.put_object("vfl_from_middle_to_top.txt", middle_to_top_dict_d)#label owner那里记得得到就删
        time12 = time.perf_counter()
        communicate_time = communicate_time + time12 - time11 

        print(f"waiting for top_middle")   
        time11 = time.perf_counter()
        while bucket.object_exists("vfl_from_top_model.txt") == False:
            time.sleep(0.1)#内网免费
        time12 = time.perf_counter()
        communicate_time = communicate_time + time12 - time11 

        if "CRY" in args.model.upper():
            time11 = time.perf_counter()
            middle_grad = bucket.get_object("vfl_from_top_model.txt").read()
            time12 = time.perf_counter()
            communicate_time = communicate_time + time12 - time11 



        else:
            time_11 = time.perf_counter()
            middle_grad_s = bucket.get_object("vfl_from_top_model.txt").read()
            time12 = time.perf_counter()
            communicate_time = communicate_time + time12 - time11 

            comp_s = time.perf_counter()
            middle_grad = pickle.loads(middle_grad_s)
            comp_e = time.perf_counter()
            real_comp = real_comp + comp_e - comp_s
            real_comp_list.append(comp_e - comp_s)  
        middle_com_time_end = time.perf_counter()
        middle_com_time = middle_com_time_end - middle_com_time_start

        time11 = time.perf_counter()
        bucket.delete_object("vfl_from_top_model.txt")
        time12 = time.perf_counter()
        communicate_time = communicate_time + time12 - time11 


        comp_s = time.perf_counter()
        if args.model.upper() =="CNN" or args.model.upper() =="MLP":
            #bp
            concat_bottom.retain_grad()
            optimizer.zero_grad()
            middle_output.backward(middle_grad)
            optimizer.step()
            bottom_grads = concat_bottom.grad
            #split
            bottom_grad_list = split_bottom_grads(bottom_grads, bottom_output_list)
            #用于下一次函数加载 
            comp_e = time.perf_counter()

            middle_model_state_dict_d = pickle.dumps(middle_model.state_dict())
   
            if epoch % 1 == 0 and rnd == args.rnd:
                middle_model_state_dict_d = pickle.dumps(middle_model.state_dict())
                time11 = time.perf_counter()
                bucket.put_object("test_model/"+ str(args.model).upper() +"_middle_model_state_dict_"+ str(epoch) +".txt", middle_model_state_dict_d)
                time12 = time.perf_counter()
                communicate_time = communicate_time + time12 - time11 
                
            time11 = time.perf_counter()
            bucket.put_object("middle_model_state_dict.txt", middle_model_state_dict_d)
            time12 = time.perf_counter()
            communicate_time = communicate_time + time12 - time11 

        else:#注意排除kmeans
            bottom_grad_list = []
            for i in range(num_clients):
                bottom_grad_list.append(middle_grad)
            comp_e = time.perf_counter()

        
        real_comp = real_comp + comp_e - comp_s
        real_comp_list.append(comp_e - comp_s)
        real_comp_list.append(1)
        for i in range(num_clients):
            comp_s = time.perf_counter()
            if "CRY" in args.model.upper():
                d = middle_grad
            else:
                d = pickle.dumps(bottom_grad_list[i])
            name = "to_bottom_grad/rank_"+ str(i) +".txt"     
            comp_e = time.perf_counter()
            real_comp = real_comp + comp_e - comp_s
            real_comp_list.append(comp_e - comp_s)
            
            time11 = time.perf_counter()
            bucket.put_object(name, d)
            time12 = time.perf_counter()
            communicate_time = communicate_time + time12 - time11    
        end_time = time.perf_counter()

        

        all_time = end_time - time1
        compute_time = all_time - communicate_time - ini_time
        if bucket.object_exists("time_log.txt"):
            time_dict = pickle.loads(bucket.get_object("time_log.txt").read())
            time_dict["all_time"].append(all_time)
            time_dict["ini_time"].append(ini_time)
            time_dict["communicate_time"].append(communicate_time)
            time_dict["compute_time"].append(real_comp)
            time_dict["middle_com_time"].append(middle_com_time)
            time_dict["compute_time_list"].append(real_comp_list)
            time_dict_s = pickle.dumps(time_dict)
            bucket.put_object("time_log.txt", time_dict_s)
        else:
            time_dict = {}
            time_dict["all_time"] = []
            time_dict["ini_time"] = []
            time_dict["communicate_time"] = []
            time_dict["compute_time"] = []
            time_dict["middle_com_time"] = []
            time_dict["compute_time_list"] = []
            time_dict["all_time"].append(all_time)
            time_dict["ini_time"].append(ini_time)
            time_dict["communicate_time"].append(communicate_time)
            time_dict["compute_time"].append(real_comp)
            time_dict["middle_com_time"].append(middle_com_time)
            time_dict["compute_time_list"].append(real_comp_list)
            time_dict_s = pickle.dumps(time_dict)
            bucket.put_object("time_log.txt", time_dict_s)
            
        print(f"\nall_time: {time_dict}\n")
            


