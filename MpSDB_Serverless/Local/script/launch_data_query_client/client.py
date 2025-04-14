import json
import multiprocessing
import os
import sys
import time
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import transmission.tenseal.tenseal_client_proxy_pb2 as tenseal_client_proxy_pb2
import transmission.tenseal.tenseal_client_proxy_pb2_grpc as tenseal_client_proxy_pb2_grpc
import grpc
import pickle
import oss2
def read_ram_user_info(file_path):
    with open(file_path, 'r') as file:
        ram_user_info = json.load(file)
    return ram_user_info

def send_request(_cid, _qid, _db_name, _table_name, _column_name, _op, result_queue):
    print(f"start:{_cid}")
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    channel = grpc.insecure_channel('127.0.0.1:60060', options=options)
    stub = tenseal_client_proxy_pb2_grpc.ClientProxyServiceStub(channel)
    start_time = time.time()
    # request = tenseal_client_proxy_pb2.query_msg_client(cid=1, qid=3457, db_name="total", column_name="value_2",
    #                                                     op="VAR_mode",
    #                                                     table_name="table_1")
    request = tenseal_client_proxy_pb2.query_msg_client(cid=_cid, qid=_qid, db_name=_db_name, table_name=_table_name,
                                                            column_name=_column_name, op=_op)
    print(f"send!")
    respose = stub.data_query(request)
    result = pickle.loads(respose.dec_result)
    end_time = time.time()
    print(f"cid:{_cid}  type:{_db_name}\ntime: {end_time-start_time}")
    result_queue.put((_cid, round(end_time - start_time, 4)))
    #print(result)
    if request.column_name == "id":
        print("id:")
        for i in result:
            print(round(i))
    elif request.column_name == "COUNT(*)":
        print(f"cnt:\n{round(result[0])}")
    else:
        print(f"{request.op}:{round(result[0],3)}")

def do_query(_cid, _qid, _is_total, _table_name, _column_name, _op, result_queue):
        client_to_parse_list = {}
        client_to_parse_list["cid"] = _cid
        client_to_parse_list["qid"] = _qid
        client_to_parse_list["column_name"] = _column_name
        client_to_parse_list["op"] = _op
        client_to_parse_list["table_name"] = _table_name
        
        client_to_parse_list["is_total"] = _is_total
        client_to_parse_list["db_name"] = "osm_a_"#"conclave_db_3_3_" #"osm_a_"
        client_to_parse_list["ip_address"] = ""
        client_to_parse_list["client_index_list"] = [0, 1 ,2]
        client_to_parse_list["query_name"] = "normal"
        print(client_to_parse_list)
        client_to_parse_list_d = pickle.dumps(client_to_parse_list)
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
        admin_bucket = oss2.Bucket(auth, "ADD_by_yourself", "fc-computes") 


        file_path = 'ADD_by_yourself/FaaS_FL2023_async/faas_fed/set_ram/key/query_user_' + str(client_to_parse_list["cid"]) + '.txt'  
        ram_user_info = read_ram_user_info(file_path)
        # 提取子 RAM 用户信息
        new_ram_access_key_id = ram_user_info.get('key-id')
        new_ram_access_key_secret = ram_user_info.get('key-secret')
        user_auth = oss2.Auth(new_ram_access_key_id, new_ram_access_key_secret)
        bucket = oss2.Bucket(auth, "ADD_by_yourself", "clients-up") 
        print(new_ram_access_key_id)

        #清理计算空间
        for obj in oss2.ObjectIterator(admin_bucket, prefix='query_user_' + str(client_to_parse_list["cid"]) + '/'):
            print(f"delete {obj.key}")
            admin_bucket.delete_object(obj.key)

        print('query_user_' + str(client_to_parse_list["cid"]) + '/')
        #清理用户空间
        #t = bucket.get_object("query_user_1/query_client_to_parse.txt").read()
        # t = bucket.get_object("in1.csv").read()
        # print(t)
        # print("\nstart\n")
        # for i in bucket.list_objects("query_user_1/"):
        #     print(i.key)
        for obj in oss2.ObjectIterator(bucket, prefix='query_user_' + str(client_to_parse_list["cid"]) + '/'):
            print(f"delete {obj.key}")
            bucket.delete_object(obj.key)  

        bucket.put_object("query_user_"+str(client_to_parse_list["cid"])+"/query_client_to_parse.txt",client_to_parse_list_d)

        put_time = round(t.time(),5)

        while bucket.object_exists("query_user_" + str(client_to_parse_list["cid"]) + "/key_server_to_client.txt") == False:
            t.sleep(0.001)
        get_time = round(t.time(),5)
        
        print(f"test_time_put_to_parse: {put_time}")
        print(f"test_time_get_from_keyserver: {get_time}")
        
        ans_dict = pickle.loads(bucket.get_object("query_user_" + str(client_to_parse_list["cid"]) + "/key_server_to_client.txt").read())
        if _cid == ans_dict["cid"] and _qid == ans_dict["qid"]:
            ans = ans_dict["result"]    
        print(type(ans))  
        response = tenseal_client_proxy_pb2.dec_query_result(dec_result=ans)
        return response
        
def start(query_num):
    #单进程
    # #设置每个任务的参数
    # table_name_ = "osm_a"#"table_1"#"osm_a"
    # op_ = "DWithin(Point(121.5, 14.5), location, 0.5)"#"DWithin(Point(121.5, 14.5), location, 0.5)" #"sum"  #  或 "KNN(Point(121.5, 14.5), location, 8)"
    # column_name_ = "COUNT(*)"#"column_1"  #"id" 或 "COUNT(*)"
    # send_request(1, 3457, "total", table_name_ , column_name_, op_, [])

    #asyncio.run(main()) 
    # 创建多个进程
    result_queue = multiprocessing.Queue()
    processes = []
    #设置每个任务的参数
    table_name_ = "osm_a"#"table_1"#"osm_a"
    op_ = "KNN(Point(121.5, 14.5), location, 8)"#"DWithin(Point(121.5, 14.5), location, 0.5)" #"sum"  #  或 "KNN(Point(121.5, 14.5), location, 8)"
    column_name_ = "id"#"column_1"  #"id" 或 "COUNT(*)"
    # params = [(1, 3457, "total", table_name_, column_name_, op_),
    #           (2, 3458, "DATABASE_1", table_name_, column_name_, op_),
    #           (3, 3459, "DATABASE_2", table_name_, column_name_, op_),
    #           (4, 3460, "DATABASE_3", table_name_, column_name_, op_),
    #           (5, 3461, "total", table_name_, column_name_, op_),
    #           (6, 3462, "total", table_name_, column_name_, op_)
    #           ]
    
    # test:  avg--ok  sum--ok  max--ok  min--ok  VARIANCE--ok  STD--ok  VAR_SAMP--ok    STDDEV_SAMP--ok VAR_MEDIAN--ok  VAR_MODE--ok
    # Dwithin  COUNT(*)--ok 1.5s 22X   
    # Dwithin  id--ok    4.8s  200X
    # KNN  id--ok    5.6s  200X
    # KN_B_N  id--ok    20.5s  200X

    
    params = []
    for i in range(query_num):
        #params.append((1+i, 3457+i, "total", "table_1", "value_2", "VAR_MODE"))
        params.append((1+i, 3457+i, "total", table_name_ , column_name_, op_ ))
    s_time = time.time()
    
    # 启动多个进程
    for param in params:
        process = multiprocessing.Process(target=send_request, args=(param[0], param[1], param[2], param[3], param[4], param[5], result_queue))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()
    e_time = time.time()
    print(f"total:{e_time - s_time}")
    #处理时间记录
    send_request_times = []
    while not result_queue.empty():
        cid, time_value = result_queue.get()
        send_request_times.append((cid, time_value))

    # Calculate min, max, and average times
    min_time = min(send_request_times, key=lambda x: x[1])
    max_time = max(send_request_times, key=lambda x: x[1])
    total_time = round(e_time - s_time, 4)

    # Write results to a file
    with open('ADD_by_yourself/FaaS_FL2023/faas_fed/script/launch_data_query_client/mulit/time_knn_' + str(query_num) + '.txt', 'w') as file:
        file.write(f' Total Time: {total_time}\nOne-Min CID: {min_time[0]} Time: {min_time[1]}   One-Max CID: {max_time[0]} Time: {max_time[1]}\n')
        for (cid, time_value) in send_request_times:
            file.write(f"{cid}:{time_value}\n")
    return total_time

if __name__ == "__main__":
    time_list = []
    t = start(10)
    time_list.append(t)
    # time.sleep(20)
    # for i in range[1]:
    #     t = start(i*5)
    #     time_list.append(t)
    #     time.sleep(20)
    print(time_list)
#test rang-query  配置同下
#KN_B_N(Point(121.5, 14.5), location, 8)   28.7s
#KNN(Point(121.5, 14.5), location, 8)      3.45s
#id  DWithin(Point(121.5, 14.5), location, 0.5)  2.83s
#COUNT(*)      DWithin(Point(121.5, 14.5), location, 0.5)   3.488s



#4vcpu  上传到目录  保留dataServer实例
# total time:  2.45s
# {'put-time': 0.37253618240356445, 'wait_for_data_query': 0.6767723560333252, 'read_data_query': 0.13933706283569336}
# {'generate_sql': 5.602836608886719e-05, 'query': 0.029389619827270508, 'enc': 0.008907556533813477}
# {'read_pk': 0.03294658660888672, 'read_sk': 0.019388914108276367}
# {'read_pk': 0.02049875259399414, 'read_sk': 0.01962566375732422}
# {'read_pk': 0.019012451171875, 'read_sk': 0.021467208862304688}
# test_time_put_to_parse: 1705723632.52875
# test_time_put_to_data_0:  1705723627.38953
# {'get_oss_inf': 0.0005300045013427734, 'read_pk': 0.0227200984954834, 'trans_pk': 0.20858001708984375}

# test_time_put_to_data_0:  1705723627.38953
# test_time_put_to_data_1:  1705723627.5183
# test_time_put_to_data_2:  1705723627.61411

# test_time_put_to_keyserver: 1705723634.14056
# test_time_get_from_keyserver: 1705723634.47052


#4vcpu      保留dataServer实例  data读pksk挂载，parse用bucket
#exp1:  4.35s
#{'put-time': 0.5988819599151611, 'wait_for_data_query': 1.3011488914489746, 'read_data_query': 0.08659863471984863}
# {'read_pk': 0.366119384765625, 'trans_pk': 0.23221182823181152}   
# {'generate_sql': 6.175041198730469e-05, 'query': 0.019794702529907227, 'enc': 0.008983135223388672}
# {'read_pk': 0.32126760482788086, 'trans_pk': 0.21379327774047852}
# {'generate_sql': 5.269050598144531e-05, 'query': 0.033135175704956055, 'enc': 0.009372472763061523}
# {'read_pk': 0.34911203384399414, 'trans_pk': 0.25455522537231445}
# {'generate_sql': 6.0558319091796875e-05, 'query': 0.030091285705566406, 'enc': 0.009130239486694336}
# test_time_put_to_parse: 1705720479.73979
# test_time_put_to_data_2:  1705720481.13423
# test_time_read_all: 1705720482.43519
# test_time_put_to_keyserver: 1705720482.72779
# test_time_get_from_keyserver: 1705720483.52492

#exp1:  4.70s

#data 4vcpu 存目录读取pk时间  (0.018, 0.024, 0.026, 0.025)  
#data 4vcpu 挂载读取pk时间  (0.26, 0.317,  0.460， 0.21， 0.27， 0.33， 0.32， 0.36)  
#data 4vcpu bucket读取pk时间  (0.37, 0.52,  0.44， 0.40， 0.36， 0.37， 0.38， 0.34)  
    
#parse 1vcpu bucket读取pk时间（0.48 , 0.46）
#parse 4vcpu bucket读取pk时间（0.35 , 0.41）
    
#VARIANCE 4.222  9.0  222.0
#gRPC：  avg--4.22  sum--38.0  max--8.0  min--1.0  VARIANCE--6.84  STD--2.615  VAR_SAMP--7.694    STDDEV_SAMP--2.774   VAR_MEDIAN--3.0  VAR_MODE--ok
    
#FaaS：  avg--4.222  sum--38.0  max--8.0  min--1.0  VARIANCE--6.84  STD--2.615  VAR_SAMP--7.694    STDDEV_SAMP--2.774   VAR_MEDIAN--3.0  VAR_MODE--ok
    
#time: 
#grpc 
# avg--1.2  sum--0.58  max--0.6  min--0.6  VARIANCE--3  STD--3  VAR_SAMP--  3+-0.2    STDDEV_SAMP--3   VAR_MEDIAN--21    VAR_MODE--ok

#faas
# 


#sum为例
#保留实例
#只分析parse
#time: 5.221481800079346   {'put-time': 0.3926117420196533, 'wait_for_data_query': 2.309051513671875, 'read_data_query': 0.1007840633392334}
#time: 6.54990816116333    {'put-time': 0.638641357421875, 'wait_for_data_query': 3.5985898971557617, 'read_data_query': 0.1284651756286621}

#分析parse 和 dataServer
    
    
# time: 6.753708839416504
# parse:
# {'put-time': 0.4254906177520752, 'wait_for_data_query': 3.0988402366638184, 'read_data_query': 0.1271672248840332}
# dataserver:
# {'read_pk': 1.1215753555297852, 'read_sk': 0.07888388633728027, 'ini': 2.9261300563812256, 'function': 0.09874200820922852, 'upload and delete': 0.0940248966217041}
# {'generate_sql': 0.00005793571472167969, 'query': 0.08903646469116211, 'enc': 0.009441852569580078}
# OSS通信：     0.001轮询
# test_time_put_to_parse: 1705571786.9317
# test_time_parse_start: 1705571786.96845

# test_time_put_to_data_0:  1705571787.8318
# test_time_data_start: 1705571788.098
# test_time_put_to_data_1:  1705571787.966
# test_time_data_start: 1705571788.09543
# test_time_put_to_data_2:  1705571788.06696
# test_time_data_start: 1705571788.09345

# test_time_data_1_end: 1705571790.89204
# test_time_data_2_end: 1705571790.94652
# test_time_data_0_end: 1705571791.16712

# test_time_read_all: 1705571791.16578

# test_time_put_to_keyserver: 1705571791.44474
# test_time_key_start: 1705571791.95195

# test_time_to_client: 1705571793.04901
# test_time_get_from_keyserver: 1705571793.09577


    
#ex1:
    
#time: 5.959700584411621 
#{'put-time': 0.3335568904876709, 'wait_for_data_query': 2.3050975799560547, 'read_data_query': 0.10222792625427246} 
#{'read_pk': 0.5225739479064941, 'read_sk': 0.01941704750061035, 'ini': 1.2533750534057617, 'function': 0.030248165130615234, 'upload and delete': 0.11864352226257324}
#{'generate_sql': 0.00022220611572265625, 'query': 0.02002429962158203, 'enc': 0.00990605354309082}

#ex2:
    
#time: 6.8588409423828125
#parser:
#{'put-time': 0.42223119735717773, 'wait_for_data_query': 3.4011518955230713, 'read_data_query': 0.12215375900268555}
#dataserver:
#{'read_pk': 1.387556791305542, 'read_sk': 0.03517866134643555, 'ini': 3.170210599899292, 'function': 0.03789258003234863, 'upload and delete': 0.14457225799560547}
#{'generate_sql': 0.00005435943603515625, 'query': 0.0286712646484375, 'enc': 0.00899648666381836}
# test_time_put_to_parse: 1705571088.2788
# test_time_parse_start: 1705571088.73017

# test_time_put_to_data_0:  1705571089.70261
# test_time_data_start: 1705571089.7297
# test_time_put_to_data_1:  1705571089.81815
# test_time_data_start: 1705571089.8177
# test_time_put_to_data_2:  1705571089.95175
# test_time_data_start: 1705571089.95174


# test_time_read_all: 1705571093.35282

# test_time_put_to_keyserver: 1705571093.59577
# test_time_get_from_keyserver: 1705571094.68193
    '''
    #asyncio.run(main()) 
    # 创建多个进程
    result_queue = multiprocessing.Queue()
    processes = []

    # params = [(1, 3457, "total", table_name_, column_name_, op_),
    #           (2, 3458, "DATABASE_1", table_name_, column_name_, op_),
    #           (3, 3459, "DATABASE_2", table_name_, column_name_, op_),
    #           (4, 3460, "DATABASE_3", table_name_, column_name_, op_),
    #           (5, 3461, "total", table_name_, column_name_, op_),
    #           (6, 3462, "total", table_name_, column_name_, op_)
    #           ]
    

    # test:  avg--ok  sum--ok  max--ok  min--ok  VARIANCE--ok  STD--ok  VAR_SAMP--ok    STDDEV_SAMP--ok VAR_MEDIAN--ok  VAR_MODE--ok
    # Dwithin  COUNT(*)--ok 1.5s 22X   
    # Dwithin  id--ok    4.8s  200X
    # KNN  id--ok    5.6s  200X
    # KN_B_N  id--ok    20.5s  200X

    
    params = []
    for i in range(1):
        #params.append((1+i, 3457+i, "total", "table_1", "value_2", "VAR_MODE"))
        params.append((1+i, 3457+i, "total", table_name_ , column_name_, op_ ))
    s_time = time.time()
    
    # 启动多个进程
    for param in params:
        process = multiprocessing.Process(target=send_request, args=(param[0], param[1], param[2], param[3], param[4], param[5], result_queue))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()
    e_time = time.time()
    print(f"total:{e_time - s_time}")

    # #处理时间记录
    # send_request_times = []
    # while not result_queue.empty():
    #     cid, time_value = result_queue.get()
    #     send_request_times.append((cid, time_value))

    # # Calculate min, max, and average times
    # min_time = min(send_request_times, key=lambda x: x[1])
    # max_time = max(send_request_times, key=lambda x: x[1])
    # total_time = round(e_time - s_time, 4)

    # # Write results to a file
    # with open('ADD_by_yourself/script/launch_data_query_server/result/time.txt', 'w') as file:
    #     file.write(f' Total Time: {total_time}\nOne-Min CID: {min_time[0]} Time: {min_time[1]}   One-Max CID: {max_time[0]} Time: {max_time[1]}\n')
    #     for (cid, time_value) in send_request_times:
    #         file.write(f"{cid}:{time_value}\n")
    '''


'''
table_name = "osm_a"
op = "DWithin(Point(121.5, 14.5), location, 0.5)"  或 “KNN(Point(121.5, 14.5), location, 8)”
column_name = "id" 或 "COUNT(*)"


test
faas \ grpc

var_samp   [9179610.082670707]  \  [9179610.082673874]
max        [9786.00000000087]   \  [9786.00000000116]
min        [345.0000000000494]  \  [344.9999999981719]
avg        [4300.199999778068]  \  [4300.199999628872]
VARIANCE   [8873623.086642087]  \  [8873623.07627677]
STDDEV     [2978.862717016378]  \  [2978.8627167341137]
STD        [2978.8627162567172] \  [2978.8627157821225]
STDDEV_SAMP[3029.7871342115895] \  [3029.787134856168]
VAR_MEDIAN
           [10.000000001487896]  \  [10.000000001285507]    median_test_no_duplication.sql
           [2.000000001830017]  \  [2.000000002033734]   、[1.999999999961811]         median_test.sql
VAR_MODE
           [29.99999999908782, 20.00000000006412, 9.999999996868668] \ [19.99999999716672, 10.00000000069337, 30.000000000646025 multi_mode_test.sql
           [20.000000001163507] \  [19.99999999870935]    no_common_element_mode_test.sql
'''