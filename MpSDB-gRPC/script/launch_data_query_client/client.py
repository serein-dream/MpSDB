import asyncio
from datetime import datetime
import multiprocessing
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import transmission.tenseal.tenseal_client_proxy_pb2 as tenseal_client_proxy_pb2
import transmission.tenseal.tenseal_client_proxy_pb2_grpc as tenseal_client_proxy_pb2_grpc
import grpc
import pickle
import time
import pathlib

import hydra
from omegaconf import DictConfig


def send_request(_cid, _qid, _db_name, _table_name, _column_name, _op, start_time, result_queue, result_path):
    print(f"start:{_cid}")     
    _cid = int(_cid)
    _qid = int(_qid)
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    channel = grpc.insecure_channel('127.0.0.1:60060', options=options)
    stub = tenseal_client_proxy_pb2_grpc.ClientProxyServiceStub(channel)
    request = tenseal_client_proxy_pb2.query_msg_client(cid=_cid, qid=_qid, db_name=_db_name, table_name=_table_name,
                                                            column_name=_column_name, op=_op)
    
    response = stub.data_query(request)
    result = pickle.loads(response.dec_result)
    end_time = time.time()
    print(f"cid:{_cid}  type:{_db_name}\ntime: {end_time-start_time}")
    result_queue.put((_cid, round(end_time - start_time, 4)))

    # if _column_name == "id":
    #     print("id:")
    #     num = 0
    #     for i in result:
    #         #print(round(i))
    #         num = num+1
    #     print(f"len:{num}")
    #     with open(result_path, "a+") as file:          
    #         file.write(f"{num} ") 

    if _column_name == "id":
        print("id:")
        num = 0
        for i in result:
            with open(result_path, "a+") as file:          
                file.write(f"{i}, ") 

    elif "COUNT" in _column_name.upper():
        print(f"cnt:\n{round(result[0])}")
        with open(result_path, "a+") as file:          
            file.write(f"{round(result[0])} ")  
    else:
        print(f"{_op}:{result}")
        with open(result_path, "a+") as file:          
            file.write(f"{result}")  

    return round(end_time - start_time, 4)


def star_multi(query_num):
    #asyncio.run(main()) 
    # 创建多个进程
    result_queue = multiprocessing.Queue()
    processes = []
    #设置每个任务的参数
    table_name_ = "table_1"#"table_1"#"osm_a"
    op_ = "sum"#"DWithin(Point(121.5, 14.5), location, 0.5)" #"sum"  #  或 "KNN(Point(121.5, 14.5), location, 8)"
    column_name_ = "column_1"#"column_1"  #"id" 或 "COUNT(*)" 
 
    params = []
    for i in range(query_num):
        #params.append((1+i, 3457+i, "total", "table_1", "value_2", "VAR_MODE"))
        params.append((1+i, 3457+i, "total", table_name_ , column_name_, op_ ))
    
    # 启动多个进程
    for param in params:
        process = multiprocessing.Process(target=send_request, args=(param[0], param[1], param[2], param[3], param[4], param[5], 1.0, result_queue))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    #处理时间记录
    send_request_times = []
    total_time = 0
    while not result_queue.empty():
        cid, time_value = result_queue.get()
        send_request_times.append((cid, time_value))
        total_time = total_time + time_value

    avg_time = total_time / query_num

    print(f"send_request_times:{send_request_times}\navg_time:{avg_time}")

    # # Write results to a file
    # with open('/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/muliti/time__dis_count_' + str(query_num) + '_AVG20.txt', 'w') as file:
    #     file.write(f' Total Time: {total_time}\nOne-Min CID: {min_time[0]} Time: {min_time[1]}   One-Max CID: {max_time[0]} Time: {max_time[1]}\n')
    #     for (cid, time_value) in send_request_times:
    #         file.write(f"{cid}:{time_value}\n")
    return avg_time
def err_call_back(err):
        print(f'出错啦~ error：{str(err)}\n\n')
            
def star_pool(query_num, _op,  _table, _column_name, result_path, client_num_attend):
    #asyncio.run(main()) 
    # 创建多个进程
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    p = multiprocessing.Pool(4)
    #设置每个任务的参数
    table_name_ = _table #"table_1"#"table_1"#"osm_a"
    op_ = _op #"DWithin(Point(121.5, 14.5), location, 0.5)" #"sum"  #  或 "KNN(Point(121.5, 14.5), location, 8)"
    column_name_ = _column_name #"passenger_count"#"column_1"  #"id" 或 "COUNT(*)" 
 
    params = []
    for i in range(query_num):
         params.append((str(1+i), str(3457+i), "total"+str(client_num_attend), table_name_ , column_name_, op_ ))
    s_time = time.time()
    
    # 启动多个进程
    for param in params:
        ti = time.time()
        p.apply_async(send_request, args=(param[0], param[1], param[2], param[3], param[4], param[5], ti, result_queue, result_path), error_callback=err_call_back)
    p.close()
    p.join()
    
    e_time = time.time()
    print(f"total:{e_time - s_time}")
    #处理时间记录
    send_request_times = []
    time_list = []
    total_time = 0
    while not result_queue.empty():
        cid, time_value = result_queue.get()
        send_request_times.append((cid, time_value))
        time_list.append(time_value)
        total_time = total_time + time_value
    avg_time = total_time / query_num
    print(f"send_request_times:{send_request_times}\navg_time:{avg_time}")
    print(multiprocessing.cpu_count())


    # # Write results to a file
    # with open('/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/muliti/time__dis_count_' + str(query_num) + '_AVG20.txt', 'w') as file:
    #     file.write(f' Total Time: {total_time}\nOne-Min CID: {min_time[0]} Time: {min_time[1]}   One-Max CID: {max_time[0]} Time: {max_time[1]}\n')
    #     for (cid, time_value) in send_request_times:
    #         file.write(f"{cid}:{time_value}\n")
    return avg_time,send_request_times,time_list

def process_query_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化总和和当前查询的临时总和
    total_sum = {}
    r = []
    nm = []
    current_query_sum = 0.0

    # 是否当前正在处理查询中的数字
    processing_query = False

    for line in lines[1:]:  # 忽略第一行
        line = line.strip()
        if line == '' or "RDS" in line:
            continue

        if "query_num: " in line:
            nm.append(line)
            # 设置正在处理查询的标志
            processing_query = True
        elif "mysql_num:" in line:
            mysql_num_ = int(line.split(":")[-1].strip())
            ans = round(current_query_sum/mysql_num_, 5)
            total_sum[nm[-1]] = ans
            print(f"{nm[-1]} : {ans}\n")
            r.append(ans)
            current_query_sum = 0.0      
        else:
            # 如果当前正在处理查询中的数字，则将数字添加到当前查询的临时总和中
            if processing_query:
                current_query_sum += float(line)
    return r 

def test_iaas_vertical_oneRequest(table_name_list,data_name):
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    # x_values = []
    repeat = 10
    # table_name_ = "table_1"
    column_name_ = ""
    time_list = []
    all = {}

    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/tongji_all.txt", "a+") as file:
        file.write(f"\n{data_name}\n repeat:{repeat}\n")   
            
    for op_index in range(3,4):
        oo = op_index
        op_avg_list = []
        st = 1
        for j in range(5):
            table_name_ = table_name_list[j]
            if oo==0:
                op_ = "count_vertical"
                op_name = "count_vertical"
            elif oo==1:
                op_ = "sum_vertical"
                op_name = "sum_vertical"
            elif oo==2:
                op_ = "avg_vertical"
                op_name = "avg_vertical"
            elif oo==3:
                op_ = "std_vertical"
                op_name = "std_vertical"
            mulu = f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_{data_name}_bingx/"
            record_path = mulu + op_ + "_repeat_" + str(repeat) + "_record.txt"
            avg_time_path = mulu + op_ + "_repeat_" + str(repeat) + "_avg_time.txt"
            result_path = mulu + op_ + "_repeat_" + str(repeat) + "_result.txt"
            ruc_time_path = mulu + op_ + "_repeat_" + str(repeat) + "_ruc_time.txt"

            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(ruc_time_path)
            if not path.is_file():
                path.touch()

            if st == 1:
                with open(record_path, "a+") as file:
                    file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1,op:{op_}, table_name:{table_name_}\n\n")        
                with open(avg_time_path, "a+") as file:
                    # 获取当前时间
                    current_time = datetime.now()
                    # 格式化当前时间为字符串
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                    file.write(f"time:{formatted_time}\niaas_RDS_mysql, pool: 4,, RDS mysql fixed _RCU_1_1, op:{op_}, table_name:{table_name_}\n\n\n\n")                           
                with open(ruc_time_path, "a+") as file:
                    # 获取当前时间
                    current_time = datetime.now()
                    # 格式化当前时间为字符串
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                    file.write(f"vertical:{op_name}, start time:{formatted_time}\n") 
                with open(record_path, "a") as file:
                    file.write(f"\nop:{op_}, table_name:{table_name_}\n\n")
                st = 0
            

            tt = 0
            for j in range(repeat):
                with open(result_path, "a+") as file:          
                    file.write(f"\n\nop:{op_}, table_name:{table_name_}         repeat:{j}\n")  
                result_queue = multiprocessing.Queue()   
                t = time.time()
                t1 = send_request(1, 3456, "total3", table_name_, column_name_, op_, t, result_queue, result_path)
                tt = tt+t1
                time.sleep(1)
                with open(record_path, "a") as file:
                    file.write(f"{t1}, ")  
            avg = round(tt/repeat,4)
            op_avg_list.append(avg)
            with open(avg_time_path, "a+") as file:
                file.write(f"{avg}, ")
            time_list.append(avg)
            time.sleep(1)       
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
        with open(avg_time_path, "a+") as file:
            file.write(f"\n\n")
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/tongji_all.txt", "a+") as file:
            file.write(f"\n{op_}: {op_avg_list}\n")  
        all[op_] = op_avg_list     
        print(f"{op_name}:{op_avg_list}")    

    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/tongji_all.txt", "a+") as file:
        file.write(f"\n {all}\n\n")   

def test_iaas_h_oneRequest(table_name_list, data_name):
    # table_name_list = ["h_taxi_1k", "h_taxi_10k","h_taxi_100k","h_taxi_1m","h_taxi_10m"]
    repeat = 10
    if "tpch" in data_name:
        column_name_ = "size"
    else:
        column_name_ = "passenger_count"

    # column_name_ = "passenger_count"
    time_list = []
    all = {}
    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/tongji_all.txt", "a+") as file:
        file.write(f"\n{data_name}\n repeat:{repeat}\n")   
    for op_index in range(1):
        oo = op_index
        op_avg_list = []
        st = 1
        for j in range(1):
            table_name_ = table_name_list[j]
            if oo==0:
                op_ = "count"
                op_name = "count"
            elif oo==1:
                op_ = "avg"
                op_name = "avg"
            elif oo==2:
                op_ = "std"
                op_name = "std"
            elif oo==3:
                op_ = "sum"
                op_name = "sum"
            mulu = f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_{data_name}_bingx/"
            record_path = mulu + op_ + "_repeat_" + str(repeat) + "_record.txt"
            avg_time_path = mulu + op_ + "_repeat_" + str(repeat) + "_avg_time.txt"
            result_path = mulu + op_ + "_repeat_" + str(repeat) + "_result.txt"
            ruc_time_path = mulu + op_ + "_repeat_" + str(repeat) + "_ruc_time.txt"

            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(ruc_time_path)
            if not path.is_file():
                path.touch()

            if st == 1:
                with open(record_path, "a+") as file:
                    file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4_4,op:{op_}, table_name:{table_name_}\n\n")        
                with open(avg_time_path, "a+") as file:
                    # 获取当前时间
                    current_time = datetime.now()
                    # 格式化当前时间为字符串
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                    file.write(f"time:{formatted_time}\niaas_RDS_mysql, pool: 4,, RDS mysql fixed _RCU_4_4, op:{op_}, table_name:{table_name_}\n\n\n\n")                           
                with open(ruc_time_path, "a+") as file:
                    # 获取当前时间
                    current_time = datetime.now()
                    # 格式化当前时间为字符串
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                    file.write(f"vertical:{op_name}, start time:{formatted_time}\n") 
                with open(record_path, "a") as file:
                    file.write(f"\nop:{op_}, table_name:{table_name_} \n\n")
                st = 0
            
            tt = 0
            for j in range(repeat):
                with open(result_path, "a+") as file:          
                    file.write(f"\n\nop:{op_}, table_name:{table_name_}         repeat:{j}\n")  
                result_queue = multiprocessing.Queue()   
                t = time.time()
                t1 = send_request(1, 3456, "total3", table_name_, column_name_, op_, t, result_queue, result_path)
                tt = tt+t1
                time.sleep(1)
                with open(record_path, "a") as file:
                    file.write(f"{t1}, ")  
            avg = round(tt/repeat,4)
            op_avg_list.append(avg)
            with open(avg_time_path, "a+") as file:
                file.write(f"{avg}, ")
            time_list.append(avg)
            time.sleep(1)       
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
        with open(avg_time_path, "a+") as file:
            file.write(f"\n\n")
        with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/tongji_all.txt", "a+") as file:
            file.write(f"\n{op_}: {op_avg_list}\n")  
        all[op_] = op_avg_list     
        print(f"{op_name}:{op_avg_list}")    

    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/tongji_all.txt", "a+") as file:
        file.write(f"\n {all}\n\n")      
    # print(all)

def test_h_multi():
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    x_values = []
    repeat = 10
    table_name_ = "table_1"#"table_1"#"osm_a"
    column_name_ = "passenger_count"#"column_1"  #"id" 或 "COUNT(*)" 
    op_list = ["VARIANCE", "VAR_SAMP", "STDDEV_SAMP", "std"]
    #op_list = ["std", "count", "VARIANCE", "VAR_SAMP", "STDDEV_SAMP"]
    for _ in range(3):#重复进行三次
        for op_index in range(len(op_list)):
            time_list = []
            mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/taxi/multi_iaas_RDS_mysql_10e7/"
            record_path = mulu + op_list[op_index] + "_pool6_repeat_10_record.txt"
            avg_time_path = mulu + op_list[op_index] + "_pool6_repeat_10_avg_time.txt"
            then_query_time_path = mulu + op_list[op_index] + "_pool6_repeat_10_query_time.txt"
            query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(query_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(then_query_time_path)
            if not path.is_file():
                path.touch()
            with open(record_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 6, RDS mysql fixed _RCU_4_4, op:{op_list[op_index]}\n\n")
            with open(avg_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                file.write(f"time:{formatted_time}\niaas_RDS_mysql, pool: 6,, RDS mysql fixed _RCU_4_4, op:{op_list[op_index]}\n\n")                           
            with open(query_time_path, "a+") as f:
                f.write(f"iaas_RDS_mysql, pool: 6, RDS mysql fixed _RCU_4_4, op:{op_list[op_index]}\n\n")   
            if op_index == 0:
                a = 15
            else:
                a = 1      
            for i in range(a,30):
                tt = 0
                with open(record_path, "a") as file:
                    file.write(f"\nrequest num:{i}:\n")               
                with open(query_time_path, "a+") as f:
                    f.write(f"query_num: {i}\n")
                for j in range(repeat):
                    t, list_, _ = star_pool(i, op_list[op_index], table_name_, column_name_)
                    while len(list_)!= i:
                        t, list_, _ = star_pool(i, op_list[op_index], table_name_, column_name_)
                    tt = tt+t
                    time.sleep(5)
                    with open(record_path, "a") as file:
                        file.write(f"{list_}\n")
                avg = tt/repeat
                with open(avg_time_path, "a+") as file:
                    file.write(f"{avg}, ")
                time_list.append(avg)
                x_values.append(i)
                time.sleep(5)       
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
            with open(avg_time_path, "a+") as file:
                file.write(f"\n\n")
            # 读取文件
            with open(query_time_path, 'r') as input_file:
                query_time_content = input_file.read()
            r = process_query_time(query_time_path, repeat)
            # 创建新文件并写入内容
            with open(then_query_time_path, 'a+') as output_file:
                output_file.write(f"avg_query_mysql_time_one_request:\n{r}\n")
                output_file.write(f"{query_time_content}")

            # 清空原始文件内容
            with open(query_time_path, 'w') as input_file:
                input_file.write('')

        print(f"{op_list[op_index]}:\n{time_list}")


def test_faas_vshufu_datarange_0523():
    time_list = []
    repeat = 10
    table_name_ = "osm_a"#"table_1"#"osm_a"
    db_name_list = ["osm_10e3_3_","osm_10e4_3_","osm_10e5_3_","osm_10e6_3_","osm_10e7_3_",]
    op_avg_list = []
    for j in range(5):
        for oo in range(3): 
            if oo==0:
                column_name_ = "COUNT(*)"
                op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "Count_Distance"
            elif oo==1:
                column_name_ = "id"
                op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "id_Distance"
            elif oo==2:
                column_name_ = "id"
                op_ = "KNN(Point(114.0, 22.2), location, 8)"
                op_name = "id_KNN_K8"
            elif oo==3:
                column_name_ = "id"
                op_ = "KN_B_N(Point(114.0, 22.2), location, 8)"
                op_name = "id_KN_B_N_K8"
            time_list = []
            mulu = f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/osm_0523"
            record_path = mulu + op_name + "_repeat_" + str(repeat) + "_record.txt"
            avg_time_path = mulu + op_name + "_repeat_" + str(repeat) + "_avg_time.txt"
            result_path = mulu + op_name + "_repeat_" + str(repeat) + "_result.txt"
            ruc_time_path = mulu + op_name + "_repeat_" + str(repeat) + "_ruc_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch() 
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch() 
            path = pathlib.Path(ruc_time_path)
            if not path.is_file():
                path.touch()
            with open(record_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, RDS mysql range _RCU_1, op:{op_}\n\n")
            with open(avg_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                file.write(f"time:{formatted_time}\niaas_RDS_mysql, RDS mysql range _RCU_1, op:{op_}\n\n") 
            with open(ruc_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S") 
                file.write(f"time:{formatted_time}")               
            table_name_ = db_name_list[j]  #"osm_500w_" + str(random_list[q]) + "_4_"#"conclave_db_big_3_" #"osm_big_"   
            tt = 0
            with open(record_path, "a") as file:
                file.write(f"\ndata:{table_name_}:  ")  
            for jjj in range(repeat):
                with open(result_path, "a+") as file:          
                    file.write(f"\n\ndata:{table_name_}        repeat:{jjj}\n")  
                result_queue = multiprocessing.Queue()   
                t1 = send_request(1, 3456, "total3", table_name_, column_name_, op_, time.time(),result_queue, result_path)
                tt = tt+t1
                with open(record_path, "a") as file:
                    file.write(f"{t1}, ")                   
                time.sleep(0.2)                
            avg = round(tt/repeat,4)
            op_avg_list.append(avg)
            with open(avg_time_path, "a+") as file:
                file.write(f"{avg}, ")
            time_list.append(avg)
            time.sleep(0.2)
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
            with open(avg_time_path, "a+") as file:
                file.write(f"\n\n")            
            print(time_list) 
    print(op_avg_list)  

def test_hufu_old():
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    x_values = []
    repeat = 10
    table_name_ = "osm_a_"
    #op_list = ["std"]
    for _ in range(3):#重复进行三次
        for i in range(3):
            if i==0:
                column_name_ = "COUNT(*)"
                op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "Count_Distance"
            elif i==1:
                column_name_ = "id"
                op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "id_Distance"
            elif i==2:
                column_name_ = "id"
                op_ = "KNN(Point(114.0, 22.2), location, 8)"
                op_name = "id_KNN_K8"
            elif i==3:
                column_name_ = "id"
                op_ = "KN_B_N(Point(114.0, 22.2), location, 8)"
                op_name = "id_KN_B_N_K8"
            time_list = []
            mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/osm_0523/"
            record_path = mulu + op_name + "_record.txt"
            avg_time_path = mulu + op_name + "_avg_time.txt"
            then_query_time_path = mulu + op_name + "_query_time.txt"
            # record_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool6_repeat_3_RCU_0.5_1_10w_record.txt"
            # avg_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool6_repeat_3_RCU_0.5_1_10w_avg_time.txt"
            query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(query_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(then_query_time_path)
            if not path.is_file():
                path.touch()
            with open(record_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, data:100w*4, op:{op_name}\n\n")
                #file.write(f"iaas_RDS_mysql, pool: 6, local mysql, op:{op_name}\n\n")

            with open(avg_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                file.write(f"time:{formatted_time}\n")                  
            with open(avg_time_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 6, RDS mysql fixed _RCU_1_1, data:100w*4, op:{op_name}\n\n")              
                #file.write(f"iaas_RDS_mysql, pool: 6, local mysql, op:{op_name}\n\n")              
            with open(query_time_path, "a+") as f:
                f.write(f"iaas_RDS_mysql, pool: 6, RDS mysql fixed _RCU_1_1, data:100w*4, op:{op_name}\n\n")     
                #f.write(f"iaas_RDS_mysql, pool: 6, local mysql, op:{op_name}\n\n")           
            for i in range(1,30):
                tt = 0
                with open(record_path, "a") as file:
                    file.write(f"\nrequest num:{i}:\n")               
                with open(query_time_path, "a+") as f:
                    f.write(f"query_num: {i}\n")
                for j in range(repeat):
                    t,_ , list_ = star_pool(i, op_, table_name_, column_name_)
                    while len(list_)!= i:
                        t,_ , list_ = star_pool(i, op_, table_name_, column_name_)
                    tt = tt+t
                    time.sleep(5)
                    if "oneRequest" in record_path:
                        with open(record_path, "a") as file:
                            file.write(f"{list_[0]}, ")
                    else:
                        with open(record_path, "a") as file:
                            file.write(f"{list_}\n")
                avg = tt/repeat
                with open(avg_time_path, "a+") as file:
                    file.write(f"{avg}, ")
                time_list.append(avg)
                x_values.append(i)
                time.sleep(5)       
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
            with open(avg_time_path, "a+") as file:
                file.write(f"\n\n")
            # 读取文件
            with open(query_time_path, 'r') as input_file:
                query_time_content = input_file.read()
            r = process_query_time(query_time_path, repeat)
            # 创建新文件并写入内容
            with open(then_query_time_path, 'a+') as output_file:
                output_file.write(f"avg_query_mysql_time_one_request:\n{r}\n")
                output_file.write(f"{query_time_content}")

            # 清空原始文件内容
            with open(query_time_path, 'w') as input_file:
                input_file.write('')

        print(f"{op_}:\n{time_list}")    



def test_hufu_rangK():
    x_values = []
    repeat = 5

    # dt = "osm"
    dt = "lmis"
    ok = 0
    for _ in range(3):#重复进行三次
        ok = 0
        for kNN_k in range(3,21):
            if dt=="osm":
                table_name_ = "osm_10e7_3_"
                op_ = f"KNN(Point(114.0, 22.2), location, {kNN_k})"
            else:
                table_name_ = "lmis_10e7_3_"
                op_ = f"KNN(Point(37.5, 25), location, {kNN_k})"

            column_name_ = "id"
            op_name = "id_KNN_K8"
            time_list = []
            mulu = f"/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_rangK/{dt}/"
            record_path = mulu + op_name + "_record.txt"
            avg_time_path = mulu + op_name + "_avg_time.txt"
            then_query_time_path = mulu + op_name + "_query_time.txt"
            result_path = mulu + op_name + "_client3" + f"_pool4_repeat_{repeat}_result.txt"
            # record_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool6_repeat_3_RCU_0.5_1_10w_record.txt"
            # avg_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool6_repeat_3_RCU_0.5_1_10w_avg_time.txt"
            query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(query_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(then_query_time_path)
            if not path.is_file():
                path.touch()
            if ok == 0:
                with open(record_path, "a+") as file:
                    file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4_4, data:1000w*4, op:{op_name}\n\n")
                    #file.write(f"iaas_RDS_mysql, pool: 6, local mysql, op:{op_name}\n\n")
                with open(avg_time_path, "a+") as file:
                    # 获取当前时间
                    current_time = datetime.now()
                    # 格式化当前时间为字符串
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                    file.write(f"time:{formatted_time}\n")                  
                with open(avg_time_path, "a+") as file:
                    file.write(f"iaas_RDS_mysql, pool: 6, RDS mysql fixed _RCU_4_4, data:1000w*4, op:{op_name}\n\n")              
                    #file.write(f"iaas_RDS_mysql, pool: 6, local mysql, op:{op_name}\n\n")              
                with open(query_time_path, "a+") as f:
                    f.write(f"iaas_RDS_mysql, pool: 6, RDS mysql fixed _RCU_4_4, data:1000w*4, op:{op_name}\n\n")     
                    #f.write(f"iaas_RDS_mysql, pool: 6, local mysql, op:{op_name}\n\n")           
                ok = 1
  
            with open(record_path, "a") as file:
                file.write(f"\nrequest k:{kNN_k}:\n")  
            with open(result_path, "a") as file:
                file.write(f"\nrequest k:{kNN_k}:\n")  
            list_ = []
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()
            for j in range(repeat):  
                ti = time.time()
                q_t = send_request("11", "22", "total3", table_name_, column_name_, op_, ti, result_queue, result_path)
                q_t = round(q_t,4)
                list_.append(q_t)
                time.sleep(0.5)

            with open(record_path, "a+") as file:
                file.write(f"{list_}\n")
            avg = round(sum(list_)/len(list_),4)
            with open(avg_time_path, "a+") as file:
                file.write(f"{avg}, ")
            time.sleep(1)         



@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def test_hufu_data(cfg: DictConfig):
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    repeat = 10
    table_name_list = ["lmis_10e3_4_","lmis_10e4_4_","lmis_10e5_4_","lmis_10e6_4_","lmis_10e7_4_",]
    #op_list = ["std"]
    for _ in range(2):#重复进行三次
        for iiii in range(1,3):
            i = iiii
            if i==0:
                column_name_ = "COUNT(*)"
                op_ = "DWithin(Point(37.5, 25), location, 0.05)"
                # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "Count_Distance"
            elif i==1:
                column_name_ = "id"
                op_ = "DWithin(Point(37.5, 25), location, 0.05)"
                # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "id_Distance"
            elif i==2:
                column_name_ = "id"
                op_ = "KNN(Point(37.5, 25), location, 8)"
                # op_ = "KNN(Point(114.0, 22.2), location, 8)"
                op_name = "id_KNN_K8"
            elif i==3:
                column_name_ = "id"
                op_ = "KN_B_N(Point(114.0, 22.2), location, 8)"
                op_name = "id_KN_B_N_K8"
            time_list = []
            mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/lmis_0722/"
            d = int(cfg.servers.data_server_3.name[-5])
            record_path = mulu + op_name + f"_10e_{d}_pool4_repeat_{repeat}_record.txt"
            avg_time_path = mulu + op_name + f"_10e_{d}_pool4_repeat_{repeat}_avg_time.txt"
            then_query_time_path = mulu + op_name + f"_10e_{d}_pool4_repeat_{repeat}_query_time.txt"
            result_path = mulu + op_name + f"_10e_{d}_pool4_repeat_{repeat}_result.txt"
            # record_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_record.txt"
            # avg_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_avg_time.txt"
            query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(query_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(then_query_time_path)
            if not path.is_file():
                path.touch()
            with open(record_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")
                #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")
            with open(avg_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                file.write(f"time:{formatted_time}\n")                  
            with open(avg_time_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")              
                #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")              
            with open(query_time_path, "a+") as f:
                f.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")     
                #f.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")           
            for table_idx in range(5):
                tt = 0
                table_name_ = table_name_list[table_idx]
                # with open(record_path, "a+") as file:
                #     file.write(f"\nrequest num:{i}:\n")               
                # with open(query_time_path, "a+") as f:
                #     f.write(f"query_num: {i}\n")     
                # with open(result_path, "a+") as file:          
                #     file.write(f"\n\nquery_num:{i}:\n") 
                for _ in range(repeat):
                    one_t = 0
                    mysql_num = 0
                    list_ = []
                    # while len(list_) != i:
                    #     one_t, _ , list_ = star_pool(i, op_, table_name_, column_name_, result_path, 3)
                    #     time.sleep(1)
                    #     mysql_num = mysql_num+i
                    one_t, _ , list_ = star_pool(1, op_, table_name_, column_name_, result_path, 3)
                    tt = tt+one_t
                    time.sleep(1)
                    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
                        f.write(f"mysql_num: {mysql_num}\n") 
                    # a = True
                    with open(record_path, "a+") as file:
                        file.write(f"{list_[0]}, ")
                    # if a:
                    #     with open(record_path, "a+") as file:
                    #         file.write(f"{list_[0]}, ")
                    # else:
                    #     with open(record_path, "a+") as file:
                    #         file.write(f"{list_}\n")
                avg = tt/repeat
                with open(avg_time_path, "a+") as file:
                    file.write(f"{avg}, ")
                time_list.append(avg)
                time.sleep(2)       
                with open(record_path, "a+") as file:
                    file.write(f"\n\n")
                with open(avg_time_path, "a+") as file:
                    file.write(f"\n\n")
            # # 读取文件
            # with open(query_time_path, 'r') as input_file:
            #     query_time_content = input_file.read()
            # r = process_query_time(query_time_path)
            # # 创建新文件并写入内容
            # with open(then_query_time_path, 'a+') as output_file:
            #     output_file.write(f"avg_query_mysql_time_one_request:\n{r}   avg:{round(sum(r)/len(r)/4,4)}\n")
            #     #除以客户端数量
            #     output_file.write(f"{query_time_content}")

            # 清空原始文件内容
            with open(query_time_path, 'w') as input_file:
                input_file.write('')

        print(f"{op_}:\n{time_list}")    

def test_hufu_multi():
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    x_values = []
    repeat = 3
    table_name_ = "lmis_10e7_4_"
    #op_list = ["std"]
    for _ in range(2):#重复进行三次
        for iiii in range(3):
            i = iiii
            if i==0:
                column_name_ = "COUNT(*)"
                op_ = "DWithin(Point(37.5, 25), location, 0.05)"
                # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "Count_Distance"
            elif i==1:
                column_name_ = "id"
                op_ = "DWithin(Point(37.5, 25), location, 0.05)"
                # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                op_name = "id_Distance"
            elif i==2:
                column_name_ = "id"
                op_ = "KNN(Point(37.5, 25), location, 8)"
                # op_ = "KNN(Point(114.0, 22.2), location, 8)"
                op_name = "id_KNN_K8"
            time_list = []
            mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0725_imis_multi/"
            record_path = mulu + op_name + f"_10M_pool4_repeat_{repeat}_record.txt"
            avg_time_path = mulu + op_name + f"_10M_pool4_repeat_{repeat}_avg_time.txt"
            then_query_time_path = mulu + op_name + f"_10M_pool4_repeat_{repeat}_query_time.txt"
            result_path = mulu + op_name + f"_10M_pool4_repeat_{repeat}_result.txt"
            # record_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_record.txt"
            # avg_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_avg_time.txt"
            query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(query_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(then_query_time_path)
            if not path.is_file():
                path.touch()
            with open(record_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4, op:{op_}\n\n")
                #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")
            with open(avg_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                file.write(f"time:{formatted_time}\n")                  
            with open(avg_time_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4, op:{op_}\n\n")              
                #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")              
            with open(query_time_path, "a+") as f:
                f.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4, op:{op_}\n\n")     
                #f.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")           
            
            for i in range(1,21):
                tt = 0
                with open(record_path, "a+") as file:
                    file.write(f"\nrequest num:{i}:\n")               
                with open(query_time_path, "a+") as f:
                    f.write(f"query_num: {i}\n")     
                with open(result_path, "a+") as file:          
                    file.write(f"\n\nquery_num:{i}:\n") 
                avg_list = []
                for j in range(repeat):
                    one_t = 0
                    mysql_num = 0
                    list_ = []
                    while len(list_) != i:
                        one_t, _ , list_ = star_pool(i, op_, table_name_, column_name_, result_path, 4)
                        time.sleep(1)
                        mysql_num = mysql_num + len(list_)
                    avg_list.append(round(one_t,3))
                    time.sleep(1)
                    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
                        f.write(f"mysql_num: {mysql_num}\n") 
                    with open(record_path, "a+") as file:
                        file.write(f"{list_}\n")
                with open(record_path, "a+") as file:
                    file.write(f"{avg_list}\n")
                avg = round(sum(avg_list)/len(avg_list), 3)
                with open(avg_time_path, "a+") as file:
                    file.write(f"{avg}, ")
                time_list.append(avg)
                x_values.append(i)
                time.sleep(2)       
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
            with open(avg_time_path, "a+") as file:
                file.write(f"\n\n")
            # 读取文件
            with open(query_time_path, 'r') as input_file:
                query_time_content = input_file.read()
            r = process_query_time(query_time_path)
            # 创建新文件并写入内容
            with open(then_query_time_path, 'a+') as output_file:
                output_file.write(f"avg_query_mysql_time_one_request:\n{r}   avg:{round(sum(r)/len(r)/4,4)}\n")
                #除以客户端数量
                output_file.write(f"{query_time_content}")

            # 清空原始文件内容
            with open(query_time_path, 'w') as input_file:
                input_file.write('')

        print(f"{op_}:\n{time_list}")    

def test_hufu_com():
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    x_values = []
    repeat = 7
    table_name_ = "osm_a"
    #op_list = ["std"]
    for _ in range(2):#重复进行三次
        r_list = [0.15898545358028826, 0.4212364412809189, 1.1571136806605484, 1.775262771040979, 3.268488559551399, 5.234980140041148, 6.160005447762548, 6.686533548215142, 7.308196270295051]
        for i in range(len(r_list)):
            column_name_ = "id"
            op_ = f"DWithin(Point(114.0, 22.2), location, {r_list[i]})"
            op_name = "id_Distance"
        
            time_list = []
            mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/osm_0320/id_com/"
            record_path = mulu + op_name + f"_com_10w_{i}_pool4_repeat_{repeat}_record.txt"
            avg_time_path = mulu + op_name + f"_com_10w_{i}_pool4_repeat_{repeat}_avg_time.txt"
            then_query_time_path = mulu + op_name + f"_com_10w_{i}_pool4_repeat_{repeat}_query_time.txt"
            result_path = mulu + op_name + f"_com_10w_{i}_pool4_repeat_{repeat}_result.txt"
            # record_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_record.txt"
            # avg_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_avg_time.txt"
            query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
            path = pathlib.Path(record_path)
            if not path.is_file():
                path.touch()        
            path = pathlib.Path(avg_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(query_time_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(result_path)
            if not path.is_file():
                path.touch()
            path = pathlib.Path(then_query_time_path)
            if not path.is_file():
                path.touch()
            with open(record_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")
                #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")
            with open(avg_time_path, "a+") as file:
                # 获取当前时间
                current_time = datetime.now()
                # 格式化当前时间为字符串
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                file.write(f"time:{formatted_time}\n")                  
            with open(avg_time_path, "a+") as file:
                file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")              
                #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")              
            with open(query_time_path, "a+") as f:
                f.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")     
                #f.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")           
            for i in range(1,2):
                tt = 0
                with open(record_path, "a+") as file:
                    file.write(f"\nrequest num:{i}:\n")               
                with open(query_time_path, "a+") as f:
                    f.write(f"query_num: {i}\n")     
                with open(result_path, "a+") as file:          
                    file.write(f"\n\nquery_num:{i}:\n") 
                for j in range(repeat):
                    one_t = 0
                    mysql_num = 0
                    list_ = []
                    while len(list_) != i:
                        one_t, _ , list_ = star_pool(i, op_, table_name_, column_name_, result_path, 4)
                        time.sleep(1)
                        mysql_num = mysql_num+i
                    tt = tt+one_t
                    time.sleep(1)
                    with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt", "a+") as f:
                        f.write(f"mysql_num: {mysql_num}\n") 
                    a = True
                    if a:
                        with open(record_path, "a+") as file:
                            file.write(f"{list_[0]}, ")
                    else:
                        with open(record_path, "a+") as file:
                            file.write(f"{list_}\n")
                avg = tt/repeat
                with open(avg_time_path, "a+") as file:
                    file.write(f"{avg}, ")
                time_list.append(avg)
                x_values.append(i)
                time.sleep(2)       
            with open(record_path, "a+") as file:
                file.write(f"\n\n")
            with open(avg_time_path, "a+") as file:
                file.write(f"\n\n")
            # 读取文件
            with open(query_time_path, 'r') as input_file:
                query_time_content = input_file.read()
            r = process_query_time(query_time_path)
            # 创建新文件并写入内容
            with open(then_query_time_path, 'a+') as output_file:
                output_file.write(f"avg_query_mysql_time_one_request:\n{r}   avg:{round(sum(r)/len(r)/4,4)}\n")
                #除以客户端数量
                output_file.write(f"{query_time_content}")

            # 清空原始文件内容
            with open(query_time_path, 'w') as input_file:
                input_file.write('')

        print(f"{op_}:\n{time_list}")    

def test_hufu_client_range(client_num_attend):
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    x_values = []
    repeat = 5
    table_name_ = "imis_400w_"
    #op_list = ["std"]
    for _ in range(2):
        client_nums_all = 10     
        for ss in range(2, client_nums_all+1): 
            client_nums = client_nums_all + 2 - ss
            for iiii in range(1):
                # i = 2 - iiii
                i = 1
                if i==0:
                    column_name_ = "COUNT(*)"
                    op_ = "DWithin(Point(37.5, 25), location, 0.05)"
                    # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                    op_name = "Count_Distance"
                elif i==1:
                    column_name_ = "id"
                    op_ = "DWithin(Point(37.5, 25), location, 0.081)"
                    # op_ = "DWithin(Point(37.5, 25), location, 0.05)"
                    # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
                    op_name = "id_Distance"
                elif i==2:
                    column_name_ = "id"
                    op_ = "KNN(Point(37.5, 25), location, 8)"
                    # op_ = "KNN(Point(114.0, 22.2), location, 8)"
                    op_name = "id_KNN_K8"
                time_list = []
                mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0726_client_range/"
                record_path = mulu + op_name + "_client" + str(client_num_attend) + "_pool4_repeat_{repeat}_record.txt"
                avg_time_path = mulu + op_name + "_client" + str(client_num_attend) + "_pool4_repeat_{repeat}_avg_time.txt"
                then_query_time_path = mulu + op_name + "_client" + str(client_num_attend) + "_pool4_repeat_{repeat}_query_time.txt"
                result_path = mulu + op_name + "_client" + str(client_num_attend) + "_pool4_repeat_{repeat}_result.txt"
                # record_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_record.txt"
                # avg_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/iaas_RDS/" + op_name + "_pool4_repeat_3_RCU_0.5_1_10w_avg_time.txt"
                query_time_path = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/analyze/query_time.txt"
                path = pathlib.Path(record_path)
                if not path.is_file():
                    path.touch()        
                path = pathlib.Path(avg_time_path)
                if not path.is_file():
                    path.touch()
                path = pathlib.Path(query_time_path)
                if not path.is_file():
                    path.touch()
                path = pathlib.Path(result_path)
                if not path.is_file():
                    path.touch()
                path = pathlib.Path(then_query_time_path)
                if not path.is_file():
                    path.touch()
                with open(record_path, "a+") as file:
                    file.write(f"client_num:{client_nums}    iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_1_1, op:{op_}\n\n")
                    #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")
                if client_nums==10:
                    with open(avg_time_path, "a+") as file:
                        # 获取当前时间
                        current_time = datetime.now()
                        # 格式化当前时间为字符串
                        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")                
                        file.write(f"time:{formatted_time}\n")                  
                    with open(avg_time_path, "a+") as file:
                        file.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4_4,  0.081  , op:{op_}\n\n")              
                        #file.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")              
                    with open(query_time_path, "a+") as f:
                        f.write(f"iaas_RDS_mysql, pool: 4, RDS mysql fixed _RCU_4_4,  0.081 , op:{op_}\n\n")     
                        #f.write(f"iaas_RDS_mysql, pool: 4, local mysql, op:{op_name}\n\n")           

                list_ = []
                manager = multiprocessing.Manager()
                result_queue = manager.Queue()
                for j in range(repeat):  
                    ti = time.time()
                    q_t = send_request("11", "22", "total"+str(client_nums), table_name_, column_name_, op_, ti, result_queue, result_path)
                    q_t = round(q_t,4)
                    list_.append(q_t)
                    time.sleep(0.5)

                with open(record_path, "a+") as file:
                    file.write(f"{list_}\n")
                avg = round(sum(list_)/len(list_),4)
                with open(avg_time_path, "a+") as file:
                    file.write(f"{avg}, ")
                time.sleep(1)       


def test_query_time():
    # t = start(5)
    # time_list.append(t)
    # time.sleep(5)
    repeat = 1
    table_name_list = ["lmis_10e3_4_","lmis_10e4_4_","lmis_10e5_4_","lmis_10e6_4_","lmis_10e7_4_"]
    # table_name_list = ["osm_10e3_3_","osm_10e4_3_","osm_10e5_3_","osm_10e6_3_","osm_10e7_3_"]
    # table_name_list = ["tpch_part_10e3_4","tpch_part_10e4_4","tpch_part_10e5_4","tpch_part_10e6_4","tpch_part_10e7_4"]
    # table_name_list = ["h_taxi_1k", "h_taxi_10k","h_taxi_100k","h_taxi_1m","h_taxi_10m"]
    #op_list = ["std"]
    for ii in range(5):#重复进行三次
            column_name_ = "COUNT(*)"
            op_ = "DWithin(Point(37.5, 25), location, 0.05)"
            op_name = "cnt_Distance"


            # column_name_ = "COUNT(*)"
            # op_ = "DWithin(Point(114.0, 22.2), location, 0.12)"
            # op_name = "cnt_Distance"

            
            # column_name_ = "size"
            # op_ = "count"
            # op_name = "count"
            
            # column_name_ = "passenger_count"
            # op_ = "count"
            # op_name = "count"
            
            mulu = "/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_query_time/"
            result_path = mulu + op_name + f"_10e_pool4_repeat_{repeat}_result.txt"
            ti = time.time()
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()
            q_t = send_request("11", "22", "total3", table_name_list[ii], column_name_, op_, ti, result_queue, result_path)



if __name__ == "__main__":
    #test_hufu_client_range(2)
    #test_hufu()   
    #test_hufu_com()
    #test_iaas_vertical_oneRequest()
    #test_hufu_one() 
    #test_conclave()
    # for i in range(2,11):
    #     test_hufu_client_range(i)


    # test_iaas_vertical_oneRequest()
    # test_iaas_h_oneRequest()
    # test_faas_vshufu_datarange_0523()
    # test_hufu_multi()

    # test_hufu_client_range(10)
    # test_hufu_rangK()

    # 0722补充实验
    # test_hufu_data()

    # test_query_time()


    #test_iaas_h_oneRequest()
    #time.sleep(1)
   # test_iaas_vertical_oneRequest()



    # table_name_list = ["v_tpch_part_10e3_4", "v_tpch_part_10e4_4","v_tpch_part_10e5_4","v_tpch_part_10e6_4","v_tpch_part_10e7_4"]
    # test_iaas_vertical_oneRequest(table_name_list,"tpch_v")
    # table_name_list = ["tpch_part_10e3_4", "tpch_part_10e4_4","tpch_part_10e5_4","tpch_part_10e6_4","tpch_part_10e7_4"]
    # test_iaas_h_oneRequest(table_name_list,"tpch_h")

    
    # table_name_list = ["v_taxi_1k", "v_taxi_10k","v_taxi_100k","v_taxi_1m","v_taxi_10m"]
    # test_iaas_vertical_oneRequest(table_name_list,"taxi_v")
    # table_name_list = ["h_taxi_1k", "h_taxi_10k","h_taxi_100k","h_taxi_1m","h_taxi_10m"]
    # test_iaas_h_oneRequest(table_name_list,"taxi_h")

    test_query_time()