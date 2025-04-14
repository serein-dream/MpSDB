from datetime import datetime
import json
import multiprocessing
import os
import pathlib
import sys
import time as t
import pickle
import oss2
def read_ram_user_info(file_path):
    with open(file_path, 'r') as file:
        ram_user_info = json.load(file)
    return ram_user_info
def prepare_query_params(cid, qid, is_total, db_name, table_name, column_name, op, client_nums):
    cid = int(cid)
    qid = int(qid)
    client_to_parse_list = {
        "cid": cid,
        "qid": qid,
        "column_name": column_name,
        "op": op,
        "db_name": db_name,
        "table_name": table_name,
        "is_total": is_total,
        "ip_address": "",
        "client_index_list": list(range(1 if "conclave" in db_name or "taxi" in db_name else 0, client_nums + 1)),
        "query_name": "normal"
    }
    return client_to_parse_list
def execute_query(client_to_parse_list):
    client_to_parse_list_d = pickle.dumps(client_to_parse_list)
    auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself<ACCESS_KEY_SECRET>')
    admin_bucket = oss2.Bucket(auth, "ADD_by_yourself<ENDPOINT>", "ADD_by_yourself<BUCKET_NAME>") 
    ram_user_info = read_ram_user_info('<RAM_USER_INFO_PATH>')
    user_auth = oss2.Auth(ram_user_info['key-id'], ram_user_info['key-secret'])
    bucket = oss2.Bucket(user_auth, "ADD_by_yourself<ENDPOINT>", "ADD_by_yourself<BUCKET_NAME>") 
    cleanup_bucket(admin_bucket, client_to_parse_list["cid"])
    cleanup_bucket(bucket, client_to_parse_list["cid"])
    bucket.put_object(f"query_user_{client_to_parse_list['cid']}/query_client_to_parse.txt", client_to_parse_list_d)
    ans_dict = wait_for_result(bucket, client_to_parse_list["cid"])
    result = pickle.loads(ans_dict["result"])
    return result
def cleanup_bucket(bucket, cid):
    for obj in oss2.ObjectIterator(bucket, prefix=f'query_user_{cid}/'):
        bucket.delete_object(obj.key)
def wait_for_result(bucket, cid):
    while not bucket.object_exists(f"query_user_{cid}/key_server_to_client.txt"):
        t.sleep(0.001)
    return pickle.loads(bucket.get_object(f"query_user_{cid}/key_server_to_client.txt").read())
def print_result(op, result, column_name):
    print("query result:")
    print("*" * 50)
    if "VERTICAL" in op.upper():
        for key, value in result.items():
            print(f"{key}: {round(value[0], 3)}")
    elif column_name == "id":
        for i in result:
            print(f"{round(i)}")
    elif column_name == "COUNT(*)":
        print(f"The count of records within the specified distance is:  {round(result[0])}")
    else:
        print(f"{op}:{round(result[0], 3)}")
    print("*" * 50)
def start(db_name, table_name, column_name, op, client_num):
    client_to_parse_list = prepare_query_params(1, 3457, "total", db_name, table_name, column_name, op, client_num)
    result = execute_query(client_to_parse_list)
    print_result(op, result, column_name)
def query_osm():
    table_name = "osm_a"
    client_num = 4
    db_name = "osm_10w_3_"
    while True:
        print("="* 50)
        print("\nThe current federal team has Four parties: client1, client2, client3, client4")
        print("Your current point are (114.0, 22.2)")
        print("\nPlease select the type of query operation:")
        print("0: Count the occurrences within a certain distance from your point")
        print("1: Retrieve records based on their distance from your point")
        print("2: Perform k-nearest neighbor (KNN) search for records around your point")
        print("3: Exit")
        i = int(input("\nEnter the operation number (0-3): "))
        if i == 0:
            column_name = "COUNT(*)"
            distance = float(input("Please enter the distance from your point for your operation: "))
            op = f"DWithin(Point(114.0, 22.2), location, {distance})"
        elif i == 1:
            column_name = "id"
            distance = float(input("Please enter the distance from your point for your operation: "))
            op = f"DWithin(Point(114.0, 22.2), location, {distance})"
        elif i == 2:
            column_name = "id"
            k_value = int(input("Please enter the value of k for KNN search: "))
            op = f"KNN(Point(114.0, 22.2), location, {k_value})"
        elif i == 3:
            print("Exiting the current query.")
            return
        else:
            print("Invalid selection. Please try again.")
            exit()
        start(db_name, table_name, column_name, op, client_num)
def query_vertical():
    table_name = "wine_vq_int_"
    client_num = 4
    db_name = "wine_vq_int_"
    column_name = ""
    while True:
        print("="* 50)
        print("\nThe current federal team has Four parties: client1, client2, client3, client4 ")
        print("\nNow, Query them vertically,Please select the type of query operation:")
        print("0: Count")
        print("1: Average")
        print("2: Variance")
        print("3: Standard Deviation")
        print("4: Exit")
        i = int(input("\nEnter the operation number (0-4): "))
        if i == 0:
            op = "count_vertical"
        elif i == 1:
            op = "avg_vertical"
        elif i == 2:
            op = "variance_vertical"
        elif i == 3:
            op = "std_vertical"
        elif i == 4:
            print("Exiting the current query.")
            return
        else:
            print("Invalid selection. Please try again.")
            exit()
        start(db_name, table_name, column_name, op, client_num)
def query_taxi():
    table_name = "table_1"
    client_num = 3
    db_name = "conclave_db_10_3_"
    column_name = "column_1"
    while True:
        print("="* 50)
        print("\nThe current federal team has three parties: client1, client2, client3 ")
        print("\nNow, Query them vertically,Please select the type of query operation:")
        print("0: Count")
        print("1: Average")
        print("2: Variance")
        print("3: Standard Deviation")
        print("4: Exit")
        i = int(input("\nEnter the operation number (0-4): "))
        if i == 0:
            op = "count"
        elif i == 1:
            op = "avg"
        elif i == 2:
            op = "variance"
        elif i == 3:
            op = "std"
        elif i == 4:
            print("Exiting the current query.")
            return
        else:
            print("Invalid selection. Please try again.")
            exit()
        start(db_name, table_name, column_name, op, client_num)
if __name__ == "__main__":
    while True:
        j = int(input("We provide: \nQueries for relational databases horizontally (1)\nQueries for relational databases vertically (2) \nQuery of Spatial Coordinate Database  (3) \nExit  (4)\nPlease select: "))
        if j == 1:
            query_taxi()
        elif j == 2:
            query_vertical()
        elif j == 3:
            query_osm()
        elif j == 4:
            print("Exiting the program.")
            break
        else:
            print("Invalid selection. Please try again.")
            exit()