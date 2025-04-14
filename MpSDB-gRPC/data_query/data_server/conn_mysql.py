
from datetime import datetime
import logging
import pymysql
import numpy as np
import transmission.tenseal.tenseal_key_server_pb2 as tenseal_key_server_pb2
import pickle
import re
from omegaconf import DictConfig
import time

"""
conn_mysql.py provides query operations over database:
1. get max value
2. get min value
3. get sum value
"""
logger = logging.getLogger()
def conn_faas(db_name, de_ip):
    print(db_name)
    conn_ok = True
    try:
        # connection = pymysql.connect(
        #     # Replace it with your host name.
        #     host=os.environ['MYSQL_ENDPOINT'],
        #     # Replace it with your port number.
        #     port=int(os.environ['MYSQL_PORT']),
        #     # Replace it with your username.
        #     user="root",
        #     # Replace the  password with the one corresponding to your username.
        #     passwd="Xvyang0615",
        #     # Replace it with the name of your database.
        #     db=db_name,
        #     connect_timeout=5)
        connection = pymysql.connect(
            # Replace it with your host name.
            host=de_ip,
            # Replace it with your port number.
            port=3306,
            # Replace it with your username.
            user='root',
            # Replace the  password with the one corresponding to your username.
            passwd='Xvyang0615',
            # Replace it with the name of your database.
            db=db_name,
            connect_timeout=5)
    except Exception as e:
        conn_ok = False
        connection = 0
        logger.error(e)
        logger.error(
            "ERROR: Unexpected error: Could not connect to MySql instance.")
        #raise Exception(str(e))
    return connection, conn_ok

def parse_op(op, column_name):
    column_name = '(' + column_name + ')'
    op_sql = ''
    pattern = r'[+,\-,*,/,(,)]'
    func_list = [i for i in re.split(pattern, op) if i != '']
    op_list = re.findall(pattern, op)
    if '(' or ')' in op_list:
        j = 0
        for i in range(len(func_list)):
            if func_list[i].isdigit():
                if j < len(op_list):
                    op_sql = op_sql + func_list[i] + op_list[j]
                else:
                    op_sql = op_sql + func_list[i]
            else:
                if j < len(op_list):
                    op_sql = op_sql + func_list[i] + column_name + op_list[j]
                else:
                    op_sql = op_sql + func_list[i] + column_name

            if j + 1 < len(op_list):
                if op_list[j] == ')':
                    op_sql = op_sql + op_list[j + 1]
                    j += 2
                    if op_list[j] == '(':
                        op_sql = op_sql + op_list[j]
                        j += 1
                elif op_list[j + 1] == '(':
                    op_sql = op_sql + op_list[j + 1]
                    j += 2
                else:
                    j += 1
            else:
                j += 1
    else:
        for i in range(len(func_list)):
            if func_list[i].isdigit():
                op_sql = op_sql + func_list[i] + op_list[i]
            else:
                if i < len(op_list):
                    op_sql = op_sql + func_list[i] + column_name + op_list[i]
                else:
                    op_sql = op_sql + func_list[i] + column_name
    op_sql = op_sql.upper()
    return op_sql


def get_column_name(db_name, table_name, ip):
    db, coon_ok = conn_faas(db_name, ip)
    while coon_ok == False:
        db, coon_ok = conn_faas(db_name, ip)
    try:
        with db.cursor() as cursor:
            query = f"SHOW COLUMNS FROM {table_name};"
            cursor.execute(query)
            table_columns = [column[0] for column in cursor.fetchall()]
    finally:
        db.close()
    return table_columns

# generate sql from request
def generate_sql(local_table_name, request):
    table_name = request.table_name
    column_name =  request.column_name
    op = request.op.upper()
    if "DWITHIN" in op or "KNN" in op or "KN_B_N" in op:
            pattern = r'[\,,(,), ]'
            split_list = [i for i in re.split(pattern, op) if i != '']
            distance_str = "Distance(ST_GeomFromText('" +split_list[1] + "(" + split_list[2] + " " + split_list[3] + ")'), " + split_list[4] + ")"
            k_or_r = split_list[5].strip()
            if "DWITHIN" in op:         
                sql = "SELECT " + column_name + " from " + local_table_name + " WHERE " + distance_str + " <= " + k_or_r + ";"
            else:
                sql = "SELECT " + column_name + "," + distance_str + " from " + local_table_name + " WHERE TRUE ORDER BY  " + distance_str + " ASC  LIMIT " + k_or_r + ";"
    else:
        k_or_r = "8"    
        table_name = request.table_name.upper()
        column_name = request.column_name.upper()
        op_sql = parse_op(op, column_name)
        sql = "SELECT " + op_sql + " FROM {0}".format(table_name)
    return sql, k_or_r

def get_r_by_k(db_name, sql, db_ip):
    result_list = []
    #db = conn(db_name, cfg)
    db, coon_ok = conn_faas(db_name, db_ip)
    while coon_ok == False:
        db, coon_ok = conn_faas(db_name, db_ip)
    try:
        with db.cursor() as cursor:
            cursor.execute(sql)    
            results = cursor.fetchall()
            #print(results)
            for row in results:
                result_list.append((row[0],float(row[1])))
    finally:
        db.close()
    result_np = np.array(result_list)
    sorted_indices = np.argsort(result_np[:, 1])
    result_np = result_np[sorted_indices]
    k = int(sql[-2])-1
    r = result_np[k][1]
    return r



def get_query_results(db_name, db_ip, sql):
    result_list = []
    #print(f"\n\ndb_name:  {db_name}\ncfg:  {cfg}\n\n")
    #db = conn(db_name, cfg)
    db, coon_ok = conn_faas(db_name, db_ip)
    while coon_ok == False:
        db, coon_ok = conn_faas(db_name, db_ip)
    try:
        with db.cursor() as cursor:
            st = time.time()
            cursor = db.cursor()
            #print(f"\n\ndb_name:{db_name}  \nsql:  {sql}\n\n")
            #sql = "SELECT MIN(COLUMN_1) FROM table_1;"
            cursor.execute(sql)
            et = time.time()      

            
            current_time = datetime.now()
            # 格式化当前时间为字符串
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")  
            with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_query_time/all_query_time.txt", "a") as file:
                file.write(f"time:{formatted_time}\n")  
            with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_query_time/all_query_time.txt", "a") as file:
                file.write(f"\nsql:{sql}\nquery time:{round(et-st, 4)}\n")  
            with open("/home/maxvyang01/FaaS_FL2023/SecureDatabase/script/launch_data_query_client/0820_query_time/ok_query_time.txt", "a") as file:
                file.write(f"{round(et-st, 4)}, ")  


            if "COUNT(*)" in sql:
                results = cursor.fetchone()
                result_list.append((results[0]))
                # # close conn
                # db.close()
                # #print(f"results:\n{results}")
                # return results
            
            elif "<=" in sql:
                results = cursor.fetchall()
                #print(results)
                for row in results:
                    result_list.append((row[0]))
                # db.close()
                # return result_list
            
            elif "id,Distance" in sql:
                results = cursor.fetchall()
                #print(results)
                for row in results:
                    result_list.append((row[0],float(row[1])))
                # db.close()
                # return result_list
                
            elif "GROUP BY" in sql:
                results = cursor.fetchall()
                for row in results:
                    result_list.append((row[0],float(row[1])))
                # db.close()
                # return result_list
            elif "LIMIT" in sql:
                results = cursor.fetchall()
                print("In LIMIT", results)
                # db.close()
                for i in range(len(results)):
                    result_list.append(float(results[i][0]))
                # return result_list
            else:
                results = cursor.fetchone()
                for i in range(len(results)):
                    result_list.append(float(results[i]))
    # close conn
    finally:
        db.close()


    return result_list
'''
# get query result from database_dbname
def get_query_results(db_name, cfg, sql):
    result_list = []
    db = conn(db_name, cfg)
    cursor = db.cursor()
    cursor.execute(sql)
    results = cursor.fetchone()
    # close conn
    db.close()
    for i in range(len(results)):
        result_list.append(float(results[i]))

    return result_list
'''

# get the query results with laplace noise
def get_noise_query_results(db_name, cfg, cid, qid, sql, key_stub):
    # sensitivity = 1
    # epsilon = 5
    # noise = np.random.laplace(loc=0, scale=sensitivity / epsilon)
    noise_request = tenseal_key_server_pb2.get_noise_request(db_name=db_name, cid=cid, qid=qid)
    response = key_stub.get_noise(noise_request)
    noise_msg = response.noise_msg
    noise = pickle.loads(noise_msg)

    result_list = get_query_results(db_name, sql)
    noise_result = np.add(result_list, noise).tolist()

    return noise_result