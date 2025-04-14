import logging
import os
import pymysql
import numpy as np
import pickle
import re
import time as t
import logging
import os
import pymysql
import numpy as np
import re
logger = logging.getLogger()
def conn_faas(db_name, ip):
    try:
        connection = pymysql.connect(
            host=ip,
            port=int(os.environ['MYSQL_PORT']),
            user=os.environ['MYSQL_USER'],
            passwd=os.environ['MYSQL_PASSWORD'],
            db=db_name,
            connect_timeout=5)
    except Exception as e:
        logger.error(e)
        logger.error("ERROR: Unexpected error: Could not connect to MySql instance.")
        raise Exception(str(e))
    return connection
def parse_op(op, column_name):
    column_name = '(' + column_name + ')'
    op_sql = ''
    pattern = r'[+,\-,*,/,(,)]'
    func_list = [i for i in re.split(pattern, op) if i != '']
    op_list = re.findall(pattern, op)
    for i in range(len(func_list)):
        if func_list[i].isdigit():
            if i < len(op_list):
                op_sql += func_list[i] + op_list[i]
            else:
                op_sql += func_list[i]
        else:
            if i < len(op_list):
                op_sql += func_list[i] + column_name + op_list[i]
            else:
                op_sql += func_list[i] + column_name
    op_sql = op_sql.upper()
    return op_sql
def generate_sql(local_table_name, request):
    table_name = request["table_name"]
    column_name = request["column_name"]
    op = request["op"].upper()
    if "DWITHIN" in op or "KNN" in op or "KN_B_N" in op:
        pattern = r'[\,,(,), ]'
        split_list = [i for i in re.split(pattern, op) if i != '']
        distance_str = f"Distance(ST_GeomFromText('{split_list[1]}({split_list[2]} {split_list[3]}')'), {split_list[4]})"
        k_or_r = split_list[5].strip()
        if "DWITHIN" in op:
            sql = f"SELECT {column_name} FROM {local_table_name} WHERE {distance_str} <= {k_or_r};"
        else:
            sql = f"SELECT {column_name}, {distance_str} FROM {local_table_name} WHERE TRUE ORDER BY {distance_str} ASC LIMIT {k_or_r};"
    else:
        op_sql = parse_op(op, column_name)
        sql = f"SELECT {op_sql} FROM {table_name}"
    return sql, "8"
def get_column_name(db_name, table_name, ip):
    db = conn_faas(db_name, ip)
    cursor = db.cursor()
    query = f"SHOW COLUMNS FROM {table_name};"
    cursor.execute(query)
    table_columns = [column[0] for column in cursor.fetchall()]
    db.close()
    return table_columns
def get_query_results(db_name, ip, sql):
    result_list = []
    db = conn_faas(db_name, ip)
    try:
        with db.cursor() as cursor:
            cursor.execute(sql)
            if "COUNT(*)" in sql:
                results = cursor.fetchone()
                result_list.append(results[0])
            elif "<=" in sql:
                results = cursor.fetchall()
                for row in results:
                    result_list.append(row[0])
            elif "id,Distance" in sql or "LIMIT" in sql or "GROUP BY" in sql:
                results = cursor.fetchall()
                for row in results:
                    result_list.append((row[0], float(row[1])))
            else:
                results = cursor.fetchone()
                for i in range(len(results)):
                    result_list.append(float(results[i]))
    finally:
        db.close()
    return result_list
def get_noise_query_results(db_name, cid, qid, sql, bucket):
    key_server_dict = {
        "op": "get_noise",
        "db_name_number": int(db_name[-1]),
        "cid": cid,
        "qid": qid
    }
    key_server_dict_d = pickle.dumps(key_server_dict)
    bucket.put_object(f"query_user_{cid}/key_server_dict.txt", key_server_dict_d)
    while not bucket.object_exists(f"query_user_{cid}/get_noise.txt"):
        t.sleep(0.2)
    ans_dict = pickle.loads(bucket.get_object(f"query_user_{cid}/get_noise.txt").read())
    bucket.delete_object(f"query_user_{cid}/get_noise.txt")
    noise_msg = ans_dict["result"]
    noise = pickle.loads(noise_msg)
    result_list = get_query_results(db_name, sql)
    noise_result = np.add(result_list, noise).tolist()
    return noise_result