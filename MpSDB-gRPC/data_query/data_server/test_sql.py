import pymysql
import numpy as np
import pickle
import re
from omegaconf import DictConfig
import hydra
"""
conn_mysql.py provides query operations over database:
1. get max value
2. get min value
3. get sum value
"""


# establish connection with mysql
def conn(name, cfg: DictConfig):
    db = pymysql.connect(host=cfg.database.host,
                         port=int(cfg.database.port),
                         user=cfg.database.user,
                         password=cfg.database.password,
                         database='osm_db')
    return db


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


# generate sql from request
def generate_sql(request):
    table_name = request.table_name
    column_name = request.column_name
    op = request.op.upper()
    op_sql = parse_op(op, column_name)
    sql = "SELECT " + op_sql + " FROM {0}".format(table_name)
    return sql


# get query result from database_dbname
def get_query_results(db_name, cfg, sql):
    result_list = []
    db = conn(db_name, cfg)
    cursor = db.cursor()
    cursor.execute(sql)
    if "GROUP BY" in sql:
        results = cursor.fetchall()
        for row in results:
            result_list.append((row[0],float(row[1])))
        db.close()
        return result_list
    if "LIMIT" in sql:
        results = cursor.fetchall()
        print("In LIMIT", results)
        db.close()
        for i in range(len(results)):
            result_list.append(float(results[i][0]))
        return result_list

    results = cursor.fetchone()
    # close conn
    db.close()
    for i in range(len(results)):
        result_list.append(float(results[i]))

    return result_list

'''
SELECT id FROM osm_a WHERE   KNN(Point(121.5, 14.5), location, 8)
SELECT id,Distance(location, ST_GeomFromText('POINT(60.0 7.0)')) from osm_a_2 WHERE TRUE ORDER BY  ST_Distance(location, ST_GeomFromText('POINT(60.0 7.0)')) ASC  LIMIT 8

SELECT COUNT(*) cnt FROM osm_a WHERE DWithin(Point(121.5, 14.5), location, 0.5);
SELECT COUNT(*) cnt from osm_a_2 WHERE ST_Distance(location, ST_GeomFromText('POINT(60.0 7.0)'))
'''
@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(cfg: DictConfig):
    column_name = "KNN(Point(121.5, 14.5), location, 8)"
    op = "KNN"
    if op in ["RANGE QUERY", "RANGE COUNT", "KNN"]:
        sql = ".00."
    else:
        sql = ".00."
    db_name = "osm_a_1"
    db = conn(db_name, cfg)  
    sql = "SELECT id from osm_a_1 WHERE Distance(ST_GeomFromText('POINT(121.500000 14.500000)'), location) <= 0.5;"    

    cl = "KNN(Point(121.5, 14.5), location, 8)"
    pattern = r'[()]'
    sql = [i for i in re.split(pattern, cl) if i != '']

    # cursor = db.cursor()
    # cursor.execute(sql)
    # results = cursor.fetchall()
    # db.close
    print(f"sql:\n{sql}\nresult:\n")

if __name__ == "__main__":
    main()