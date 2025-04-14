import oss2,json,pickle,time,random,numpy as np,tenseal as ts,os
from parsing import *
def handler(event,context):
    time_log={}
    t1=round(time.time(),5)
    creds=context.credentials
    auth=oss2.StsAuth(creds.access_key_id,creds.access_key_secret,creds.security_token)
    oss_raw_data=json.loads(event)
    oss_info_map=oss_raw_data['events'][0]['oss']
    bucket_name=oss_info_map['bucket']['name']
    endpoint='oss-'+oss_raw_data['events'][0]['region']+'-internal.aliyuncs.com'
    bucket=oss2.Bucket(auth,endpoint,bucket_name)
    object_name=oss_info_map['object']['key']
    admin_auth=oss2.Auth('ADD_by_yourself','ADD_by_yourself')
    admin_bucket=oss2.Bucket(admin_auth,endpoint,"fc-computes")
    sleep_time=0.1
    t2=round(time.time(),5)
    pk_bytes=open('/code/beifen/ts_ckks_pk.config',"rb").read()
    t3=round(time.time(),5)
    pk_ctx=ts.context_from(pk_bytes)
    t4=round(time.time(),5)
    time_log["get_oss_inf"]=t2-t1
    time_log["read_pk"]=t3-t2
    time_log["trans_pk"]=t4-t3
    request_dict=pickle.loads(bucket.get_object(object_name).read())
    request_dict["query_name"]="normal"
    key_server_dict={}
    key_server_dict["ip_address"]=request_dict["ip_address"]
    key_server_dict["cid"]=request_dict["cid"]
    key_server_dict["qid"]=request_dict["qid"]
    op=request_dict["op"].upper()
    column_name=request_dict["column_name"]
    enc_vector=request_parsing(request_dict,pk_ctx,admin_bucket)
    if "VERTICAL"in op:
        serialize_msg=pickle.dumps(enc_vector)
        key_server_dict["op"]="return_results_vertical"
    elif column_name=="id"and("DWITHIN"in op or "KNN"in op or "KN_B_N"in op):
        key_server_dict["op"]="return_range_query_results"
        enc_vector_list=enc_vector
        enc_vector_list.append(op)
        enc_vector_list.append(column_name)
        enc_vector_list_d=pickle.dumps(enc_vector_list)
        serialize_msg=enc_vector_list_d
    else:
        serialize_msg=enc_vector.serialize()
        key_server_dict["op"]="return_results"
    key_server_dict["enc_vector"]=serialize_msg
    key_server_dict_d=pickle.dumps(key_server_dict)
    admin_bucket.put_object("query_user_"+str(request_dict["cid"])+"/key_server_dict.txt",key_server_dict_d)