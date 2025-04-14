from conn_mysql import generate_sql, get_noise_query_results, get_query_results, get_r_by_k, get_column_name
import tenseal as ts
import pickle
import random
import copy
from typing import Dict, List
class DatabaseServer:
    def __init__(self, db_name: str, db_id: int):
        self.n_th_cache = {}
        self.hash_cache = {}
        with open("/path/to/ts_ckks_pk.config", "rb") as f:
            pk_bytes = f.read()
        self.pk_ctx = ts.context_from(pk_bytes)
        self.ip = ["IP_LIST"][db_id]
        self.sleep_time = 0.01 if "taxi_" in db_name or "tpch" in db_name else 0.01
        self.db_number = db_id if "taxi_" in db_name or "tpch" in db_name else int(db_name[-1])
        self.database_name = db_name
        self.table_name = db_name
    def _generate_encrypted_vector(self, query_result: List[float]) -> bytes:
        plain_vector = ts.plain_tensor(query_result)
        enc_vector = ts.ckks_vector(self.pk_ctx, plain_vector)
        return enc_vector.serialize()
    def query_DWithin(self, request: Dict) -> bytes:
        sql, k_or_r = generate_sql(self.table_name, request)
        query_result = get_query_results(self.database_name, self.ip, sql)
        if request["column_name"] == "id":
            query_result = [float(x) for x in query_result] + [float(k_or_r)]
        return self._generate_encrypted_vector(query_result)
    def query_r_by_k(self, request: Dict) -> float:
        greater = request["greater"]
        sql, _ = generate_sql(self.table_name, request)
        r = get_r_by_k(self.database_name, sql)
        p = random.uniform(1.01, 1.2) if greater else random.uniform(0.8, 0.99)
        return r * p
    def query_operation(self, request: Dict) -> bytes:
        sql, _ = generate_sql(self.table_name, request)
        query_result = get_query_results(self.database_name, self.ip, sql)
        return self._generate_encrypted_vector(query_result)
    def query_operation_vertical(self, request: Dict) -> Dict[str, bytes]:
        table_column_name_list = get_column_name(self.database_name, self.table_name, self.ip)
        query_vertical = {}
        for table_column_name in table_column_name_list:
            if table_column_name == "id":
                continue
            request_ = copy.deepcopy(request)
            request_["column_name"] = table_column_name
            request_["op"] = request["op"].replace("_vertical", "")
            need_sqrt = "STD" in request["op"].upper()
            request_["op"] = "variance" if need_sqrt else request_["op"]
            sql, _ = generate_sql(self.table_name, request_)
            query_result = get_query_results(self.database_name, self.ip, sql)
            query_vertical[table_column_name] = self._generate_encrypted_vector(query_result)
        return query_vertical
    def noise_query_operation(self, request: Dict, bucket: int) -> List[float]:
        sql, _ = generate_sql(request)
        cid = request["cid"]
        qid = request["qid"]
        return get_noise_query_results(self.database_name, cid, qid, sql, bucket)
    def query_median_posi(self, request: Dict) -> Dict[str, bytes]:
        cid = request["cid"]
        qid = request["qid"]
        table_name = request["table_name"]
        column_name = request["column_name"]
        median = request["median"]
        avg = request["avg"]
        std = request["std"]
        sigma3_left = avg - 3 * std
        sigma3_right = avg + 3 * std
        le_sql = f"SELECT COUNT(*) FROM {self.database_name}.{table_name} WHERE {column_name} <= {median}"
        le_result = get_query_results(self.database_name, le_sql)
        g_sql = f"SELECT COUNT(*) FROM {self.database_name}.{table_name} WHERE {column_name} > {median}"
        g_result = get_query_results(self.database_name, g_sql)
        le_enc_vector = self._generate_encrypted_vector(le_result)
        g_enc_vector = self._generate_encrypted_vector(g_result)
        return {"less_e": le_enc_vector, "greater": g_enc_vector}
    def get_nearest(self, request: Dict) -> List[bytes]:
        table_name = request["table_name"]
        column_name = request["column_name"]
        value = request["value"]
        sql = f"SELECT {column_name} FROM {self.database_name}.{table_name} ORDER BY ABS({column_name} - {value}) LIMIT 3;"
        query_result = get_query_results(self.database_name, self.ip, sql)
        return [self._generate_encrypted_vector([result]) for result in query_result[:3]]
    def n_th_query_operation(self, request: Dict) -> Dict[str, bytes]:
        cid = request["cid"]
        qid = request["qid"]
        n = request["n"]
        mode = request["mode"]
        table_name = request["table_name"]
        column_name = request["column_name"]
        if mode == "clean":
            self.n_th_cache.clear()
            self.hash_cache.clear()
            sql = f"SELECT {column_name}, COUNT(*) AS i FROM {self.database_name}.{table_name} GROUP BY {column_name} ORDER BY i"
            query_result = get_query_results(self.database_name, self.ip, sql)
            for i, result in enumerate(query_result):
                self.n_th_cache[i] = result
                self.hash_cache[hash(result[0] + 0.01)] = result
        available = n in self.n_th_cache
        query_result = [self.n_th_cache[n][1]] if available else [0]
        hash_value = hash(self.n_th_cache[n][0] + 0.01) if available else 0
        return {"hash_value": hash_value, "result": self._generate_encrypted_vector(query_result), "available": available}
    def query_mode_using_hash(self, hash_: bytes) -> Dict[str, bytes]:
        hash_list = pickle.loads(hash_)
        available_list = [False] * len(hash_list)
        query_result = [0] * len(hash_list)
        for i, h in enumerate(hash_list):
            if h in self.hash_cache:
                available_list[i] = True
                query_result[i] = self.hash_cache[h][0]
        result_list = [self._generate_encrypted_vector([qr]) for qr in query_result]
        return {"mode": pickle.dumps(result_list), "available": pickle.dumps(available_list)}
    def query_from_buffer(self, hash_: bytes) -> Dict[str, bytes]:
        available = hash_ in self.hash_cache
        query_result = [self.hash_cache[hash_][1]] if available else [0]
        return {"result": self._generate_encrypted_vector(query_result), "available": available}