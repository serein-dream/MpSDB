from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import ast
import oss2
from a_deom.learning_h.conf import args_parser
from a_deom.train_hv import train_h, train_v
from a_deom.learning_h.script.launch_data_modeling_trainer.start_server import upload_init_params
from a_deom.query_hv import upload_db_info, query_relational, query_spatial
app = Flask(__name__)
CORS(app, origins="*")
file_path = 'result.txt'
result_num = 0
def parse_request_args(request):
    args = args_parser()
    args.batch_size = int(request.get_json().get('batch_size'))
    args.n_features = 30
    args.epoch = int(request.get_json().get('local_epoch'))
    args.rounds = int(request.get_json().get('rounds'))
    args.num_classes = int(request.get_json().get('num_classes'))
    args.lr = float(request.get_json().get('learning_rate'))
    args.num_channels = 1
    return args
def determine_data_type(data_type_str):
    if data_type_str[4:] == "vertically":
        return "v"
    elif data_type_str[4:] == "horizontally":
        return "h"
    return None
def set_model_rounds(model, args):
    if "Linear".upper() in model.upper() or "kmean".upper() in model.upper():
        args.rnd = round(5120 / args.batch_size)
    elif "LR" in model.upper():
        args.rnd = int(480 / args.batch_size)
    elif args.model.upper() == "MLP":
        args.rnd = int(480 / args.batch_size)
    elif args.model.upper() == "CNN":
        args.rnd = int(60000 / args.batch_size)
def log_training_info(args, model, data_type):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a+") as file:
        file.write(f"\ntrain:{model},  {data_type},  {current_time}\n")
@app.route('/api/train', methods=['POST'])
def handle_request_train():
    global result_num
    with open(file_path, 'r', encoding='utf-8') as file:
        result_num = len(file.readlines())
    model = request.get_json().get('train_model')
    if model is None:
        return jsonify({'message': 'Missing or incorrectly formatted parameters!'})
    model_mapping = {"HFL_linear": "Linear", "HFL_LR": "LR", "HFL_MLP": "MLP", "HFL_CNN": "cnn"}
    args = parse_request_args(request)
    args.model = model_mapping.get(model, "kmeans")
    data_type = determine_data_type(request.get_json().get('data_distribution'))
    if data_type is None:
        return jsonify({'message': 'Missing or incorrectly formatted parameters!'})
    client_name_list = request.get_json().get('client_name_list')
    if not client_name_list:
        return jsonify({'message': 'Missing or incorrectly formatted parameters!'})
    args.num_users = len(request.get_json().get('user_list'))
    set_model_rounds(args.model, args)
    log_training_info(args, args.model, data_type)
    auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself<ACCESS_KEY_SECRET>')
    bucket = oss2.Bucket(auth, "ADD_by_yourself<ENDPOINT>", "ADD_by_yourself<BUCKET_NAME>")
    if data_type == "h":
        upload_init_params(args, bucket)
        train_h(args, bucket)
    else:
        train_v(args, bucket)
    return jsonify({'message': 'Request received successfully'})
@app.route('/api/train/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file_dir = os.path.join(current_dir, 'train_file')
    if not os.path.exists(train_file_dir):
        os.makedirs(train_file_dir)
    upload_file_path = os.path.join(train_file_dir, file.filename)
    file.save(upload_file_path)
    return 'File uploaded successfully'
@app.route('/api/train/getResult', methods=['GET'])
def train_get_result():
    global result_num
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    response = {'output': [], 'train_type': None, 'train_distribution': None}
    for line in lines[result_num:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("train"):
            train_info = line.split(",")
            response['train_type'] = train_info[0].split(":")[1].strip()
            response['train_distribution'] = train_info[1].strip()
            continue
        line_items = line.split(",")
        if response['train_distribution'] == "v":
            tmp_item = {'epoch': line_items[0].split(":")[1], 'loss': line_items[1].split(":")[1].strip()}
        else:
            tmp_item = {'rank': line_items[0].split(":")[1].strip(), 'rnd': line_items[1].split(":")[1].strip(), 'loss': line_items[2].split(":")[1].strip()}
        response['output'].append(tmp_item)
    result_num = len(lines)
    return jsonify(response)
def parse_query_args(request):
    data_type = request.get_json().get('data_distribution')
    if data_type is None:
        return None, None, None, None, None, None, None, None, None, None, None, None
    data_type = "v" if data_type == "vertically" else "h"
    query_data_type = request.get_json().get('query_type')
    query_type = request.get_json().get('operation')
    query_x = request.get_json().get('point_x', "114.0")
    query_y = request.get_json().get('point_y', "22.2")
    column_name = request.get_json().get('column_name')
    client_name_list = request.get_json().get('client_name_list')
    database_name_list = request.get_json().get('database_name_list')
    table_name_list = request.get_json().get('table_name_list')
    user_list = request.get_json().get('user_list')
    connect_ip_list = request.get_json().get('connect_ip_list')
    password_list = request.get_json().get('password_list')
    distance = request.get_json().get('distance')
    k = request.get_json().get('k')
    return data_type, query_data_type, query_type, query_x, query_y, column_name, client_name_list, database_name_list, table_name_list, user_list, connect_ip_list, password_list, distance, k
def map_query_type(query_data_type, query_type):
    if query_data_type == "relational":
        mapping = {"count": 0, "avg": 1, "variance": 2, "std": 3, "sum": 4, "max": 5, "min": 6}
    else:
        mapping = {"cnt_dis": 0, "id_dis": 1, "id_knn": 2}
    return mapping.get(query_type, None)
@app.route('/api/query', methods=['POST'])
def handle_request():
    global result_num
    with open(file_path, 'r', encoding='utf-8') as file:
        result_num = len(file.readlines())
    data_type, query_data_type, query_type, query_x, query_y, column_name, client_name_list, database_name_list, table_name_list, user_list, connect_ip_list, password_list, distance, k = parse_query_args(request)
    if any(arg is None for arg in [data_type, query_data_type, query_type, column_name, client_name_list]):
        return jsonify({'message': 'Missing or incorrectly formatted parameters!'})
    query_op = {
        "data_type": data_type,
        "query_data_type": query_data_type,
        "column_name": column_name[0] if data_type == "h" else column_name,
        "client_num": len(user_list),
        "query_type": map_query_type(query_data_type, query_type)
    }
    upload_db_info(query_op["data_type"])
    if query_op["query_data_type"] == "relational":
        query_relational(query_op)
    else:
        query_op["distance"] = distance
        query_op["column_name"] = "location"
        query_spatial(query_op, query_x, query_y)
    return jsonify({'message': 'Request received successfully'})
@app.route('/api/query/getResult', methods=['GET'])
def query_get_result():
    global result_num
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    result, trip_distance, fare_amount, total_amount = [], '', '', ''
    for line in lines[result_num:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("("):
            data_dict = ast.literal_eval(line[1:-4])
            trip_distance = data_dict['trip_distance'][0]
            fare_amount = data_dict['fare_amount'][0]
            total_amount = data_dict['total_amount'][0]
            continue
        line_items = line.split(" ")
        result.extend(line_items)
    result_num = len(lines)
    return jsonify({
        "result": result,
        "trip_distance": trip_distance,
        "fare_amount": fare_amount,
        "total_amount": total_amount
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)