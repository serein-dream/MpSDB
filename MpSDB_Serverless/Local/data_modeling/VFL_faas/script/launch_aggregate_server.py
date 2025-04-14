import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.utils import seed_torch
seed_torch()
import transmission.pickle.aggregate_server_pb2_grpc as aggregate_server_pb2_grpc
import grpc
from concurrent import futures
#from model.DragonNet import MiddleModel
#from model.PropensityNet import MiddleModel
from conf import args_parser
from aggregate_server import AggregateServer
from model.utils import get_device


def launch_aggregate_server():
    device = get_device()
    args = args_parser()
    address = args.server_address
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    if args.model == "Linear":
        from model.Linear import MiddleModel
        model = MiddleModel()
    if args.model == "LR":
        from model.Logistic import MiddleModel
        model = MiddleModel()
    if args.model.upper() == "MLP":
        from model.MLP import MiddleModel
        model = MiddleModel(28*28)
    elif args.model.upper() == "CNN":
        from model.CNN import MiddleModel
        model = MiddleModel(28*28)
    model.to(device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    print("num of clients:", args.num_clients)
    aggregate_server_pb2_grpc.add_AggregateServerServiceServicer_to_server(
        AggregateServer(args.num_clients, model, args.label_owner_address),
        server)
    server.add_insecure_port(address)
    server.start()
    print("Aggregate Server start")
    server.wait_for_termination()


if __name__ == '__main__':
    launch_aggregate_server()
