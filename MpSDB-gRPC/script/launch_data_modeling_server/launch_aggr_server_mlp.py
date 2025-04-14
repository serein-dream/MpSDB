import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#sys.path.append("/home/maxvyang01/SecureDatabase")
import hydra
from omegaconf import DictConfig
from data_modeling import AggregateServer
import transmission.tenseal.tenseal_aggregate_server_pb2_grpc as tenseal_aggregate_server_pb2_grpc
import grpc
from concurrent import futures
from conf import args_parser
from data_modeling.model import Logistic,Linear,DNN,MLP,CNNMnist,CNNFashion_Mnist,CNNCifar
from data_modeling.src.data_utils import mlp_ini_dataset

def launch_aggregate_server(host, port):
    args = args_parser()
    aggr_server_address = "127.0.0.1:52090"
    args.model = "mlp"
    args.n_features=30
    args.server_address="127.0.0.1:52090"
    args.trainer_address="127.0.0.1:62000"
    args.lr = 0.05
    max_msg_size = 1000000000
    pk_ctx_file = "../../transmission/ts_ckks_pk.config"
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    print(f"\nmodel:  {args.model}\n")
    if args.model == "cnn":
        if args.dataset == 'mnist':
            model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            model = CNNCifar(args=args)
    elif args.model == "LR":
        model = Logistic(n_f=args.n_features)
    elif args.model == "Linear":
        model = Linear(n_f=args.n_features)
    elif args.model == "mlp":
        td = mlp_ini_dataset(args)
        img_size = td.shape
        len_in = 1
        for x in img_size:
            len_in *= x
            model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    tenseal_aggregate_server_pb2_grpc.add_AggregateServerServiceServicer_to_server(
        AggregateServer(3, pk_ctx_file, model),
        server)
    server.add_insecure_port(aggr_server_address)
    server.start()
    print("Aggregate Server start")
    server.wait_for_termination()


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(cfg: DictConfig):
    paths = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(paths)
    host = cfg.servers.aggregate_server.host
    port = int(cfg.servers.aggregate_server.port)
    launch_aggregate_server(host, port)


if __name__ == '__main__':
    main()
