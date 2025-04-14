python launch_aggr_server.py --model='Linear' --n_features=11  --server_address='127.0.0.1:50090' --trainer_address='127.0.0.1:60000' --lr=0.002
python launch_aggr_server.py --model=LR --n_features=30  --server_address='127.0.0.1:50091' --trainer_address='127.0.0.1:60001' --lr=0.002 
python launch_aggr_server.py --model=mlp --n_features=30  --server_address='127.0.0.1:50092' --trainer_address='127.0.0.1:60002' --lr=0.05 
python launch_aggr_server.py --model=cnn --n_features=30  --server_address='127.0.0.1:50093' --trainer_address='127.0.0.1:60003' --lr=0.05 

python launch_linear_trainer_1.py --model=Linear --n_features=11  --server_address='127.0.0.1:50090' --trainer_address='127.0.0.1:60000' --lr=0.002
python launch_linear_trainer_2.py --model=Linear --n_features=11  --server_address='127.0.0.1:50090' --trainer_address='127.0.0.1:60000' --lr=0.002
python launch_linear_trainer_3.py --model=Linear --n_features=11  --server_address='127.0.0.1:50090' --trainer_address='127.0.0.1:60000' --lr=0.002
python launch_logistic_trainer_1.py --model=LR --n_features=30  --server_address='127.0.0.1:50091' --trainer_address='127.0.0.1:60001' --lr=0.002 
python launch_logistic_trainer_2.py --model=LR --n_features=30  --server_address='127.0.0.1:50091' --trainer_address='127.0.0.1:60001' --lr=0.002 
python launch_logistic_trainer_3.py --model=LR --n_features=30  --server_address='127.0.0.1:50091' --trainer_address='127.0.0.1:60001' --lr=0.002 
python launch_mlp_train_1.py --model=mlp --n_features=30  --server_address='127.0.0.1:50092' --trainer_address='127.0.0.1:60002' --lr=0.05 
python launch_mlp_train_2.py --model=mlp --n_features=30  --server_address='127.0.0.1:50092' --trainer_address='127.0.0.1:60002' --lr=0.05 
python launch_mlp_train_3.py --model=mlp --n_features=30  --server_address='127.0.0.1:50092' --trainer_address='127.0.0.1:60002' --lr=0.05 
python launch_cnn_train_1.py --model=cnn --n_features=30  --server_address='127.0.0.1:50093' --trainer_address='127.0.0.1:60003' --lr=0.05 
python launch_cnn_train_2.py --model=cnn --n_features=30  --server_address='127.0.0.1:50093' --trainer_address='127.0.0.1:60003' --lr=0.05 
python launch_cnn_train_3.py --model=cnn --n_features=30  --server_address='127.0.0.1:50093' --trainer_address='127.0.0.1:60003' --lr=0.05 