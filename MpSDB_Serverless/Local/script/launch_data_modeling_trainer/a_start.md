


taskset -c 4,5,6,7 python launch_linear_trainer_1.py
taskset -c 8,9,10,11 python launch_linear_trainer_2.py
taskset -c 0,1,2,3 python launch_linear_trainer_3.py


taskset -c 4,5,6,7 python launch_logistic_trainer_1.py
taskset -c 8,9,10,11 python launch_logistic_trainer_2.py
taskset -c 0,1,2,3 python launch_logistic_trainer_3.py



taskset -c 4,5,6,7 python launch_mlp_train_1.py
taskset -c 8,9,10,11 python launch_mlp_train_2.py
taskset -c 0,1,2,3 python launch_mlp_train_3.py


taskset -c 4,5,6,7 python launch_cnn_train_1.py
taskset -c 8,9,10,11 python launch_cnn_train_2.py
taskset -c 0,1,2,3 python launch_cnn_train_3.py

taskset -c 4,5,6,7 python launch_K_means_train_1.py
taskset -c 8,9,10,11 python launch_K_means_train_2.py
taskset -c 0,1,2,3 python launch_K_means_train_3.py