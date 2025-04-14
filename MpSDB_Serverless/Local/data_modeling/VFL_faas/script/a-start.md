
taskset -c 0,1,2,3 python launch_Linear_label_owner.py
taskset -c 4,5,6,7 python launch_Linear_client_1.py
taskset -c 8,9,10,11 python launch_Linear_client_2.py
taskset -c 0,1,2,3 python launch_Linear_client_3.py

taskset -c 0,1,2,3 python launch_Logistic_label_owner.py
taskset -c 4,5,6,7 python launch_Logistic_client_1.py
taskset -c 8,9,10,11 python launch_Logistic_client_2.py
taskset -c 0,1,2,3 python launch_Logistic_client_3.py


taskset -c 0,1,2,3 python launch_mlp_label_owner.py
taskset -c 4,5,6,7 python launch_mlp_client_1.py
taskset -c 8,9,10,11 python launch_mlp_client_2.py
taskset -c 0,1,2,3 python launch_mlp_client_3.py



taskset -c 0,1,2,3 python launch_cnn_label_owner.py
taskset -c 4,5,6,7 python launch_cnn_client_1.py
taskset -c 8,9,10,11 python launch_cnn_client_2.py
taskset -c 0,1,2,3 python launch_cnn_client_3.py

taskset -c 4,5,6,7 python launch_kmeans_client_1.py
taskset -c 8,9,10,11 python launch_kmeans_client_2.py
taskset -c 0,1,2,3 python launch_kmeans_client_3.py