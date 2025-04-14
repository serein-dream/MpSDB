taskset -c 4,5,6,7 python launch_cnn_client_1.py & echo $! > pid/record_1.txt

taskset -c 8,9,10,11 python launch_cnn_client_2.py & echo $! > pid/record_2.txt

taskset -c 12,13,14,15 python launch_cnn_client_3.py & echo $! > pid/record_3.txt

# taskset -c 0,1,2,3 python launch_cnn_label_owner.py & echo $! > pid/launch_cnn_label_owner.txt 

