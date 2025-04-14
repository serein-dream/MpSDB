import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from conf import args_parser
from data_modeling.trainer.linear_trainer import LinearTrainer
from data_modeling.data_loader import MysqlDataSet
import time
#echo 'export PATH=$PATH:/home/maxvyang01/mysql/usr/bin/:/home/maxvyang01/mysql/usr/sbin/' >> /home/maxvyang01/.bashrc
#source /home/maxvyang01/.bashrc
def run(arg,bt): 
    args.n_features = 11  
    before_train_start_time = time.perf_counter()
    loss_time = []
    ini_time_s = time.perf_counter()
    cols_list = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                 'quality']
    mysql_dataset_1 = MysqlDataSet(f"database_{args.rank+1}", "wine_quality", cols_list)
    mysql_dataset_eval = MysqlDataSet("test_data", "wine_quality", cols_list)
    load_data_time_e = time.perf_counter()
    load_data_time = load_data_time_e - ini_time_s
    '''
    cols_list = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','y']
    mysql_dataset_1 = MysqlDataSet("database_1", "cancer", cols_list)
    mysql_dataset_eval = MysqlDataSet("database_1", "cancer_eval", cols_list)
    '''
    lr_trainer = LinearTrainer(arg, mysql_dataset_1, mysql_dataset_eval, args.rank)

    for rnd in range(args.rounds):
        print("round: ", rnd)
        update_flag = lr_trainer.is_update()
        print(update_flag)
        if update_flag:
            lr_trainer.one_local_round()
            #lr_trainer.test()   
            l_time =  time.perf_counter() - ini_time_s
        else:
            lr_trainer.set_loss_a()
            print("not participate in this round")
            l_time = 0
        # print(rnd, logistic_trainer.model.linear.bias, '\n')  
        loss_time.append(l_time)
    # lr_trainer.write(bt,load_data_time)
    # pa  = "/home/maxvyang01/SecureDatabase/script/l_result/temp/linear"
    # with open(f"{pa}/loss_time1.txt", 'w') as train_los_0:
    #     train_los_0.write(str(loss_time))

if __name__ == '__main__':
    b_t = time.perf_counter()
    st = time.time()
    args = args_parser()
    args.rank = 1
    args.model = "Linear"
    args.n_features=11
    args.lr = 0.002
    processes = []
    run(args,b_t)
    et = time.time()
    print(f"time:{et - st}")