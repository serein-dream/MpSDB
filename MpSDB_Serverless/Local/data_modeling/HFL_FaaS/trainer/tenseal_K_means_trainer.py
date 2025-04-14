from conf import args_parser
import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from transmission.utils import flatten_tensors
import time as t
import datetime
import matplotlib.pyplot as plt
import oss2
import pickle
sys.path.append("../../../")
def randomly_init_centroid(min_value, max_value, n_dims, repeats=1):
    if repeats == 1:
        return min_value + (max_value - min_value) * np.random.rand(n_dims)
    else:
        return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)
class KMeans:
    def __init__(self, client_centroids, args, dataset, test_dataset, rank):
        self.filetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time = t.perf_counter()
        self.client_centroids = client_centroids
        self.rank = rank
        self.clients_counts = np.zeros(self.client_centroids.shape[0])
        self.client_train_data = dataset
        self.test_data = test_dataset
        self.epoch_lr = args.lr
        self.n_clusters = args.n_clusters
        self.n_dims = args.n_dims
        self.clients_updates = np.zeros((self.n_clusters, self.n_dims))
        self.train_davies_bouldin_list = []
        self.test_davies_bouldin_list = []
    def compute_step_for_client(self, train_data):
        centroids = self.client_centroids
        differences = np.expand_dims(train_data, axis=1) - np.expand_dims(centroids, axis=0)
        sq_dist = np.sum(np.square(differences), axis=2)
        labels = np.argmin(sq_dist, axis=1)
        centroid_updates = np.zeros_like(centroids)
        counts = np.zeros(centroids.shape[0], dtype=int)
        for i in range(centroids.shape[0]):
            mask = np.equal(labels, i)
            counts[i] = np.sum(mask)
            if counts[i] > 0:
                centroid_updates[i] = np.sum(train_data[mask].numpy() - centroids[i], axis=0)
        return centroid_updates, counts
    def davies_bouldin(self, x, labels):
        NUM_CLUSTERS = self.client_centroids.shape[0]
        distances = np.sqrt(np.sum(np.square(x - self.client_centroids[labels]), axis=1))
        centroid_dist_matrix = np.expand_dims(self.client_centroids, axis=0) - np.expand_dims(self.client_centroids, axis=1)
        centroid_dist_matrix = np.sqrt(np.sum(np.square(centroid_dist_matrix), axis=2))
        centroid_dist_matrix[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
        intra_dist = np.zeros(NUM_CLUSTERS)
        for i in range(NUM_CLUSTERS):
            intra_dist[i] = np.mean(distances[i == labels])
        s_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)
        d_i = np.nanmax(s_ij / centroid_dist_matrix, axis=1)
        db_score = np.nanmean(d_i)
        return db_score
    def predict(self, x):
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.client_centroids[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels
    def evaluate(self, splits={'train', 'test'}, use_metric="davies_bouldin"):
        scores = {}
        x = {}
        for split, dataset in zip(splits, [self.client_train_data, self.test_data]):
            data_loader = DataLoader(dataset, batch_size=100000)
            x[split] = np.concatenate([batch[0] for batch in data_loader], axis=0)
            labels = self.predict(x[split])
            if use_metric == "davies_bouldin":
                score = self.davies_bouldin(x[split], labels)
            scores[split] = score
        return scores
    def one_local_round(self):
        client_step_centroids = self.client_centroids
        auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself<ACCESS_KEY_SECRET>')
        bucket = oss2.Bucket(auth, "ADD_by_yourself<ENDPOINT>", "ADD_by_yourself<BUCKET_NAME>")
        for _ in range(self.args.epoch):
            data_loader = DataLoader(self.client_train_data, batch_size=self.args.batch_size)
            for batch in data_loader:
                train_data, _ = batch
                client_updates_sum, client_counts = self.compute_step_for_client(train_data)
                interim_updates = client_updates_sum / np.expand_dims(np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
                if self.epoch_lr is not None:
                    interim_updates *= self.epoch_lr
                client_step_centroids += interim_updates
        clients_local_centroids_expanded = client_step_centroids * np.expand_dims(client_counts, axis=1)
        clients_local_centroids_expanded = torch.tensor(clients_local_centroids_expanded)
        clients_local_centroids_expanded = flatten_tensors(clients_local_centroids_expanded).detach()
        enc_clients_local_centroids_expanded = self.client.enc_tensor(clients_local_centroids_expanded)
        enc_clients_local_centroids_expanded = enc_clients_local_centroids_expanded.serialize()
        client_counts = torch.tensor(client_counts)
        send_clients_counts_dumps = pickle.dumps(client_counts)
        bucket.put_object(f"client_counts_{self.rank}.txt", send_clients_counts_dumps)
        bucket.put_object(f"clients_k_update_params_{self.rank}.txt", enc_clients_local_centroids_expanded)
        while not bucket.object_exists('average_done.txt'):
            t.sleep(0.1)
        ans_s = bucket.get_object('global_params.txt').read()
        ans = pickle.loads(ans_s)
        last_global_centroids_dumps = ans["new_enc_centroids_s"]
        total_counts = ans["total_counts"]
        latest_centroids = self.client.dec_kmeans_tonumpy(last_global_centroids_dumps, self.args.n_dims)
        change_num = latest_centroids / np.expand_dims(np.maximum(total_counts, np.ones_like(total_counts)), axis=1)
        changed = np.any(np.absolute(change_num) > 0.01)
        if changed:
            self.not_change_number = 0
        else:
            self.not_change_number += 1
        return self.not_change_number
    def test(self):
        scores = self.evaluate()
        results = {key: [value] for key, value in scores.items()}
        results_avg = {key: round(np.mean(value), 3) for key, value in results.items()}
        for key in results_avg:
            if results_avg[key] > 5:
                results_avg[key] = 5
        self.train_davies_bouldin_list.append(results_avg['train'])
        self.test_davies_bouldin_list.append(results_avg['test'])
    def write_kmeans(self):
        pa = "<RESULT_PATH>"
        plt.figure()
        plt.plot(range(len(self.train_davies_bouldin_list)), self.train_davies_bouldin_list, color='blue', linestyle="solid", label="Train davies_bouldin")
        plt.plot(range(len(self.test_davies_bouldin_list)), self.test_davies_bouldin_list, color='green', linestyle="solid", label="Test davies_bouldin")
        plt.xlabel('rounds')
        plt.title(f'DBI rank_{self.rank}_n_clusters_{self.args.n_clusters}_lr_{self.args.lr}_rounds_{self.args.rounds}_local_epoch_{self.args.epoch}')
        plt.savefig(f'{pa}/rank_{self.rank}_DBI_n_clusters_{self.args.n_clusters}_lr_{self.args.lr}_rounds_{self.args.rounds}_local_epoch_{self.args.epoch}.png')
    def write(self, begin_time, load_time):
        end_time = t.perf_counter()
        total_time = end_time - self.start_time
        pa = "<TIME_LOG_PATH>"
        fileloader = f"{self.filetime}Linear__wine_quity__lr_{self.args.lr}__epoch_{self.args.epoch}__rounds_{self.args.rounds}__batch_size__{self.args.batch_size}"
        pat = f"<LOG_FILE_PATH>{fileloader}"
        with open(f"{pa}/train_loss_list_{self.rank}.txt", 'w') as f:
            f.write(str(getattr(self, f'train_loss_list_{self.rank}')))
        with open(f"{pa}/train_RMSE_list_{self.rank}.txt", 'w') as f:
            f.write(str(getattr(self, f'train_RMSE_list_{self.rank}')))
        with open(f"{pa}/test_set_RMSE_list_{self.rank}.txt", 'w') as f:
            f.write(str(getattr(self, f'test_set_RMSE_list_{self.rank}')))
        with open(f"{pa}/pat.txt", 'w') as f:
            f.write(pat)
        with open(f"{pa}/time.txt", 'a') as f:
            f.write(f"rank--{self.args.rank}--rounds--{self.args.rounds}--epoch--{self.args.epoch}--model--{self.args.model}--\n"
                    f"begin time:{begin_time}\nload_time:{load_time}\ncom_time:{self.com_time}\ntotal time: {total_time}\n\n")
if __name__ == '__main__':
    args = args_parser()
    # trainer = KMeans(client_centroids, args, dataset, test_dataset, rank)
    # trainer.one_local_round()