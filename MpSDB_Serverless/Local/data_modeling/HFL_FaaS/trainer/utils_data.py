import os
import numpy as np
import pandas as pd
import matplotlib as plt
def load(subsample_train_frac=None, num_train=None, num_test=None, verbose=False, seed=None):
    np.random.seed(seed)
    train_path = os.path.join(os.path.dirname(os.getcwd()), "data", "train")
    test_path = os.path.join(os.path.dirname(os.getcwd()), "data", "test")
    x = {}
    x['train'] = _load_csv(os.path.join(train_path, "x.csv"), num_train)
    x['test'] = _load_csv(os.path.join(test_path, "x.csv"), num_test)
    if subsample_train_frac:
        house_ids = _load_house_ids(os.path.join(train_path, "house_ids.csv"), num_train)
        unique_ids = np.unique(house_ids)
        subsample_num = max(1, int(subsample_train_frac * len(unique_ids)))
        train_ids = np.random.choice(unique_ids, size=subsample_num, replace=False)
        train_mask = np.isin(house_ids, train_ids)
        val_mask = np.invert(train_mask)
        x['val'] = x['train'][val_mask, :]
        x['train'] = x['train'][train_mask, :]
    if verbose:
        _plot_random_samples(x['train'], 5)
    return x
def _load_csv(file_path, nrows=None):
    return pd.read_csv(file_path, index_col=False, header=None, nrows=nrows).values
def _load_house_ids(file_path, nrows=None):
    return pd.read_csv(file_path, index_col=False, header=None, nrows=nrows).values[:, 0]
def _plot_random_samples(data, num_samples):
    rand_plot_idx = np.random.choice(range(data.shape[0]), size=num_samples, replace=False)
    plt.plot(data[rand_plot_idx, :].T)
    plt.show()
def create_dummy_data(dims=1, clients_per_cluster=10, samples_each=10, clusters=10, scale=0.5, verbose=False):
    num_clients = clients_per_cluster * clusters
    means = np.tile(np.arange(1, clusters+1), reps=clients_per_cluster)
    noise = np.random.normal(loc=0.0, scale=scale, size=(num_clients, samples_each, dims))
    data = np.expand_dims(np.expand_dims(means, axis=1), axis=2) + noise
    if verbose:
        print("dummy data shape: ", data.shape)
    return [data[i] for i in range(num_clients)], means
def load_federated_dummy(seed=None, verbose=False, clients_per_cluster=10, clusters=10):
    np.random.seed(seed)
    data, means = create_dummy_data(clients_per_cluster=2*clients_per_cluster, clusters=clusters, verbose=verbose)
    mid = clients_per_cluster * clusters
    return {"train": data[:mid], "test": data[mid:]}, {"train": means[:mid], "test": means[mid:]}
def load_federated(limit_csv=None, verbose=False, seed=None, dummy=False, clusters=None):
    return load_federated_dummy(seed=seed, verbose=verbose, clusters=clusters) if dummy else load_federated_real(limit_csv, verbose, seed)
def load_federated_real(limit_csv=None, verbose=False, seed=None):
    np.random.seed(seed)
    x = {}
    client_ids = {}
    for spl in ['train', 'test']:
        path = os.path.join(os.path.dirname(os.getcwd()), "data", f"{spl}")
        data = _load_csv(os.path.join(path, "x.csv"), limit_csv)
        house_ids = _load_house_ids(os.path.join(path, "house_ids.csv"), limit_csv)
        data.set_index(house_ids, inplace=True)
        df = data.groupby(level=0, sort=True).apply(lambda y: y.values)
        x[spl] = list(df.values)
        client_ids[spl] = list(df.index.values)
    if verbose:
        _plot_random_samples(x['train'][0], 5)
    return x, client_ids
def gaussian_density(mean, var, x):
    return np.exp(-np.square(x - mean) / (2*var)) / np.sqrt(2*np.pi*var)
def gaussian_24(mean, var):
    return gaussian_density(mean, var, np.arange(24))
def get_random_gaussian_mixtures(mix_num=2, repeats=1, dims=24, seed=None):
    np.random.seed(seed)
    centroids = np.zeros((repeats, dims))
    for r in range(repeats):
        mean = np.random.randint(-2, dims+2, mix_num)
        var = np.random.randint(1, dims, mix_num)
        for i in range(mix_num):
            centroids[r, :] += gaussian_24(mean[i], var[i]) * (0.5 + np.random.random())
    centroids += 0.5
    centroids /= np.expand_dims(np.sum(centroids, axis=1), axis=1)
    centroids *= dims * (0.5 + np.random.random())
    return centroids
def init_centroids_gmm(init_centroids, num_clusters, seed, dims, verbose=False):
    if init_centroids == "GMM":
        init_centroids = get_random_gaussian_mixtures(mix_num=3, repeats=num_clusters, dims=dims, seed=seed)
        if verbose:
            plt.plot(init_centroids.T)
            plt.show()
    return init_centroids
def test():
    create_dummy_data(dims=1, clients_per_cluster=2, samples_each=5, clusters=3, verbose=True)
if __name__ == "__main__":
    test()