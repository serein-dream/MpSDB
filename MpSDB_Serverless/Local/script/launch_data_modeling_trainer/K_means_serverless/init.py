import numpy as np
import oss2
import pickle
import os,sys
sys.path.insert(1,os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from conf import args_parser
import torch
from transmission.utils import flatten_tensors
import tenseal as ts

def randomly_init_centroid(min_value, max_value, n_dims, repeats=1):
    if repeats == 1:
        return min_value + (max_value - min_value) * np.random.rand(n_dims)
    else:
        return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)
        
def do_init_centroids(init_centroids,n_clusters,n_dims):   
    pk_ctx_file = "ADD_by_yourself/transmission/ts_ckks_pk.config" 
    pk_ctx_bytes = open(pk_ctx_file, "rb").read()
    pk_ctx = ts.context_from(pk_ctx_bytes)
    #self.reline_keys = self.pk_ctx.relin_keys()
    if isinstance(init_centroids, str):
        if init_centroids == 'random':
            # # assumes data is in range 0-1
            # centroids = np.random.rand(n_clusters, n_dims)
            # for dummy data
            centroids = randomly_init_centroid(0, n_clusters+1, n_dims, n_clusters)
        else:
            raise NotImplementedError
    elif init_centroids.shape == (n_clusters, n_dims):
        centroids = init_centroids
    else:
        raise NotImplementedError
    centroids = torch.tensor(centroids).detach()
    centroids = flatten_tensors(centroids).detach()
    #print(f"\n\n\ncentroids  :   {type(centroids)}\n\n\n")
    centroids_numpy = ts.plain_tensor(centroids)
    enc_centroids_numpy = ts.ckks_vector(pk_ctx, centroids_numpy)
    print(f"centroids:\n  {centroids}")
    return enc_centroids_numpy



def create_dummy_data(dims=1, clients_per_cluster=10, samples_each=10, clusters=10, scale=0.5, verbose=False):
    num_clients = clients_per_cluster * clusters
    # create gaussian data set, per client one mean
    means = np.arange(1, clusters+1)
    means = np.tile(A=means, reps=clients_per_cluster)
    noise = np.random.normal(loc=0.0, scale=scale, size=(num_clients, samples_each, dims))
    data = np.expand_dims(np.expand_dims(means, axis=1), axis=2) + noise
    if verbose:
        # print(means)
        # print(noise)
        print("dummy data shape: ", data.shape)
    data = [data[i] for i in range(num_clients)]
    return data, means

def load_federated_dummy(seed=None, verbose=False, points_per_cluster=10, clusters=10):
    # assert dims == 1, "only one dimension implemented"
    np.random.seed(seed)
    x = {}
    ids = {}
    data, means = create_dummy_data(clients_per_cluster=2*points_per_cluster, clusters=clusters, verbose=verbose)#为什么*2   因为train和test
    mid = points_per_cluster * clusters
    x["train"], ids["train"] = data[:mid], means[:mid]
    x["test"], ids["test"] = data[mid:], means[mid:]
    # print(len(x['train']), x['train'][0].shape)
    return x, ids



def load_federated(limit_csv=None, verbose=False, seed=None, dummy=False, clusters=None):
    if dummy:
        return load_federated_dummy(seed=seed, verbose=verbose, clusters=clusters)
    else:
        #return load_federated_real(limit_csv, verbose, seed)
        raise 


if __name__ == '__main__':
    args = args_parser()
    init_centroids = "random"
    n_clusters = args.n_clusters
    n_dims = args.n_dims
    auth = oss2.Auth('ADD_by_yourself', 'ADD_by_yourself')
    bucket = oss2.Bucket(auth, "ADD_by_yourself", "usergroup1")
    x, _ = load_federated(
    limit_csv=None,
    verbose=False,
    seed=9549905,
    dummy=True,
    clusters=n_clusters,
)
    data_d = pickle.dumps(x)
    bucket.put_object('data.txt', data_d)
    #print(f"data:\n{x['train']}")


    #wine
    init_centroids = np.array([[6.4,0.25,0.28,4.9,0.03,29.0,98.0,0.99024,3.09,0.58,12.8],
                      [5.7,0.22,0.28,1.3,0.027,26.0,101.0,0.98948,3.35,0.38,12.5],
                      [6.8,0.475,0.33,3.95,0.047,16.0,81.0,0.98988,3.23,0.53,13.4]])
    
    enc_centroids = do_init_centroids(init_centroids,n_clusters,n_dims)

    ini_global_parmas_dumps = enc_centroids.serialize()#pickle.dumps(enc_centroids)
    bucket.put_object('global_params.txt', ini_global_parmas_dumps)
    #print(ini_global_parmas_dumps)
    raise
    clients_list=[]
    t_clients_num=3
    for i in range(t_clients_num):
      clients_list.append(i)
    clients_list_dumps=pickle.dumps(clients_list)
    bucket.put_object("clients_list_dumps.txt",clients_list_dumps)
    '''
    same = 0
    same_d = pickle.dumps(same)
    bucket.put_object("if_stop.txt",same_d)    
    '''
    print("done!")