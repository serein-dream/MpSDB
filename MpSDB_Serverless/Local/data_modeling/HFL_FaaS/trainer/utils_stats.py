import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
def davies_bouldin(x, labels, centroids, verbose=False):
    NUM_CLUSTERS = centroids.shape[0]
    distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))
    centroid_dist_matrix = np.sqrt(np.sum(np.square(np.expand_dims(centroids, axis=0) - np.expand_dims(centroids, axis=1)), axis=2))
    centroid_dist_matrix[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
    intra_dist = np.array([np.mean(distances[i == labels]) for i in range(NUM_CLUSTERS)])
    s_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)
    d_i = np.nanmax(s_ij / centroid_dist_matrix, axis=1)
    db_score = np.nanmean(d_i)
    return db_score
def euclidean_dist(x, labels, centroids):
    distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))
    return np.mean(distances)
def evaluate(kmeans, x, splits, use_metric='euclidean', federated=False):
    scores = {}
    centroids = kmeans.cluster_centers_
    for split in splits:
        if federated:
            x[split] = np.concatenate(x[split], axis=0)
        labels = kmeans.predict(x[split])
        if "davies_bouldin" == use_metric:
            score = davies_bouldin(x[split], labels, centroids)
        elif "silhouette" == use_metric:
            score = metrics.silhouette_score(x[split], labels)
        else:
            assert use_metric == 'euclidean'
            score = euclidean_dist(x[split], labels, centroids)
        scores[split] = score
    return scores
def plot_stats(stats, x_variable, x_variable_name, metric_name):
    for spl, spl_dict in stats.items():
        for stat, stat_values in spl_dict.items():
            stats[spl][stat] = np.array(stat_values)
    x_variable = ["single" if i == 0.0 else i for i in x_variable]
    x_axis = np.array(range(len(x_variable)))
    plt.plot(stats['train']['avg'], 'r-', label='Train')
    plt.plot(stats['test']['avg'], 'b-', label='Test')
    plt.fill_between(x_axis, stats['train']['avg'] - stats['train']['std'], stats['train']['avg'] + stats['train']['std'], facecolor='r', alpha=0.3)
    plt.fill_between(x_axis, stats['test']['avg'] - stats['test']['std'], stats['test']['avg'] + stats['test']['std'], facecolor='b', alpha=0.2)
    plt.xticks(x_axis, x_variable)
    plt.xlabel(x_variable_name)
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f"results/stats_{x_variable_name}.png", dpi=600, bbox_inches='tight')
    plt.show()
def plot_progress(progress_means, progress_stds, record_at):
    num_clusters = progress_means[0].shape[0]
    num_records = len(progress_means)
    true_means = np.arange(1, num_clusters+1)
    fig = plt.figure()
    for i in range(num_clusters):
        ax = fig.add_subplot(1, 1, 1)
        x_axis = np.array(range(num_records))
        true_means_i = np.repeat(true_means[i], num_records)
        means = np.array([x[i] for x in progress_means])
        stds = np.array([x[i] for x in progress_stds])
        ax.plot(means, 'r-', label='centroid mean')
        ax.plot(true_means_i, 'b-', label='true mean')
        ax.fill_between(x_axis, means - stds, means + stds, facecolor='r', alpha=0.4, label='centroid std')
        plt.xticks(x_axis, record_at)
    plt.xlabel("Round")
    plt.ylabel("Cluster distribution")
    plt.savefig("results/stats_progress.png", dpi=600, bbox_inches='tight')
    plt.show()