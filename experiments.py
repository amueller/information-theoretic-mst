import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans

# Normalized mutual information is only available
# in the current development version. See if we can import,
# otherwise use dummy.

normalized_mutual_info_score = lambda x, y: np.NaN
try:
    from sklearn.metrics import normalized_mutual_info_score
except ImportError:
    pass

from heuristics import cut_biggest
from mean_nn import mean_nn
from plot_clustering import plot_clustering
from tree_entropy import tree_information
from itm import itm


def do_experiments(dataset, plot, three_d=False):
    X, y = dataset.data, dataset.target
    dataset_name = dataset.DESCR.split('\n')[0]
    if dataset_name.startswith("Iris"):
        # iris has duplicate data points. That messes up our
        # MeanNN implementation.
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(X))
        doubles = np.unique(np.where(np.tril(dist - 1, -1) == -1)[0])
        mask = np.ones(X.shape[0], dtype=np.bool)
        mask[doubles] = False
        X = X[mask]
        y = y[mask]

    n_cluster = len(np.unique(y))
    print("\n\nDataset %s samples: %d, features: %d, clusters: %d"
            % (dataset_name, X.shape[0], X.shape[1], n_cluster))
    print("=" * 70)

    if plot:
        fig = plt.figure()

    sts = []
    informations = []
    functions = []
    names = []
    functions = [itm, cut_biggest]
    names = ["MST multi cut", "cut biggest"]
    X_plot = X
    for i, method in enumerate(zip(functions, names)):
        function, name = method
        st_, i_ = function(X, n_cluster=n_cluster)
        sts.append(st_)
        informations.append(i_)
        y_ = st_

        if plot:
            ax = fig.add_subplot(1, len(names) + 2, i + 1)
            ax = plot_clustering(X_plot, y_, ax, three_d=three_d)
            ax.set_title("%-15s ARI %.3f, AMI: %.3f, NMI: %.3f obj: %.2f"
                    % (name, adjusted_rand_score(y, y_),
                        adjusted_mutual_info_score(y, y_),
                        normalized_mutual_info_score(y, y_), i_))
        print("%-15s ARI: %.3f, AMI: %.3f, NMI: %.3f objective: %.3f" % (name,
            adjusted_rand_score(y, y_), adjusted_mutual_info_score(y, y_),
            normalized_mutual_info_score(y, y_), i_))
    kmeans = KMeans(k=n_cluster, n_init=1).fit(X)
    kmeans_ARI = adjusted_rand_score(y, kmeans.labels_)
    kmeans_AMI = adjusted_mutual_info_score(y, kmeans.labels_)
    kmeans_NMI = normalized_mutual_info_score(y, kmeans.labels_)
    i_kmeans = tree_information(X, kmeans.labels_)

    # repeat MeanNN ten times, keep best
    opt_value = np.inf
    for init in xrange(10):
        mean_nn_labels_, blub = mean_nn(X, n_cluster=n_cluster)
        if blub < opt_value:
            mean_nn_labels = mean_nn_labels_
            opt_value = blub

    mean_nn_ARI = adjusted_rand_score(y, mean_nn_labels)
    mean_nn_AMI = adjusted_mutual_info_score(y, mean_nn_labels)
    mean_nn_NMI = normalized_mutual_info_score(y, mean_nn_labels)
    i_mean_nn = tree_information(X, mean_nn_labels)

    i_gt = tree_information(X, y)

    if plot:
        kmeans_plot = fig.add_subplot(1, len(names) + 2, len(names) + 1)
        kmeans_plot = plot_clustering(X_plot, kmeans.labels_, kmeans_plot,
                three_d=three_d)
        kmeans_plot.set_title("kmeans ARI: %.3f, AMI: %.3f" % (kmeans_ARI,
            kmeans_AMI))

        mean_nn_plot = fig.add_subplot(1, len(names) + 4, len(names) + 3)
        mean_nn_plot = plot_clustering(X_plot, mean_nn_labels, mean_nn_plot,
                three_d=three_d)
        mean_nn_plot.set_title("mean_nn ARI: %.3f, AMI: %.3f" % (mean_nn_ARI,
            mean_nn_AMI))

        gt_plot = fig.add_subplot(1, len(names) + 2, len(names) + 2)
        gt_plot = plot_clustering(X_plot, y, gt_plot, three_d=three_d)
        gt_plot.set_title("ground truth objective: %.3f" % i_gt)

    print("%-15s ARI: %.3f, AMI: %.3f, NMI: %.3f objective: %.3f" %
            ("MeanNN", mean_nn_ARI, mean_nn_AMI, mean_nn_NMI, i_mean_nn))
    print("%-15s ARI: %.3f, AMI: %.3f, NMI: %.3f objective: %.3f" %
            ("K-Means", kmeans_ARI, kmeans_AMI, kmeans_NMI, i_kmeans))
    print("GT objective: %.3f" % i_gt)

    if plot:
        plt.show()

if __name__ == "__main__":
    from sklearn import datasets
    usps = datasets.fetch_mldata("usps")
    vehicle = datasets.fetch_mldata("vehicle")
    waveform = datasets.fetch_mldata("Waveform IDA")
    vowel = datasets.fetch_mldata("vowel")
    faces = datasets.fetch_olivetti_faces()
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    dataset_list = [digits, faces, iris, vehicle, usps, vowel, waveform]
    for dataset in dataset_list:
        do_experiments(dataset, plot=False)
