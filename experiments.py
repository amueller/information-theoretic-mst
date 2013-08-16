import numpy as np
from time import time

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans, Ward

# Normalized mutual information is only available
# in the current development version. See if we can import,
# otherwise use dummy.

normalized_mutual_info_score = lambda x, y: np.NaN
try:
    from sklearn.metrics import normalized_mutual_info_score
except ImportError:
    pass

#from heuristics import cut_biggest
#from mean_nn import mean_nn
from tree_entropy import tree_information
from itm import ITM


def do_experiments(dataset):
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

    n_clusters = len(np.unique(y))
    print("\n\nDataset %s samples: %d, features: %d, clusters: %d" %
          (dataset_name, X.shape[0], X.shape[1], n_clusters))
    print("=" * 70)

    classes = [ITM(n_clusters=n_clusters),
               ITM(n_clusters=n_clusters, infer_dimensionality=True),
               Ward(n_clusters=n_clusters), KMeans(n_clusters=n_clusters)]
    names = ["ITM", "ITM ID", "Ward", "KMeans"]
    for clusterer, method in zip(classes, names):
        start = time()
        clusterer.fit(X)
        y_pred = clusterer.labels_

        ari = adjusted_rand_score(y, y_pred)
        ami = adjusted_mutual_info_score(y, y_pred)
        nmi = normalized_mutual_info_score(y, y_pred)
        objective = tree_information(X, y_pred)

        runtime = time() - start

        print("%-15s ARI: %.3f, AMI: %.3f, NMI: %.3f objective: %.3f time:"
              "%.2f" % (method, ari, ami, nmi, objective, runtime))

    i_gt = tree_information(X, y)
    print("GT objective: %.3f" % i_gt)


if __name__ == "__main__":
    from sklearn import datasets
    usps = datasets.fetch_mldata("usps")
    vehicle = datasets.fetch_mldata("vehicle")
    waveform = datasets.fetch_mldata("Waveform IDA")
    vowel = datasets.fetch_mldata("vowel")
    mnist = datasets.fetch_mldata("MNIST original")
    faces = datasets.fetch_olivetti_faces()
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    #dataset_list = [iris, vehicle, vowel, digits, faces, usps, waveform]
    dataset_list = [mnist]
    for dataset in dataset_list:
        do_experiments(dataset)
