import numpy as np
import warnings

from scipy.spatial.distance import pdist, squareform
from mst import mst_dual_boruvka


def tree_information(X, y, dists=None, classes=None):
    """ X datapoints, y cluster assignments, dists precomputed
    weights in X, classes = clusters to sum over.
    Computes mutual information between cluster assignments in y and data."""
    n_samples, n_features = X.shape
    entropy = 0
    if dists == None:
        dists = squareform(pdist(X))

    if classes is None:
        classes = np.unique(y)

    for c in classes:
        inds = np.where(y == c)[0]
        if len(inds) == 1:
            continue
        X_ = X[y == c]
        dists_ = dists[inds[:, np.newaxis], inds]
        n_samples_c = X_.shape[0]
        L = spanning_tree_length(X_, dists_)
        if L == 0:
            warnings.warn("L is zero. This means there are identical points in"
                    "the dataset")
            L = 1e-10
        entropy += n_samples_c * ((n_features - 1) * np.log(n_samples_c) -
                                  n_features * np.log(L))
    return entropy / n_samples


def spanning_tree_length(X, dists=None, graph=None):
    if X.shape[0] < 2:
        return 0
    edges = mst_dual_boruvka(X)
    return np.sum([e[2] for e in edges])
