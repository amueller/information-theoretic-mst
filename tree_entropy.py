import numpy as np
import warnings

from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.metrics import euclidean_distances


def tree_information(X, y):
    """Computes mutual information objective using MST entropy estimate.

    Parameters
    ----------
    X: numpy array, shape=[n_samples, n_features]
        datapoints
    y: numpy array, shape=[n_samples], dtype=int
        cluster assignments
    """
    n_samples, n_features = X.shape
    entropy = 0
    classes = np.unique(y)

    for c in classes:
        inds = np.where(y == c)[0]
        if len(inds) == 1:
            continue
        X_ = X[y == c]
        n_samples_c = X_.shape[0]
        L = spanning_tree_length(X_)
        if L == 0:
            warnings.warn("L is zero. This means there are identical points in"
                          "the dataset")
            L = 1e-10
        entropy += n_samples_c * ((n_features - 1) * np.log(n_samples_c) -
                                  n_features * np.log(L))
    return entropy / n_samples


def spanning_tree_length(X):
    """Compute the length of the euclidean MST of X.

    Parameters
    ----------
    X: ndarray, shape=[n_samples, n_features]
    """
    if X.shape[0] < 2:
        return 0
    return minimum_spanning_tree(euclidean_distances(X)).sum()


def tree_information_sparse(forest, n_features):
    """Computes mutual information objective from forest.

    Parameters
    ----------
    forest: sparse matrix
        graph containing trees representing cluster
    n_features: int
        dimensionality of input space.
    """
    entropy = 0
    sym_forest = forest + forest.T
    n_components, components = connected_components(sym_forest)
    if np.any(components < 0):
        # there is a lonely node
        entropy -= 1e10
    # n_samples = len(components)

    for i in range(n_components):
        inds = np.where(components == i)[0]
        subforest = forest[inds[:, np.newaxis], inds]
        L = subforest.sum()
        n_samples_c = len(inds)
        if L == 0:
            warnings.warn("L is zero. This means there are identical points in"
                          " the dataset")
            L = 1e-10
        entropy += (n_samples_c * ((n_features - 1) * np.log(n_samples_c) -
                                   n_features * np.log(L)))
    return entropy
