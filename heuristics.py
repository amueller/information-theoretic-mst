import numpy as np
from scipy import sparse
from mst import mst


def cut_biggest(X, n_cluster=2):
    """Single link agglomerative clustering. Cuts longest edge in MST.

    Computes the euclidean MST of the data and cuts the longest edge
    until the desired number of clusters is reached. We avoid single
    point clusters.

    Parameters
    ----------
    X: numpy array, shape=[n_samples, n_features]
        input data
    n_cluster: int
        Desired number of clusters

    Returns
    -------
    labels: numpy array, shape=[n_samples]
        cluster membership indicators

    obj: 0
        Dummy "objective value" of 0 for interface compatibility.
    """
    edges = mst(X)
    weights = edges[:, 2]
    inds = np.argsort(weights)[::-1]
    n_samples = len(edges) + 1
    forest = sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
            shape=(n_samples, n_samples)).tocsr()
    i = 0
    while len(forest.nonzero()[0]) > n_samples - n_cluster:
        e = edges[inds[i]]
        forest[e[0], e[1]] = 0
        if np.min(sparse.cs_graph_components(forest + forest.T)[1]) < 0:
            # only one node in new component. messes up cs_graph_components
            forest[e[0], e[1]] = weights[i]
        elif (np.min(np.bincount(sparse.cs_graph_components(forest +
                forest.T)[1])) < 2):
            # disallow small clusters
            forest[e[0], e[1]] = weights[i]

        i += 1
    return sparse.cs_graph_components(forest + forest.T)[1], 0
