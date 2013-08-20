import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

from mst import euclidean_mst


class SingleLink(BaseEstimator, ClusterMixin):
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

    mst_precomputed : array-like or None, default=None
        Precomputed minimum spanning tree, given as weighted edge-list.
        Can be used to speed up computations.

    Attributes
    ----------
    labels_ : numpy array, shape (n_samples,)
        Cluster membership indicators.

    """
    def __init__(self, n_clusters=2, nearest_neighbor_algorithm='auto'):
        self.n_clusters = n_clusters
        self.nearest_neighbor_algorithm = nearest_neighbor_algorithm

    def fit(self, X):
        self.nearest_neighbors_ = NearestNeighbors(algorithm=self.nearest_neighbor_algorithm)
        self.nearest_neighbors_.fit(X)
        forest = euclidean_mst(X, self.nearest_neighbors_)
        weights = forest.data
        inds = np.argsort(weights)[::-1]
        edges = np.vstack(forest.nonzero()).T
        n_samples = len(edges) + 1
        i = 0
        while len(forest.nonzero()[0]) > n_samples - self.n_clusters:
            e = edges[inds[i]]
            forest[e[0], e[1]] = 0
            if np.min(sparse.cs_graph_components(forest + forest.T)[1]) < 0:
                # only one node in new component. messes up cs_graph_components
                forest[e[0], e[1]] = weights[i]
            elif (np.min(np.bincount(sparse.cs_graph_components(forest +
                                                                forest.T)[1])) <
                  2):
                # disallow small clusters
                forest[e[0], e[1]] = weights[i]

            i += 1
        self.labels_ = sparse.cs_graph_components(forest + forest.T)[1]
        return self
