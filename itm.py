import warnings

from scipy.sparse.csgraph import connected_components
from scipy.sparse.base import SparseEfficiencyWarning
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

from block_diag import block_diag
from tree_entropy import tree_information_sparse
from infer_dimensionality import estimate_dimension

from mst import euclidean_mst

DTYPE = np.float
ITYPE = np.int


class ITM(BaseEstimator, ClusterMixin):
    """Information Theoretic Minimum Spanning Tree Clustering.

    Recursively splits dataset into two cluster.
    Finds best cluster to split based on increase of objective.

    Parameters
    ----------
    n_clusters : int, default=None
        Number of clusters the data is split into.

    infer_dimensionality : bool, default=True
        Whether to infer the dimensionality using a dimension estimation
        method.  If False, the input dimensionality will be used.

    nearest_neighbor_algorithm : bool, default='auto'
        Nearest neighbor data structure used for distance queries.
        This parameter is passed on to sklearn.NearestNeighbors.
        Possible choices are 'brute', 'ball_tree', 'kd_tree' and 'auto'.

    verbose : int, default=0
        Verbosity level.

    Returns
    ------
    y : ndarray, shape (n_samples,)
        Cluster labels

    """
    def __init__(self, n_clusters=2, infer_dimensionality=True,
                 nearest_neighbor_algorithm='auto', verbose=0):
        self.n_clusters = n_clusters
        self.infer_dimensionality = infer_dimensionality
        self.nearest_neighbor_algorithm = nearest_neighbor_algorithm
        self.verbose = verbose

    def fit(self, X):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        ------
        self
        """
        n_samples, n_features = X.shape

        self.nearest_neighbors_ = NearestNeighbors(
            algorithm=self.nearest_neighbor_algorithm)
        if self.verbose:
            print("Fitting neighbors data structure.")
        self.nearest_neighbors_.fit(X)
        if self.verbose:
            print("Datastructure used: %s" %
                  self.nearest_neighbors_._fit_method)
        if self.verbose:
            print("Bulding minimum spanning tree.")
        forest = euclidean_mst(X, self.nearest_neighbors_,
                               verbose=self.verbose)

        # the dimensionality of the space can at most be n_samples
        if self.infer_dimensionality:
            if self.verbose:
                print("Estimating dimensionality.")
            intrinsic_dimensionality = estimate_dimension(
                X, neighbors_estimator=self.nearest_neighbors_)
            if self.verbose > 0:
                print("Estimated dimensionality: %d" %
                      intrinsic_dimensionality)
        elif n_samples < n_features:
            warnings.warn("Got dataset with n_samples < n_features. Setting"
                          "intrinsic dimensionality to n_samples. This is most"
                          " likely to high, leading to uneven clusters. It "
                          "is recommendet to set infer_dimensionality=True.")
            intrinsic_dimensionality = n_samples
        else:
            intrinsic_dimensionality = n_features

        if self.verbose:
            print("Cutting spanning tree.")
        clusters = [(forest, np.arange(n_samples))]
        cut_improvement = [itm_binary(forest.copy(), intrinsic_dimensionality,
                                      return_edge=True)]
        # init cluster_infos to anything.
        # doesn't matter any way as there is only one component
        cluster_infos = [0]
        y = np.zeros(n_samples, dtype=np.int)
        removed_edges = []
        # keep all possible next splits, pick the one with highest gain.
        while len(clusters) < self.n_clusters:
            if self.verbose > 1:
                print("Finding for split %d." % len(clusters))
            possible_improvements = (np.array([cut_i[1] * cut_i[0].shape[0] for
                                               cut_i in cut_improvement]) -
                                     np.array(cluster_infos))
            i_to_split = np.argmax(possible_improvements)
            split, info, edge = cut_improvement.pop(i_to_split)
            # get rid of old cluster
            cluster_infos.pop(i_to_split)
            # need the indices of the nodes in the cluster to keep track
            # of where our datapoint went
            _, old_inds = clusters.pop(i_to_split)
            removed_edges.append((old_inds[list(edge[:2])], edge[2]))

            n_split_components, split_components_indicator = \
                connected_components(split + split.T)
            assert(n_split_components == 2)
            assert(len(np.unique(split_components_indicator)) == 2)

            for i in range(n_split_components):
                inds = np.where(split_components_indicator == i)[0]
                clusters.append((split[inds, :][:, inds],
                                 old_inds[inds]))
                mi = tree_information_sparse(clusters[-1][0],
                                             intrinsic_dimensionality)
                cluster_infos.append(mi)
                imp = itm_binary(clusters[-1][0].copy(),
                                 intrinsic_dimensionality, return_edge=True)
                cut_improvement.append(imp)

        # correspondence of nodes to datapoints not present in sparse matrices
        # but we saved the indices.
        c_inds = [c[1] for c in clusters]
        y = np.empty(n_samples, dtype=np.int)
        assert len(np.hstack(c_inds)) == n_samples

        for i, c in enumerate(c_inds):
            y[c] = i

        # for computing the objective, we don't care about the indices
        result = block_diag([c[0] for c in clusters], format='csr')
        self.labels_ = y
        self.tree_information_ = (tree_information_sparse(
            result, intrinsic_dimensionality) / n_samples)
        return self


def itm_binary(graph, intrinsic_dimensionality, return_edge=False):
    """Calculate best split of a MST according to MI objective.

    Calculate split criterion for all edges using "message passing" style
    algorithm to sum edges and count nodes in subtrees. "up" is the root,
    "down" the leaves.

    Parameters
    ----------
    graph : sparse matrix, shape=[n_samples, n_samples]
        non-zero entries represent edges in the MST,
        values give the length of the edge.

    intrinsic_dimensionality : int
        dimensionality of the input space

    return_edge : boolean
        Whether to return the edge that was cut
    """
    n_samples = graph.shape[0]

    graph_sym = graph + graph.T
    graph_sym = graph_sym.tocsr()
    distances = np.asarray(graph_sym.data, dtype=DTYPE, order='C')
    neighbors = np.asarray(graph_sym.indices, dtype=ITYPE, order='C')
    indptr = np.asarray(graph_sym.indptr, dtype=ITYPE, order='C')

    # from leaves to root pass
    # need this to see if all messages have arrived yet so we can go on
    up_message_count = indptr[1:] - indptr[:-1] - 1  # number of children
    visited = np.zeros(n_samples, dtype=np.bool)
    incoming_up = np.zeros(n_samples)
    leaves = np.where(up_message_count == 0)[0]
    nodes_below = np.zeros(n_samples, dtype=np.int)
    to_visit = leaves.tolist()
    while to_visit:
        x = to_visit.pop()
        visited[x] = True
        for i in range(indptr[x], indptr[x + 1]):
            n = neighbors[i]
            if visited[n]:
                # this is where we were coming from
                continue
            incoming_up[n] += incoming_up[x] + distances[i]

            nodes_below[n] += nodes_below[x] + 1
            up_message_count[n] -= 1
            if up_message_count[n] == 0:
                to_visit.append(n)

    # from root back to leaves
    # declare last visited node as "root"
    # we only need that to know which one is the "parent" i.e. closer to the
    # root. We could alternatively first pick a root, then go down the tree,
    # then go back up for the incoming_up.
    root = [x]
    parent = np.zeros(n_samples, dtype=np.int)
    parent[x] = -1
    to_visit = [x]
    visited = np.zeros(n_samples, dtype=np.bool)

    while to_visit:
        x = to_visit.pop()
        visited[x] = True
        for i in range(indptr[x], indptr[x + 1]):
            n = neighbors[i]
            if n != parent[x]:
                parent[n] = x
                to_visit.append(n)

    best_cut = None
    best_objective = -np.inf
    for x in range(n_samples):
        if parent[x] == -1:
            # was the root, doesn't have parent
            continue
        p = parent[x]
        # sum in parent part:
        p_weights = float(incoming_up[root] - incoming_up[x] - graph_sym[x, p])
        # sum in child part:
        c_weights = float(incoming_up[x])
        # nodes in child part:
        c_nodes = nodes_below[x] + 1  # count self
        if c_nodes <= 2:
            # single node
            continue
        p_nodes = n_samples - c_nodes
        if p_nodes <= 2:
            # single node
            continue

        assert(p_weights > 0)
        objective = (p_nodes * ((intrinsic_dimensionality - 1) *
                                np.log(p_nodes)
                     - intrinsic_dimensionality * np.log(p_weights)))
        objective += (c_nodes * ((intrinsic_dimensionality - 1) *
                                 np.log(c_nodes)
                      - intrinsic_dimensionality * np.log(c_weights)))
        if objective > best_objective:
            best_cut = x
            best_objective = objective
    if best_cut is None:
        return graph, -np.inf
    best_objective /= n_samples
    with warnings.catch_warnings():
        # catch sparse efficiency warning for setting elements in CSR
        warnings.simplefilter('ignore', SparseEfficiencyWarning)
        graph[best_cut, parent[best_cut]] = 0
        graph[parent[best_cut], best_cut] = 0

    if return_edge:
        weight_best_cut = graph_sym[best_cut, parent[best_cut]]
        # edges are sorted!
        if best_cut < parent[best_cut]:
            edge = [best_cut, parent[best_cut]]
        else:
            edge = [parent[best_cut], best_cut]
        return graph, best_objective, (edge[0], edge[1], weight_best_cut)
    return graph, best_objective
