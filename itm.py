from scipy import sparse
import numpy as np

from block_diag import block_diag
from mst import mst
from tree_entropy import tree_information_sparse

DTYPE = np.float
ITYPE = np.int


def itm(X, n_cluster=2, return_everything=False):
    """Information Theoretic Minimum Spanning Tree Clustering.

    Recursively splits dataset into two cluster.
    Finds best cluster to split based on increase of objective.

    Parameters
    ----------
    X: ndarray, shape=[n_samples, n_features]
        Input data
    n_clusters: int
        number of clusters the data is split into
    return_everything: bool
        whether to return the euclidean MST, removed edges and objectives
        or just the resulting clustering.

    Returns
    ------
    y: ndarray, shape=[n_samples]
        Cluster labels
    obj: float
        objective value of solution
    """

    n_samples, n_features = X.shape
    edges = mst(X)
    weights = edges[:, 2]
    edges = edges[:, :2].astype(np.int)
    forest = sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
            shape=(n_samples, n_samples)).tocsr()
    clusters = [(forest, np.arange(n_samples))]
    cut_improvement = [itm_binary(forest.copy(), n_features,
        return_edge=True)]
    # init cluster_infos to anything.
    # doesn't matter any way as there is only one component
    cluster_infos = [0]
    y = np.zeros(n_samples, dtype=np.int)
    removed_edges = []
    # keep all possible next splits, pick the one with highest gain.
    while len(clusters) < n_cluster:
        possible_improvements = (np.array([cut_i[1] * cut_i[0].shape[0]
            for cut_i in cut_improvement]) - np.array(cluster_infos))
        i_to_split = np.argmax(possible_improvements)
        split, info, edge = cut_improvement.pop(i_to_split)
        # get rid of old cluster
        cluster_infos.pop(i_to_split)
        # need the indices of the nodes in the cluster to keep track
        # of where our datapoint went
        _, old_inds = clusters.pop(i_to_split)
        removed_edges.append((old_inds[list(edge[:2])], edge[2]))

        n_split_components, split_components_indicator = \
            sparse.cs_graph_components(split + split.T)
        assert(n_split_components == 2)

        for i in xrange(n_split_components):
            inds = np.where(split_components_indicator == i)[0]
            clusters.append((split[inds[np.newaxis, :], inds], old_inds[inds]))
            mi = tree_information_sparse(clusters[-1][0], n_features)
            cluster_infos.append(mi)
            imp = itm_binary(clusters[-1][0].copy(), n_features,
                    return_edge=True)
            cut_improvement.append(imp)

    # correspondence of nodes to datapoints not present in sparse matrices
    # but we saved the indices.
    c_inds = [c[1] for c in clusters]
    y = np.empty(n_samples, dtype=np.int)
    for i, c in enumerate(c_inds):
        y[c] = i

    # for computing the objective, we don't care about the indices
    result = block_diag([c[0] for c in clusters], format='csr')
    if return_everything:
        concatenated_inds = [x for c in c_inds for x in c]
        inverse_inds = np.argsort(concatenated_inds)
        sorted_result = result[inverse_inds[:, np.newaxis], inverse_inds]
        return sorted_result, removed_edges
    return y, tree_information_sparse(result, n_features) / n_samples


def itm_binary(graph, n_features, return_edge=False):
    """Calculate best split of a MST according to MI objective.

    Calculate split criterion for all edges using "message passing" style
    algorithm to sum edges and count nodes in subtrees. "up" is the root,
    "down" the leaves.

    Parameters
    ----------
    graph: sparse matrix, shape=[n_samples, n_samples]
        non-zero entries represent edges in the MST,
        values give the length of the edge.
    n_features: int
        dimensionality of the input space
    return_edge: boolean
        Whether to return the edge that was cut
    """
    n_samples = graph.shape[0]

    graph_sym = graph + graph.T
    graph_sym = graph_sym.tocsr()
    distances = np.asarray(graph_sym.data,
                            dtype=DTYPE, order='C')
    neighbors = np.asarray(graph_sym.indices,
                            dtype=ITYPE, order='C')
    indptr = np.asarray(graph_sym.indptr,
                         dtype=ITYPE, order='C')

    # from leaves to root pass
    # need this to see if all messages have arrived yet so we can go on
    up_message_count = indptr[1:] - indptr[:-1] - 1  # number of children
    visited = np.zeros(n_samples, dtype=np.bool)
    incoming_up = sparse.lil_matrix((n_samples, n_samples))
    incoming_up_accumulated = np.zeros(n_samples)
    leaves = np.where(up_message_count == 0)[0]
    nodes_below = np.zeros(n_samples, dtype=np.int)
    to_visit = leaves.tolist()
    while to_visit:
        x = to_visit.pop()
        visited[x] = True
        for i in xrange(indptr[x], indptr[x + 1]):
            n = neighbors[i]
            if visited[n]:
                #this is where we were coming from
                continue
            incoming_up[n, x] = incoming_up_accumulated[x] + distances[i]
            incoming_up_accumulated[n] += incoming_up[n, x]

            nodes_below[n] += nodes_below[x] + 1
            up_message_count[n] -= 1
            if up_message_count[n] == 0:
                to_visit.append(n)

    # from root back to leaves
    # declare last visited node as "root"

    to_visit = [x]
    parent = np.zeros(n_samples, dtype=np.int)
    parent[x] = -1
    visited = np.zeros(n_samples, dtype=np.bool)
    incoming_down = np.zeros(graph_sym.shape[0])

    #root to leave pass
    while to_visit:
        x = to_visit.pop()
        visited[x] = True
        for i in xrange(indptr[x], indptr[x + 1]):
            n = neighbors[i]
            if n != parent[x]:
                incoming_down[n] += (incoming_down[x] + distances[i] +
                        incoming_up_accumulated[x] - incoming_up[x, n])
                parent[n] = x
                to_visit.append(n)

    best_cut = None
    best_objective = -np.inf
    for x in xrange(n_samples):
        if parent[x] == -1:
            # was the root, doesn't have parent
            continue
        p = parent[x]
        # sum in parent part:
        p_weights = float(incoming_down[p] + incoming_up_accumulated[p] -
                incoming_up_accumulated[x] - graph_sym[x, p])
        # sum in child part:
        c_weights = float(incoming_up_accumulated[x])
        # nodes in child part:
        c_nodes = nodes_below[x] + 1  # count self
        if c_nodes <= 1:
            # single node
            continue
        p_nodes = n_samples - c_nodes
        if p_nodes <= 1:
            # single node
            continue

        assert(p_weights > 0)
        objective = (p_nodes * ((n_features - 1) * np.log(p_nodes)
                    - n_features * np.log(p_weights)))
        objective += (c_nodes * ((n_features - 1) * np.log(c_nodes)
                    - n_features * np.log(c_weights)))
        if objective > best_objective:
            best_cut = x
            best_objective = objective
    if best_cut is None:
        return graph, -np.inf

    best_objective /= n_samples
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
