import numpy as np
import warnings

from scipy import sparse
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


def tree_information_sparse(forest, n_features):
    """ X datapoints, y cluster assignments, dists precomputed
    weights in X, classes = clusters to sum over.
    Computes mutual information between cluster assignments in y and data."""
    entropy = 0
    sym_forest = forest + forest.T
    n_components, components = sparse.cs_graph_components(sym_forest)
    if np.any(components < 0):
        # there is a lonely node
        entropy -= 1e10
    #n_samples = len(components)

    for i in xrange(n_components):
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


def tree_information_graph(graph, n_features, edges=None):
    """ graph
    Computes mutual information between cluster assignments given by connected components and data."""
    if edges is None:
        edges = np.array(graph.edges(data=True))
    entropy = 0
    connected_components = nx.connected_components(graph)
    num_components = len(connected_components)
    connected_component_length = np.zeros(num_components)
    # make node_weights very long. doesn't matter, later we use only the entries that exist.
    #node_weights = np.zeros(np.max(graph.nodes()) + 1)
    weights = np.array(map(lambda x: x['weight'], edges[:, 2]))
    edges = edges[:, :2].astype(np.int)
    # assign edge weight to arbitrary adjacent node, since we have
    # components as lists of nodes, not edges
    #for i, w in enumerate(weights):
        #node_weights[edges[i, 0]] += w
    node_weights = np.bincount(edges[:,0], weights, minlength=np.max(graph.nodes()) + 1)

    #for edge in graph.edges_iter(data=True):
    for i, cc in enumerate(connected_components):
        cc = np.array(cc)
        connected_component_length[i] = node_weights[cc].sum()

    for i, cc in enumerate(connected_components):
        n_nodes = len(cc)
        if n_nodes == 1:
        #if n_nodes < 10:
            # single point
            entropy -= 1e10
            continue
        #L = np.sum([e[2]['weight'] for e in connected_component.edges_iter(data=True)])
        L = connected_component_length[i]
        if L == 0:
            warnings.warn("L is zero. This means there are identical points in the dataset")
            L = 1e-10
        #n_features = 2
        new_entropy = n_nodes * ((n_features - 1) * np.log(n_nodes) - n_features * np.log(L))
        entropy += new_entropy
    return entropy


