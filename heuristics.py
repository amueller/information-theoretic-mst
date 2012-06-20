import numpy as np
import networkx as nx

from mst import mst_dual_boruvka


from IPython.core.debugger import Tracer
tracer = Tracer()


def graph_to_indicator(st):
    y = np.zeros(st.number_of_nodes(), dtype=np.int)
    for i, c in enumerate(nx.connected_component_subgraphs(st)):
        y[np.array(c.nodes())] = i
    return y


def cut_biggest(X, n_cluster=2):
    """Trivial clustering heuristic cutting longest edges"""
    from scipy import sparse
    edges = mst_dual_boruvka(X)
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
        elif np.min(np.bincount(sparse.cs_graph_components(forest + forest.T)[1])) < 2:
            # disallow small clusters
            forest[e[0], e[1]] = weights[i]

        i += 1
    return sparse.cs_graph_components(forest + forest.T)[1], 0
