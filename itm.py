from scipy import sparse
from tree_entropy import mst_dual_boruvka, tree_information_sparse
#import mst_split

from IPython.core.debugger import Tracer
tracer = Tracer()

import matplotlib.pyplot as plt
from plot_clustering import plot_clustering

import numpy as np
DTYPE = np.float
ITYPE = np.int


def mst_multi_split_expansion(X, n_cluster=2, return_everything=False):
    # init:
    n_samples, n_features = X.shape
    forest, removed_edges = mst_multi_split(X, n_cluster=n_cluster,
            return_everything=True)
    old_objective = -np.inf
    new_objective = tree_information_sparse(forest, n_features)
    iteration = 0
    while old_objective < new_objective:
        print("iteration %d" % iteration)
        old_objective = new_objective
        new_removed_edges = []
        for edge in removed_edges:
            plot_clustering(X, forest=forest)
            plt.savefig("blub_%d" % iteration)
            plt.close()
            n_cluster_, y = sparse.cs_graph_components(forest + forest.T)
            assert(n_cluster_ == n_cluster)
            comps = y[edge[0]]
            mask = np.where((y == comps[0]) + (y == comps[1]))[0]
            component = forest[mask[:, np.newaxis], mask]
            # place of matrix entry corresponding to edge in
            # the submatrix correstponding to the component
            mapped_edge = [int(np.where(mask == e)[0]) for e in edge[0]]
            component[mapped_edge[0], mapped_edge[1]] = edge[1]
            graph, objective, removed_edge = mst_split_test(component,
                    n_features, return_edge=True)
            mapped_re = [mask[removed_edge[0]], mask[removed_edge[1]]]
            if (np.asarray(edge[0]) == mapped_re).all():
                new_removed_edges.append(edge)
            else:
                print("replaced %s by %s" % (edge[0], mapped_re))
                forest[edge[0][0], edge[0][1]] = edge[1]
                new_removed_edges.append([mapped_re, removed_edge[2]])
                forest[mapped_re[0], mapped_re[1]] = 0

        new_objective = tree_information_sparse(forest, n_features)
        removed_edges = new_removed_edges
        iteration += 1

    if return_everything:
        return forest, removed_edges

    return y, tree_information_sparse(forest, n_features) / n_samples


def mst_multi_split(X, n_cluster=2, return_everything=False):
    """Recursively splits dataset into two cluster.
    Finds best cluster to split based on increase of objective."""

    n_samples, n_features = X.shape
    edges = mst_dual_boruvka(X)
    weights = edges[:, 2]
    edges = edges[:, :2].astype(np.int)
    forest = sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
            shape=(n_samples, n_samples)).tocsr()
    clusters = [(forest, np.arange(n_samples))]
    cut_improvement = [mst_split_test(forest.copy(), n_features,
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

        n_split_components, split_components_indicator = sparse.cs_graph_components(split + split.T)
        assert(n_split_components == 2)

        for i in xrange(n_split_components):
            inds = np.where(split_components_indicator == i)[0]
            clusters.append((split[inds[np.newaxis, :], inds], old_inds[inds]))
            cluster_infos.append(tree_information_sparse(clusters[-1][0], n_features))
            cut_improvement.append(mst_split_test(clusters[-1][0].copy(), n_features, return_edge=True))

    # correspondence of nodes to datapoints not present in sparse matrices
    # but we saved the indices.
    c_inds = [c[1] for c in clusters]
    y = np.empty(n_samples, dtype=np.int)
    for i, c in enumerate(c_inds):
        y[c] = i

    # for computing the objective, we don't care about the indices
    result = sparse.block_diag([c[0] for c in clusters], format='csr')
    if return_everything:
        concatenated_inds = [x for c in c_inds for x in c]
        inverse_inds = np.argsort(concatenated_inds)
        sorted_result = result[inverse_inds[:, np.newaxis], inverse_inds]
        return sorted_result, removed_edges
    return y, tree_information_sparse(result, n_features) / n_samples


def mst_split_test(graph, n_features, return_edge=False):
    """ calculate split criterion for all edges
    using "message passing" style algorithm to sum edges
    and count nodes in subtrees. "up" is the root, "down" the leaves."""
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
        for i in xrange(indptr[x], indptr[x+1]):
            n = neighbors[i]
            if n != parent[x]:
                incoming_down[n] += incoming_down[x] + distances[i] + incoming_up_accumulated[x] - incoming_up[x, n]
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
        p_weights = float(incoming_down[p] + incoming_up_accumulated[p] - incoming_up_accumulated[x] - graph_sym[x, p])
        # sum in child part:
        c_weights = float(incoming_up_accumulated[x])
        # nodes in child part:
        c_nodes = nodes_below[x] + 1 # count self
        if c_nodes <= 1:
            # single node
            continue
        p_nodes = n_samples - c_nodes
        if p_nodes <= 1:
            # single node
            continue

        if p_weights < 0:
            tracer()
        objective = p_nodes * ((n_features - 1) * np.log(p_nodes) - n_features * np.log(p_weights))
        objective += c_nodes * ((n_features - 1) * np.log(c_nodes) - n_features * np.log(c_weights))
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
