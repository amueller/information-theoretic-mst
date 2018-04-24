import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


def euclidean_mst(X, neighbors_estimator, verbose=2):
    n_neighbors = min(2, X.shape[0])
    while True:
        # make sure we have a connected minimum spanning tree.
        # otherwise we need to consider more neighbors
        n_neighbors = 2 * n_neighbors
        if verbose > 1:
            print("Trying to build mst with %d neighbors." % n_neighbors)
        distances = neighbors_estimator.kneighbors_graph(
            X, n_neighbors=n_neighbors, mode='distance')
        n_components, component_indicators =\
            connected_components(distances + distances.T)
        if len(np.unique(component_indicators)) > 1:
            continue
        distances.sort_indices()
        forest = minimum_spanning_tree(distances)
        _, inds = connected_components(forest + forest.T)
        assert(len(np.unique(inds)) == 1)
        break
    return forest
