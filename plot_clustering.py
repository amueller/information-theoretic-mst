import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
Axes3D

from sklearn.decomposition import RandomizedPCA


def plot_clustering(X, y=None, axes=None, three_d=False, forest=None):
    if y is None and forest is None:
        raise ValueError("give me y or a sparse matrix representing the"
                "forest")
    if y is None:
        _, y = sparse.cs_graph_components(forest + forest.T)
    if three_d and X.shape[1] > 3:
        X = RandomizedPCA(n_components=3).fit_transform(X)
    elif not three_d and X.shape[1] > 2:
        X = RandomizedPCA(n_components=2).fit_transform(X)
    if axes == None or three_d:
        plt.figure()
        axes = plt.gca()
    if three_d:
        axes = plt.gca(axes=axes, projection='3d')

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'] * 10)
    color = colors[y]
    if three_d:
        axes.scatter(X[:, 0], X[:, 1], X[:, 2], color=color)
    else:
        axes.scatter(X[:, 0], X[:, 1], color=color, s=10)
    if not forest is None:
        for edge in np.vstack(forest.nonzero()).T:
            i, j = edge
            axes.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], c=color[i])
    axes.set_xticks(())
    axes.set_yticks(())
    return axes
