"""
Author: Andreas Mueller 2012
License: BSD 3-Clause

Implements: MeanNN clustering with acceptable runtime.

"""

import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt


def plot_clustering(X, y, plot_num=0, show=True, title=None):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'] * 10)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y])
    if title != None:
        plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig("cluster_%05d.png" % plot_num)
        plt.close()


def update_information(y, dists, flip, informations, proposed_cluster):
    """ Incremental change in cluster informations when moving one point
    to another cluster.

    Parameters
    ----------
    y : cluster indicators
    dists : pairwise distances of datapoints
    flip : data point to flip
    informations: array of current cluster entropies
    proposed_cluster: id of cluster to flip to
    """
    clusterid = y[flip]
    y = y.copy()
    # handle new cluster case:
    n1, n2 = np.bincount(y, minlength=proposed_cluster + 1)[[clusterid,
        proposed_cluster]]
    dist_old = np.log(dists[y == clusterid, flip]).sum()
    dist_new = np.log(dists[y == proposed_cluster, flip]).sum()
    # n1 is size of cluster containing flip
    # subtract from old cluster
    if n1 > 2:
        old = (-informations[clusterid]
            + ((n1 - 1) * informations[clusterid] - dist_old) / (n1 - 2))
    else:
        # hackety hack-hack: the formular doesn't apply for n1 = 1
        old = -informations[clusterid]
    # add to new cluster:
    if n2 > 0:
        new = (-informations[proposed_cluster]
                + ((n2 - 1) * informations[proposed_cluster] + dist_new) / n2)
    else:
        new = 0
    return old, new


def mean_nn(X, dists=None, n_cluster=2, y_init=None):
    """MeanNN implementation using incremental information updates.
    Starts from random initialization or `y_init` if given. """
    if dists is None:
        dists = squareform(pdist(X))
    n_samples = dists.shape[0]

    # hack to not care about distances to self:
    inds = np.arange(n_samples)
    dists[inds, inds[np.newaxis, :]] = 1
    if y_init is None:
        # random initialization
        y = np.random.randint(n_cluster, size=n_samples)
    else:
        y = y_init.copy()

    informations = []
    for c in np.unique(y):
        inds = np.where(y == c)[0]
        unique_dists = np.triu(dists[inds, inds[:, np.newaxis]])
        unique_dists = unique_dists[unique_dists != 0]
        informations.append(np.log(unique_dists).sum() / float(len(inds) - 1))
    informations = np.array(informations)
    information = informations.sum()
    i = 0
    while True:
        #plot_clustering(X, y, i)
        #print("Iteration: %d, information: %f" % (i, information))
        old_information = information
        for sample in np.arange(n_samples):
            # try to flip sample
            for c in np.unique(y):
                if c == y[sample]:
                    # can't flip to label we already have
                    continue
                old, new = update_information(y, dists, sample, informations,
                        c)
                # old + new is decrease in objective after move
                if old + new < 0:
                    informations[y[sample]] += old
                    y[sample] = c
                    informations[c] += new
                    information = np.sum(informations)
        i += 1

        if old_information == information:
            break
    return y, information
