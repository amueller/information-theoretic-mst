import numpy as np
import matplotlib.pyplot as plt

from mst_split_test import mst_multi_split
from heuristics import cut_biggest
from plot_clustering import plot_clustering
from mean_nn import mean_nn
from sklearn.preprocessing import Scaler

from heuristics import graph_to_indicator

from IPython.core.debugger import Tracer
tracer = Tracer()

n_samples = 40
n_noise = 10
np.random.seed(0)
X1 = np.random.normal(size=(n_samples, 2))
X2 = np.random.normal(size=(n_samples, 2))
X3 = np.random.normal(size=(n_samples, 2))
X4 = np.random.uniform(size=(n_noise, 2)) * np.array([20, 20])
X1 -= X1.mean(axis=0)
X2 -= X2.mean(axis=0)
X3 -= X3.mean(axis=0)
X4 -= X4.mean(axis=0)

plt.figure()

X1[:, 0] -=  6
X2[:, 0] += 6
X3[:, 1] += 5
X = np.vstack([X1, X2, X3, X4])
# plot data
#st_plot = plt.subplot(131)
st_plot = plt.subplot(111)
forest, _ = mst_multi_split(X, n_cluster=1, return_everything=True)
plot_clustering(X, forest=forest, axes=st_plot)
plt.xticks(())
plt.yticks(())
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig("illustration_left.pdf")
plt.close()

# mst clustering
#st1_plot = plt.subplot(132)
st1_plot = plt.subplot(111)
forest, _ = mst_multi_split(X, n_cluster=2, return_everything=True)
plot_clustering(X, forest=forest, axes=st1_plot)
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig("illustration_center.pdf")
plt.close()
# mst flip tree
#st2_plot = plt.subplot(133)
st2_plot = plt.subplot(111)
forest, _ = mst_multi_split(X, n_cluster=3, return_everything=True)
plot_clustering(X, forest=forest, axes=st2_plot)
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig("illustration_right.pdf")


# load / generate datasets
from sklearn import datasets
from sklearn.cluster import KMeans
noise = np.random.uniform(size=(10, 2))
noise = Scaler().fit_transform(noise) * 1
data = np.loadtxt("twomoons-test.txt")
twomoons = np.vstack([data[:, 1:], noise])
twomoons = Scaler().fit_transform(twomoons)
#twomoons, _ = datasets.make_moons()
blobs, _ = datasets.make_blobs(random_state=10)
blobs = np.vstack([Scaler().fit_transform(blobs), noise])
from make_circles import make_circles
circles, _ = make_circles(factor=.4, noise=.05, n_samples=200)
circles = np.vstack([circles, noise])
circles2, _ = make_circles(factor=.1, noise=.05, n_samples=200)
circles2 = np.vstack([circles2, noise])

for data in ["twomoons", "blobs", "circles", "circles2"]:
    X = eval(data)
    if data in ["twomoons", "circles", "circles2"]:
        n_cluster = 2
    else:
        n_cluster = 3
    km = KMeans(k=n_cluster).fit(X)
    plot_clustering(X, km.labels_)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig("qualitative_%s_km.pdf" % data)
    plt.close()

    mst_st, _ = mst_multi_split(X, n_cluster=n_cluster)
    plot_clustering(X, mst_st)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig("qualitative_%s_mst.pdf" % data)
    plt.close()

    mst_agg, _ = cut_biggest(X, n_cluster=n_cluster)
    y_agg = graph_to_indicator(mst_agg)
    plot_clustering(X, y_agg)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig("qualitative_%s_aggl.pdf" % data)
    plt.close()

    best = np.inf
    for i in xrange(10):
        labels_, x = mean_nn(X, n_cluster=n_cluster)
        if x < best:
            best = x
            labels_mean_nn = labels_
    #mean_nn_st = graph_from_labels(X, labels_mean_nn)
    plot_clustering(X, labels_mean_nn)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig("qualitative_%s_mean_nn.pdf" % data)
    plt.close()
