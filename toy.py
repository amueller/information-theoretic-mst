import numpy as np
import matplotlib.pyplot as plt

from itm import ITM
from heuristics import cut_biggest
from plot_clustering import plot_clustering
from mean_nn import mean_nn

from sklearn.preprocessing import Scaler
from sklearn import datasets
from sklearn.cluster import KMeans
from make_circles import make_circles


# first, illustrate progress of algorithm
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

plt.figure(figsize=(12, 4))

X1[:, 0] -= 6
X2[:, 0] += 6
X3[:, 1] += 5
X = np.vstack([X1, X2, X3, X4])
# plot data
st_plot = plt.subplot(131, title="full mst")
forest, _ = itm(X, n_cluster=1, return_everything=True)
plot_clustering(X, forest=forest, axes=st_plot)
# first cut
st1_plot = plt.subplot(132, title="first cut")
forest, _ = itm(X, n_cluster=2, return_everything=True)
plot_clustering(X, forest=forest, axes=st1_plot)
# second cut
st2_plot = plt.subplot(133, title="second cut")
forest, _ = itm(X, n_cluster=3, return_everything=True)
plot_clustering(X, forest=forest, axes=st2_plot)
plt.show()


# load / generate datasets for comparison
noise = np.random.uniform(size=(10, 2))
noise = Scaler().fit_transform(noise) * 1
data = np.loadtxt("twomoons-test.txt")
twomoons = np.vstack([data[:, 1:], noise])
twomoons = Scaler().fit_transform(twomoons)
blobs, _ = datasets.make_blobs(random_state=10)
blobs = np.vstack([Scaler().fit_transform(blobs), noise])
circles, _ = make_circles(factor=.4, noise=.05, n_samples=200)
circles = np.vstack([circles, noise])
circles2, _ = make_circles(factor=.1, noise=.05, n_samples=200)
circles2 = np.vstack([circles2, noise])

plt.figure(figsize=(12, 10))

for i, data in enumerate(["twomoons", "blobs", "circles", "circles2"]):
    X = eval(data)
    if data in ["twomoons", "circles", "circles2"]:
        n_cluster = 2
    else:
        n_cluster = 3
    km = KMeans(k=n_cluster).fit(X)
    ax = plt.subplot(4, 4, 4 * i + 1)
    if i is 0:
        ax.set_title("k-means")
    plot_clustering(X, km.labels_, axes=ax)

    mst_st, _ = itm(X, n_cluster=n_cluster)
    ax = plt.subplot(4, 4, 4 * i + 2)
    if i is 0:
        ax.set_title("itm")
    plot_clustering(X, mst_st, axes=ax)

    y_agg, _ = cut_biggest(X, n_cluster=n_cluster)
    ax = plt.subplot(4, 4, 4 * i + 3)
    if i is 0:
        ax.set_title("single link")
    plot_clustering(X, y_agg, axes=ax)

    best = np.inf
    # we do it only once for illustration purposes
    # figure in paper used best of 10 initializations
    labels_mean_nn, x = mean_nn(X, n_cluster=n_cluster)
    ax = plt.subplot(4, 4, 4 * i + 4)
    if i is 0:
        ax.set_title("meanNN")
    plot_clustering(X, labels_mean_nn, axes=ax)
plt.subplots_adjust(.02, .02, .98, .95, 0.1, 0.1)
plt.show()
