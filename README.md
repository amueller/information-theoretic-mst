information-theoretic-mst
=========================

Information Theoretic Clustering using Minimum Spanning TreesA

This is the code accompanying "Information Theoretic Clustering using Minimum Spanning Trees"
by Andreas C. Mueller, Sebastian Nowozin, and Christoph H. Lampert

Please cite the paper if you use this code.


FEATURES
--------
Implements MeanNN (See 


DEPENDENCIES
------------
To run the algorithm, you need to install mlpack from http://mlpack.org/ for the
euclidean MST algorithm. Any other algorithm can also be used, but this is particularly fast.

For doing the experiments, scikit-learn is used for kmeans and computing the clustering scores.
You can install scikit-learn via pipy or download from scikit-learn.org.
For the normalized mutual information to work, you need scikit-learn >= 0.12-git


USAGE
-----
To reproduce the results in the paper, simply run ``python experiments.py``.
As this will download all datasets from mldata.org (once), you will need an internet connection.
The toy examples can be reproduced using ``python toy.py``.


FILES
-----
``experiments.py`` - reproduce UCI data experiments from paper
``itm.py`` - implements information theoretic clustering using minimum spanning trees
``mean_nn.py`` - implements MeanNN (see ..)
``mst.py`` - wraps several MST methods, in particluar dual tree Boruvka from mlpack
``plot_clustering.py`` - plots high-dimensional clusterings via PCA
``toy.py`` - perform toy examples
``tree_entropy.py`` - computes MST based entropy objective

