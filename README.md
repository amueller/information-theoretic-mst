information-theoretic-mst
=========================

Information Theoretic Clustering using Minimum Spanning TreesA

This is the code accompanying "Information Theoretic Clustering using Minimum Spanning Trees"
by Andreas C. Mueller, Sebastian Nowozin, and Christoph H. Lampert

Please cite the paper if you use this code.


FEATURES
--------
Implements ITM as describe in the paper "Information Theoretic Clustering using Minimum Spanning Trees".
Also implements MeanNN (See Faivishevsky, L. and Goldberger, J. [A nonparametric information theoretic clustering algorithm](http://eprints.pascal-network.org/archive/00007747/) )
and single-link agglomerative clustering for comparison.


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


LICENSE
-------
Copyright (c) 2012, Andreas Mueller 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
