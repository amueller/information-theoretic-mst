import numpy as np

from sklearn.neighbors import NearestNeighbors

def estimate_dimension(X, n_neighbors='auto', neighbors_estimator=None):
    """Estimate intrinsic dimensionality.

    Based on "Manifold-Adaptive Dimension Estimation"
    Farahmand, Szepavari, Audibert ICML 2007.

    Parameters
    ----------
    X : nd-array, shape (n_samples, n_features)
        Input data.

    n_neighbors : int or auto, default='auto'
        Number of neighbors used for estimate.
        'auto' means ``np.floor(2 * np.log(n_samples))``.

    neighbors_estimator : NearestNeighbors object or None, default=None
        A pre-fitted neighbors object to speed up calculations.
    """
    if n_neighbors == 'auto':
        n_neighbors = np.floor(2 * np.log(X.shape[0])).astype("int")

    if neighbors_estimator is None:
        neighbors_estimator = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_estimator.fit(X)
    full_dist = neighbors_estimator.kneighbors(X, n_neighbors=n_neighbors)[0][:, -1]
    half_dist = neighbors_estimator.kneighbors(X, n_neighbors=n_neighbors // 2)[0][:, -1]
    est = np.log(2) / np.log(full_dist / half_dist)
    est = np.minimum(est, X.shape[1])
    return np.round(np.mean(est))
