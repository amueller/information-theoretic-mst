import numpy as np

def make_circles(n_samples=100, shuffle=True, noise=None, random_state=None,
        factor=.8):
    """Make a large circle containing a smaller circle in 2di

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    shuffle: bool, optional (default=True)
        Whether to shuffle the samples.

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    factor : double < 1 (default=.8)
        Scale factor between inner and outer circle.
    """

    if factor > 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")

    n_samples_out = int(n_samples / 2)
    n_samples_in = n_samples - n_samples_out


    # so as not to have the first point = last point, we add one and then
    # remove it.
    n_samples_out, n_samples_in = n_samples_out + 1, n_samples_in + 1
    outer_circ_x = np.cos(np.linspace(0, 2 * np.pi, n_samples_out)[:-1])
    outer_circ_y = np.sin(np.linspace(0, 2 * np.pi, n_samples_out)[:-1])
    inner_circ_x = np.cos(np.linspace(0, 2 * np.pi, n_samples_in)[:-1]) * factor
    inner_circ_y = np.sin(np.linspace(0, 2 * np.pi, n_samples_in)[:-1]) * factor

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),\
           np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples_out - 1), np.ones(n_samples_in - 1)])

    if not noise is None:
        X += np.random.normal(scale=noise, size=X.shape)

    return X, y.astype(np.int)

