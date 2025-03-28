import numpy as np
from scipy.spatial.distance import cdist


def rbf_kernel(inputs1, inputs2, lengthscale=1, signal_variance=1, noise_variance=0):
    """
    Stationary (scalar) gaussian kernel.

    Parameters
    ----------
    inputs1 : ndarray, shape (n_samples1, n_features)
        First input data.
    inputs2 : ndarray, shape (n_samples2, n_features)
        Second input data.
    lengthscale : float, optional
        Lengthscale parameter (default is 1).
    signal_variance : float, optional
        Signal variance (default is 1).
    noise_variance : float, optional
        Noise variance (default is 0).

    Returns
    -------
    kernel_matrix : ndarray, shape (n_samples1, n_samples2)
        Computed kernel matrix.
    """
    kernel_matrix = signal_variance**2 * np.exp(-0.5 * cdist(inputs1 / lengthscale, inputs2 / lengthscale, 'sqeuclidean'))

    if inputs1.shape[0] == inputs2.shape[0]:
        noise_variance = max(noise_variance, 1e-3)
        kernel_matrix += noise_variance**2 * np.eye(inputs1.shape[0])

    return kernel_matrix