import numpy as np
from scipy.spatial.distance import cdist


def ns_rbf_kernel(inputs1, inputs2, lengthscale1, lengthscale2, 
                                 signal_variance1, signal_variance2, noise_variance):
    """
    Non-stationary (scalar) Gaussian kernel.
    
    Parameters
    ----------
    inputs1 : ndarray, shape (n_samples1, n_features)
        First input data.
    inputs2 : ndarray, shape (n_samples2, n_features)
        Second input data.
    lengthscale1 : ndarray, shape (n_samples1,)
        Length scale parameter for inputs1.
    lengthscale2 : ndarray, shape (n_samples2,)
        Length scale parameter for inputs2.
    signal_variance1 : ndarray, shape (n_samples1,)
        Signal variance for inputs1.
    signal_variance2 : ndarray, shape (n_samples2,)
        Signal variance for inputs2.
    noise_variance : ndarray, shape (n_samples1,)
        Noise variance.
    
    Returns
    -------
    kernel_matrix : ndarray, shape (n_samples1, n_samples2)
        Computed kernel matrix.

    Notes
    -----
    The mathematical formula for the kernel is:

    K_f(x, x′) = σ(x) σ(x′) √(2ℓ(x)ℓ(x′)/(ℓ(x)² + ℓ(x′)²)) exp(-(‖x - x′‖²)/(ℓ(x)² + ℓ(x′)²))
    
    Where:
    - x and x′ are points
    - ‖x - x′‖² is the squared Euclidean distance between the points
    - σ(x) is a function that scales the kernel at point x
    - ℓ(x) is the length scale function at point x
    """
    # Compute kernel matrix components
    lengthscale_sum = np.add.outer(lengthscale1**2, lengthscale2**2)
    length_signal_1 = np.sqrt(lengthscale1 * 2) * signal_variance1
    length_signal_2 = np.sqrt(lengthscale2) * signal_variance2
    
    # Calculate the kernel matrix
    kernel_matrix = np.outer(length_signal_1, length_signal_2) 
    kernel_matrix /= np.sqrt(lengthscale_sum) 
    kernel_matrix *= np.exp(-cdist(inputs1, inputs2, 'sqeuclidean') / lengthscale_sum)
    
    # Add noise to diagonal
    if inputs1.shape[0] == inputs2.shape[0]:
        if type(noise_variance) == int:
            noise_variance = max(noise_variance, 1e-3) * np.ones(kernel_matrix.shape[0])
        elif noise_variance.size == 1:  # If single float (inside np.array)
            noise_variance = max(noise_variance.item(), 1e-3) * np.ones(kernel_matrix.shape[0])
        else:  # If it's a vector, add it to the diagonal
            noise_variance = np.maximum(noise_variance, 1e-3)
        np.fill_diagonal(kernel_matrix, kernel_matrix.diagonal() + noise_variance**2)

    return kernel_matrix