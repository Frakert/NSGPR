import numpy as np
from scipy.linalg import eigvalsh

from ..kernels.stationary import rbf_kernel
from ..kernels.nonstationary import ns_rbf_kernel
from ..utils.linalg import cholesky_with_jitter, solve_cholesky, cholesky


def log_multivariate_normal_pdf(observations, mean_vector, covariance_matrix):
    """
    Calculates the log-likelihood for observations from a multivariate normal distribution.
    
    Parameters
    ----------
    observations : ndarray, shape (n_dimensions,)
        Observation vector.
    mean_vector : ndarray, shape (n_dimensions,)
        Mean vector of the distribution.
    covariance_matrix : ndarray, shape (n_dimensions, n_dimensions)
        covariance matrix/kernel of the distribution.
    
    Returns
    -------
    log_likelihood : float
        Total log-likelihood value.
    """
    cholesky_observations = cholesky_with_jitter(covariance_matrix)
    
    n_dimensions = len(observations)
    constant_term = -0.5 * n_dimensions * np.log(2 * np.pi)
    
    data_fit_term = -0.5 * np.dot((observations - mean_vector).T, solve_cholesky(cholesky_observations, (observations - mean_vector)))
    
    # calculate_log_determinant = lambda matrix: 2 * np.log(np.diag(cholesky_observations)).sum()
    
    # try:
    normalization_term = constant_term - np.log(np.diag(cholesky_observations)).sum()
    # except np.linalg.LinAlgError: # not gotten this warning yet
    #     print("Warning: Cholesky decomposition failed, increasing diagonal to fix.")
    #     min_eigenvalue = abs(min(np.linalg.eigvals(covariance_matrix)))
    #     regularized_matrix = covariance_matrix + (min_eigenvalue * 1.01) * np.eye(covariance_matrix.shape[0])
    #     normalization_term = constant_term - 0.5 * calculate_log_determinant(regularized_matrix)
    
    log_likelihood = data_fit_term + normalization_term

    return log_likelihood


def nsgpmll(gp):
    """
    Computes the non-stationary Gaussian process marginal log-likelihood.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    total_log_likelihood : float
        Total log-likelihood value.
    """
    if any(func in gp.nonstationary_functions for func in ['a', 'b']):
        gp.kernel_lengthscale = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_lengthscale, gp.alpha_lengthscale, gp.tolerance)
        gp.kernel_signal_variance = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_signal_variance, gp.alpha_signal_variance, gp.tolerance)
        gp.kernel_noise_variance = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_noise_variance, gp.alpha_noise_variance, gp.tolerance)
    
    observation_covariance = gp.get_ns_rbf_kernel_with_noise

    zero_mean = np.zeros((gp.n_samples, gp.n_targets))
    
    # Check if non-SPD or low condition number

    L, info = cholesky(observation_covariance)
    if info != 0:
        return -np.inf, None, None, None, None
    
    # good approximation (way faster than caclulating the eigen values or SVD)
    condition_number = (np.max(np.diag(L)) / np.min(np.diag(L))) ** 2
    if condition_number < 1e-15:
        return -np.inf, None, None, None, None
    
    # Compute log-likelihood terms
    observations_log_likelihood = log_multivariate_normal_pdf(
        gp.normalized_outputs, 
        zero_mean, 
        observation_covariance
    )
    
    lengthscale_log_likelihood = log_multivariate_normal_pdf(
        gp.log_lengthscale, 
        gp.mean_log_lengthscale, 
        gp.kernel_lengthscale 
    ) # if len(gp.lengthscale) > 1 else logmvnpdf(gp.lengthscale * np.ones(n), muell, gp.kernel_lengthscale)
    # commented previous part out because muell is not defined and function gets never used 
    # this is also the case for the MATLAB code

    signal_variance_log_likelihood = log_multivariate_normal_pdf(
        gp.log_signal_variance, 
        gp.mean_log_signal_variance, 
        gp.kernel_signal_variance
    ) # if len(gp.signal_variance) > 1 else logmvnpdf(gp.signal_variance * np.ones(n), musigma, gp.kernel_signal_variance)

    noise_variance_log_likelihood = log_multivariate_normal_pdf(
        gp.log_noise_variance, 
        gp.mean_log_noise_variance, 
        gp.kernel_noise_variance
    ) # if len(gp.noise_variance) > 1 else logmvnpdf(gp.noise_variance * np.ones(n), muomega, gp.kernel_noise_variance)
    
    total_log_likelihood = observations_log_likelihood + lengthscale_log_likelihood + signal_variance_log_likelihood + noise_variance_log_likelihood

    return total_log_likelihood, observations_log_likelihood, lengthscale_log_likelihood, signal_variance_log_likelihood, noise_variance_log_likelihood