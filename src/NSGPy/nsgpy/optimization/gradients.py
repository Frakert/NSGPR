import numpy as np

from ..kernels.stationary import rbf_kernel
from ..kernels.nonstationary import ns_rbf_kernel
from ..utils.linalg import solve_cholesky, solve_triangular


def lengthscale_gradient(gp):
    """
    Calculates the derivative of the lengthscale latent function with respect to 
    marginal log-likelihood.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    whitened_lengthscale_derivative : ndarray, shape (n,)
        Computed derivative.
    """
    if 'ab' in gp.nonstationary_functions:
        gp.kernel_lengthscale = rbf_kernel(
            gp.normalized_inputs, 
            gp.normalized_inputs, 
            gp.beta_lengthscale, 
            gp.alpha_lengthscale, 
            gp.tolerance
        )
    
    n_samples = gp.n_samples

    woodbury_matrix = gp.get_woodbury_matrix
    
    if 'l' not in gp.nonstationary_functions:
        # Scalar case
        kernel_without_noise = gp.get_ns_rbf_kernel
                
        lengthscale_mean_sqrt = np.exp((-2) * gp.mean_log_lengthscale)
        kernel_derivative = lengthscale_mean_sqrt * (gp.distance_matrix * kernel_without_noise)
        
        # Calculate the derivative
        lengthscale_derivative = 0.5 * np.einsum('ij,ji->i', woodbury_matrix, kernel_derivative) - solve_cholesky(gp.cholesky_lengthscale, (gp.log_lengthscale - gp.mean_log_lengthscale))
        
        # Distribute the total derivative to all points
        lengthscale_derivative = np.ones(n_samples) * lengthscale_derivative.sum()
        whitened_lengthscale_derivative = solve_triangular(gp.cholesky_signal_variance, lengthscale_derivative)[0]
        
    else:
        # Non-scalar case
        lengthscale = gp.lengthscale
        lengthscale2 = lengthscale**2
        lengthscale4 = lengthscale2**2
        lengthscale_signal = np.sqrt(lengthscale) * gp.signal_variance
        
        # Calculate intermediate matrices
        squared_lengthscale_sum = np.add.outer(lengthscale2, lengthscale2)

        kernel_derivative = np.einsum("ij,i->ij", gp.distance_matrix, 4 * lengthscale2) - np.subtract.outer(lengthscale4, lengthscale4)
        kernel_derivative *= np.outer(lengthscale_signal / (2**0.5), lengthscale_signal) 
        kernel_derivative *= np.exp(-gp.distance_matrix / squared_lengthscale_sum)
        kernel_derivative /= squared_lengthscale_sum**2 * np.sqrt(squared_lengthscale_sum)
        
        # Initialize lengthscale_derivative as zeros array of shape (n, 1)
        # lengthscale_derivative = np.zeros(n_samples)
        
        # for i in range(n_samples):
        #     rows = np.concatenate([np.arange(n_samples), np.ones(n_samples, dtype=int) * i])
        #     cols = np.concatenate([np.ones(n_samples, dtype=int) * i, np.arange(n_samples)])
        #     values = np.concatenate([kernel_derivative[:, i], kernel_derivative[i, :]])
            
        #     sparse_matrix = coo_matrix((values, (rows, cols)), shape=(n_samples, n_samples)).toarray()
        #     result = woodbury_matrix * sparse_matrix
        #     lengthscale_derivative[i] = 0.5 * result.sum()

        # faster than previous loop
        col_contrib = np.einsum("ij,ij->i", kernel_derivative, woodbury_matrix)
        row_contrib = np.einsum("ji,ji->i", kernel_derivative, woodbury_matrix)
        
        # Compute final result
        lengthscale_derivative = 0.5 * (col_contrib + row_contrib)

        lengthscale_derivative -= solve_cholesky(gp.cholesky_lengthscale, (gp.log_lengthscale - gp.mean_log_lengthscale))
        whitened_lengthscale_derivative = gp.cholesky_signal_variance.T @ lengthscale_derivative
    
    return whitened_lengthscale_derivative


def signal_variance_gradient(gp):
    """
    Derivative of the sigma latent function with respect to MLL.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    dwl_s : ndarray, shape (n,)
        Computed derivative.
    """
    n_samples = gp.n_samples
    
    if 'ab' in gp.nonstationary_functions:
        gp.kernel_signal_variance = rbf_kernel(
            gp.normalized_inputs, 
            gp.normalized_inputs, 
            gp.beta_signal_variance, 
            gp.alpha_signal_variance, 
            gp.tolerance
        )

    kernel_without_noise = gp.get_ns_rbf_kernel
    
    woodbury_matrix = gp.get_woodbury_matrix
    
    signal_variance_derivative = 2 * np.einsum('ij,ji->i', woodbury_matrix, kernel_without_noise) - solve_cholesky(gp.cholesky_signal_variance, (gp.log_signal_variance - gp.mean_log_signal_variance))
    
    if 's' in gp.nonstationary_functions:
        # Non-scalar case 
        whitened_signal_variance_derivative = gp.cholesky_signal_variance.T @ signal_variance_derivative
    else:
        # Scalar case 
        signal_variance_derivative = np.ones(n_samples) * signal_variance_derivative.sum()
        whitened_signal_variance_derivative = solve_triangular(gp.cholesky_signal_variance, signal_variance_derivative)[0]
    
    return whitened_signal_variance_derivative


def noise_variance_gradient(gp):
    """
    Derivative of the noise latent function with respect to MLL.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    whitened_noise_variance_derivative : ndarray, shape (n,)
        Computed derivative.
    """
    n_samples = gp.n_samples
    
    if 'ab' in gp.nonstationary_functions:
        gp.kernel_noise_variance = rbf_kernel(
            gp.normalized_inputs, 
            gp.normalized_inputs, 
            gp.beta_noise_variance, 
            gp.alpha_noise_variance, 
            gp.tolerance
            )
    
    woodbury_matrix = gp.get_woodbury_matrix
    
    # Ω = diag(ω^2)
    omega = np.exp(2 * gp.log_noise_variance)

    # ∂log L/∂ω = diag(woodbury_matrix * Ω) - K_ω_inv * (ω−µ_ω)
    noise_variance_derivative = np.diag(woodbury_matrix) * omega - solve_cholesky(gp.cholesky_noise_variance, (gp.log_noise_variance - gp.mean_log_noise_variance))
    
    if 'o' in gp.nonstationary_functions:
        # Non-scalar case 
        whitened_noise_variance_derivative = gp.cholesky_noise_variance.T @ noise_variance_derivative
    else:
        # Scalar case 
        noise_variance_derivative = np.ones(n_samples) * noise_variance_derivative.sum()
        whitened_noise_variance_derivative = solve_triangular(gp.cholesky_noise_variance, noise_variance_derivative)[0]
    
    return whitened_noise_variance_derivative


def alphas_gradient(gp): # Probably not needed but included because of completeness
    """
    Derivative of the alphas over MLL.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    dl, ds, do : float
        Computed derivatives.
    """
    Kl = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_lengthscale, gp.alpha_lengthscale, gp.tolerance)
    Ks = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_signal_variance, gp.alpha_signal_variance, gp.tolerance)
    Ko = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_noise_variance, gp.alpha_noise_variance, gp.tolerance)
    
    al = np.linalg.solve(Kl, gp.log_lengthscale - gp.mean_log_lengthscale)
    as_ = np.linalg.solve(Ks, gp.log_signal_variance - gp.mean_log_signal_variance)
    ao = np.linalg.solve(Ko, gp.log_noise_variance - gp.mean_log_noise_variance)
    
    dKl = 2 * gp.alpha_lengthscale**(-1) * Kl
    dKs = 2 * gp.alpha_signal_variance**(-1) * Ks
    dKo = 2 * gp.alpha_noise_variance**(-1) * Ko
    
    dl = 0.5 * np.sum(np.diag((np.outer(al, al) - np.linalg.inv(Kl)) @ dKl))
    ds = 0.5 * np.sum(np.diag((np.outer(as_, as_) - np.linalg.inv(Ks)) @ dKs))
    do = 0.5 * np.sum(np.diag((np.outer(ao, ao) - np.linalg.inv(Ko)) @ dKo))

    return dl, ds, do


def betas_gradient(gp): # Probably not needed but included because of completeness
    """
    Derivative of the betas over MLL.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    dl, ds, do : float
        Computed derivatives.
    """
    Kl = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_lengthscale, gp.alpha_lengthscale, gp.tolerance)
    Ks = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_signal_variance, gp.alpha_signal_variance, gp.tolerance)
    Ko = rbf_kernel(gp.normalized_inputs, gp.normalized_inputs, gp.beta_noise_variance, gp.alpha_noise_variance, gp.tolerance)
    
    al = np.linalg.solve(Kl, gp.log_lengthscale - gp.mean_log_lengthscale)
    as_ = np.linalg.solve(Ks, gp.log_signal_variance - gp.mean_log_signal_variance)
    ao = np.linalg.solve(Ko, gp.log_noise_variance - gp.mean_log_noise_variance)
    
    dKl = gp.beta_lengthscale**(-3) * gp.distance_matrix * Kl
    dKs = gp.beta_signal_variance**(-3) * gp.distance_matrix * Ks
    dKo = gp.beta_noise_variance**(-3) * gp.distance_matrix * Ko
    
    dl = 0.5 * np.sum(np.diag((np.outer(al, al) - np.linalg.inv(Kl)) @ dKl))
    ds = 0.5 * np.sum(np.diag((np.outer(as_, as_) - np.linalg.inv(Ks)) @ dKs))
    do = 0.5 * np.sum(np.diag((np.outer(ao, ao) - np.linalg.inv(Ko)) @ dKo))
    
    return dl, ds, do