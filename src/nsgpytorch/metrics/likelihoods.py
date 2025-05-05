import torch
import numpy as np


def log_multivariate_normal_pdf(
    observations: torch.Tensor, 
    mean_vector: torch.Tensor, 
    covariance_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the log-likelihood for observations from a multivariate normal distribution.

    Parameters
    ----------
    observations : torch.Tensor, shape (batch_size, n_samples)
        Observation vector x.
    mean_vector : torch.Tensor, shape (batch_size)
        Mean vector μ of the distribution.
    covariance_matrix : torch.Tensor, shape (batch_size, n_samples, n_samples) or (n_samples, n_samples)
        Covariance matrix Σ of the distribution.
    
    Returns
    -------
    log_likelihood : torch.Tensor, shape (batch_size)
        Log-likelihood log(p(x)) for each batch of observations.

    Notes
    -----
    The log-likelihood is computed using the probability density function (PDF) of 
    a multivariate normal distribution:

    log(p(x)) = -0.5 * [ d * log(2π) + log(|Σ|) + (x - μ)ᵀ Σ⁻¹ (x - μ) ]
    
    Where:
    - x : observation vector
    - μ : mean vector
    - Σ : covariance matrix
    - d : dimensionality of the vector
    
    Mathematical breakdown:
    1. Constant term:     -0.5 * d * log(2π)
    2. Determinant term:  -0.5 * log(|Σ|)
    3. Quadratic term:    -0.5 * (x - μ)ᵀ Σ⁻¹ (x - μ)
    """
    cholesky_observations = torch.linalg.cholesky(covariance_matrix)

    # Compute difference vector (x - μ)
    difference_vector = observations - mean_vector[:, None]

    # Solve linear system: Σ⁻¹ (x - μ)
    sol = torch.cholesky_solve(
        difference_vector.unsqueeze(2), 
        cholesky_observations, 
        upper=False
    ).squeeze(-1)

    # Compute quadratic term: -0.5 * (x - μ)ᵀ Σ⁻¹ (x - μ)
    quadratic_term = -0.5 * (difference_vector * sol).sum(dim=1)
    
    dimentionality = observations.shape[1]

    # Constant term: -0.5 * d * log(2π)
    constant_term = -0.5 * dimentionality * np.log(2 * np.pi) * torch.ones((observations.shape[0]),device=observations.device)

    # Log determinant term: -0.5 * log(|Σ|)
    log_determinant_term = -torch.sum(
        torch.log(torch.diagonal(cholesky_observations, dim1=-2, dim2=-1)), 
        dim=-1
    )

    # Combine all terms to get log-likelihood
    log_likelihood = quadratic_term + constant_term + log_determinant_term

    return log_likelihood


def nsgpmll(gp: object) -> torch.Tensor:
    """
    Computes the Non-Stationary Gaussian Process Marginal Log-Likelihood (MLL).
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    total_log_likelihood : torch.Tensor, shape (batch_size)
        Log-likelihood values for each batch, accounting for:
        - Observation likelihood
        - Hyperparameter priors

    Notes
    -----
    The marginal log-likelihood is defined as:
    
    log p(y | X, θ) = log ∫ p(y | f, θ) p(f | θ) df
    
    Where:
    - y : Observed outputs
    - X : Input features
    - f : Latent function values
    - θ : Hyperparameters (lengthscale, signal variance, noise variance)
    
    This implementation computes the log-likelihood as a sum of:
    1. Observations log-likelihood
    2. Lengthscale prior log-likelihood
    3. Signal variance prior log-likelihood
    4. Noise variance prior log-likelihood
    
    Mathematical formulation:
    MLL = log p(y | X, θ) ≈ 
        log p(y | f, θ) + 
        log p(log(lengthscale) | θ) + 
        log p(log(signal_variance) | θ) + 
        log p(log(noise_variance) | θ)
    """
    
    observation_covariance = gp.get_ns_rbf_kernel_with_noise 
    zero_mean = torch.zeros((gp.batch_size), device=gp.device)
    
    # Check if non-SPD or low condition number
    L, info = torch.linalg.cholesky_ex(observation_covariance)

    # Mask failed decompositions
    failed_mask = info > 0

    # Handle condition number (only for successful decompositions)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # Get diagonals of L for each batch
    condition_number = (torch.max(diag_L, dim=-1).values / torch.min(diag_L, dim=-1).values) ** 2
    
    # Mark bad conditions as failures
    failed_mask |= condition_number < 1e-15

    # Initialize results with -inf where decomposition failed
    total_log_likelihood = torch.full((observation_covariance.shape[0],), -torch.inf, device=observation_covariance.device)
    
    # Compute log-likelihoods only for successful cases
    valid_batches = ~failed_mask

    if valid_batches.any():
        observations_log_likelihood = log_multivariate_normal_pdf( 
            gp.batch_outputs.T[valid_batches],
            zero_mean[valid_batches],
            observation_covariance[valid_batches]
        )
        
        lengthscale_log_likelihood = log_multivariate_normal_pdf(
            gp.log_lengthscale[valid_batches],
            gp.mean_log_lengthscale[valid_batches],
            gp.kernel_lengthscale
        )

        signal_variance_log_likelihood = log_multivariate_normal_pdf(
            gp.log_signal_variance[valid_batches],
            gp.mean_log_signal_variance[valid_batches],
            gp.kernel_signal_variance
        )

        noise_variance_log_likelihood = log_multivariate_normal_pdf(
            gp.log_noise_variance[valid_batches],
            gp.mean_log_noise_variance[valid_batches],
            gp.kernel_noise_variance
        )

        total_log_likelihood[valid_batches] = (
            observations_log_likelihood + 
            lengthscale_log_likelihood + 
            signal_variance_log_likelihood + 
            noise_variance_log_likelihood
        )

    return total_log_likelihood