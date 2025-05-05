import torch

from ..kernels.stationary import rbf_kernel
from ..kernels.nonstationary import ns_rbf_kernel
from ..utils.numerics import condition_number_jitter
from ..utils.preprocessing import denormalise


def gp_posterior(
    x: torch.Tensor, 
    y: torch.Tensor, 
    xt: torch.Tensor, 
    mean_y: torch.Tensor, 
    lengthscale: float, 
    signal_variance: float, 
    noise_variance: float
) -> torch.Tensor:
    """
    Compute the non-stationary Gaussian Process posterior mean and covariance.
    
    Parameters
    ----------
    x : torch.Tensor, shape (n_train_samples, n_features)
        Training input data.
    y : torch.Tensor, shape (n_targets, n_train_samples)
        Training output data.
    xt : torch.Tensor, shape (n_test_samples, n_features)
        Test input data.
    mean_y : torch.Tensor, shape (n_targets)
        Mean of y.
    lengthscale : float
        Lengthscale.
    signal_variance : float
        Signal variance.
    noise_variance : float
        Noise variance.
    
    Returns
    -------
    posterior_mean : torch.Tensor, shape (n_targets, n_test_samples)
        Posterior mean.
    posterior_cov : torch.Tensor or None, shape (n_targets, n_test_samples, n_test_samples)
        Posterior covariance.
    
    Notes
    -----
    Given training inputs X, training outputs y, and test inputs X':
    
    1. Posterior Mean: 
       μ' = m' + K(X', X) K(X, X)⁻¹ (y - m)
    
    2. Posterior Covariance:
       Σ' = K(X', X') - K(X', X) K(X, X)⁻¹ K(X, X')
    
    Where:
    - m' is the prior mean at test points
    - m is the prior mean at training points
    - K(X, X) is the kernel matrix for training points
    - K(X', X) is the cross-kernel matrix between test and training points
    - σ² is the noise variance
    """
    # Compute kernel matrices
    K_train_train = rbf_kernel(x, x, lengthscale, signal_variance, noise_variance)
    K_train_test = rbf_kernel(x, xt, lengthscale, signal_variance, 0)
    # K_test_test = rbf_kernel(xt, xt, lengthscale, signal_variance, 0)
    
    # Add adaptive jitter to diagnonal to improve numerical stability
    jitter = condition_number_jitter(K_train_train, cond_thresh=1e4)
    idx = torch.arange(K_train_train.shape[1], device=K_train_train.device)
    K_train_train[idx, idx] += jitter * torch.ones(K_train_train.shape[0], device=K_train_train.device)

    # Cholesky decomposition for stable matrix inversion
    woodbury_chol = torch.linalg.cholesky(K_train_train)
    
    # Solve for alpha: K(X, X)⁻¹ (y - m)
    woodbury_vector = torch.cholesky_solve(
        (y - mean_y[:, None]).unsqueeze(2), 
        woodbury_chol, 
        upper=False
    ).squeeze(-1)

    # Compute posterior mean: m' = alpha K(X, X') + m
    posterior_mean = woodbury_vector @ K_train_test + mean_y[:, None]
    
    # Currently not computing posterior covariance
    posterior_cov = None

    return posterior_mean, posterior_cov


def nsgp_posterior(
    gp: object, 
    xt: torch.Tensor=None, 
    full_cov: bool=False
) -> torch.Tensor:
    """
    Compute the posterior of a non-stationary GP at test points 'xt'.
    
    Parameters
    ----------
    gp : object
        GP model instance.
    xt : torch.Tensor, shape (n_test_samples, n_features)
        Test input points (optional, default is training points).
    full_cov : bool
        Compute the full covariance matrix if True 
        else compute diagonal covariance matrix (optional, default is False).

    Returns
    -------
    posterior_mean : torch.Tensor, shape (n_targets, n_test_samples)
        Posterior mean of function f.
    fstd : torch.Tensor, shape (n_test_samples)
        Posterior standard deviation of function f.
    posterior_lengthscale : torch.Tensor, shape (n_test_samples)
        Posterior mean of lengthscale.
    posterior_signal_variance : torch.Tensor, shape (n_test_samples)
        Posterior mean of signal variance.
    posterior_noise_variance : torch.Tensor, shape (n_test_samples)
        Posterior mean of noise variance.

    Notes
    -----
    Given training inputs X, training outputs y, and test inputs X':
    
    1. Posterior Mean: 
       μ' = m' + K(X', X) K(X, X)⁻¹ (y - m)
    
    2. Posterior Covariance:
       Σ' = K(X', X') - K(X', X) K(X, X)⁻¹ K(X, X')
    
    Where:
    - m' is the prior mean at test points
    - m is the prior mean at training points
    - K(X, X) is the kernel matrix for training points
    - K(X', X) is the cross-kernel matrix between test and training points
    - σ² is the noise variance
    """
    # Use training inputs if no test points provided
    if xt is None:
        xt = gp.normalized_inputs

    # Compute posterior distributions for kernel parameters
    log_posterior_lengthscale, _ = gp_posterior(
        gp.normalized_inputs, 
        gp.log_lengthscale, 
        xt, 
        gp.mean_log_lengthscale, 
        gp.beta_lengthscale, 
        gp.alpha_lengthscale, 
        gp.tolerance
    )

    log_posterior_signal_variance, _ = gp_posterior(
        gp.normalized_inputs, 
        gp.log_signal_variance, 
        xt, 
        gp.mean_log_signal_variance, 
        gp.beta_signal_variance, 
        gp.alpha_signal_variance, 
        gp.tolerance
    )

    log_posterior_noise_variance, _ = gp_posterior(
        gp.normalized_inputs, 
        gp.log_noise_variance, 
        xt, 
        gp.mean_log_noise_variance, 
        gp.beta_noise_variance, 
        gp.alpha_noise_variance, 
        gp.tolerance
    )

    # Convert from log-space to original scale
    posterior_lengthscale = torch.exp(log_posterior_lengthscale)
    posterior_signal_variance = torch.exp(log_posterior_signal_variance)
    posterior_noise_variance = torch.exp(log_posterior_noise_variance)

    # Compute kernel matrices
    K_train_train = gp.get_ns_rbf_kernel_with_noise 

    K_train_test = ns_rbf_kernel( 
        gp.normalized_inputs, 
        xt, 
        gp.lengthscale, 
        posterior_lengthscale, 
        gp.signal_variance, 
        posterior_signal_variance, 
        0
    ) 

    K_test_test = ns_rbf_kernel( 
        xt, 
        xt, 
        posterior_lengthscale, 
        posterior_lengthscale, 
        posterior_signal_variance, 
        posterior_signal_variance, 
        0
    ) 

    # Add adaptive jitter to diagnonal to improve numerical stability
    jitter = condition_number_jitter(K_train_train, cond_thresh=1e4)        
    idx = torch.arange(K_train_train.shape[1], device=K_train_train.device)
    K_train_train[:, idx, idx] += jitter.repeat(K_train_train.shape[-1],1).T

    if gp.verbose_output:
        print(f"K_train_train nsgp jitter: {jitter}")

    # Cholesky decomposition for stable matrix inversion
    woodbury_chol = torch.linalg.cholesky(K_train_train)
    
    # Solve for alpha: K(X, X)⁻¹ (y - m)
    woodbury_vector = torch.cholesky_solve(
        gp.normalized_outputs.T.unsqueeze(2), 
        woodbury_chol, 
        upper=False
    ).squeeze(-1)

    # Compute posterior mean: m' = alpha K(X, X') + m
    posterior_mean = (woodbury_vector.unsqueeze(1) @ K_train_test).squeeze(1)

    if full_cov:
        # Compute L⁻¹ K(X, X') for covariance calculation
        scaled_cross_kernel = torch.linalg.solve_triangular(woodbury_chol, K_train_test, upper=False)

        # Compute posterior covariance matrix: K(X', X') - (K(X, X')ᵀ K(X, X)⁻¹ K(X, X'))
        posterior_cov = K_test_test - (scaled_cross_kernel.transpose(-2, -1) @ scaled_cross_kernel)

    else:
        # Extract the diagonal elements of K_test_test
        K_test_test_diag = torch.diagonal(K_test_test, dim1=-2, dim2=-1)

        # Compute L⁻¹ K(X, X') for diagonal variance computation
        scaled_cross_kernel = torch.linalg.solve_triangular(woodbury_chol, K_train_test, upper=False)

        # Compute posterior variance (diagonal only): diag(K(X', X')) - diag((K(X, X')ᵀ K(X, X)⁻¹ K(X, X')))
        posterior_cov = K_test_test_diag - torch.sum(torch.square(scaled_cross_kernel), dim=-2)
        
    # Ensure that posterior_cov is positive
    # posterior_cov = np.clip(posterior_cov, 1e-15, np.inf)

    # Compute standard deviation, ensuring numerical stability
    if torch.all(posterior_cov >= 0): 
        fstd = torch.sqrt(posterior_cov) 
    else: 
        fstd = posterior_cov
        print(f"Warning: Negative posterior covariance: {posterior_cov.min()}.")

    # Denormalize results back to original scale
    _, _, posterior_mean, fstd, posterior_lengthscale, \
    posterior_signal_variance, posterior_noise_variance = denormalise(
        gp, 
        posterior_mean, 
        fstd, 
        posterior_lengthscale, 
        posterior_signal_variance, 
        posterior_noise_variance
    )

    return posterior_mean, fstd, posterior_lengthscale, posterior_signal_variance, posterior_noise_variance