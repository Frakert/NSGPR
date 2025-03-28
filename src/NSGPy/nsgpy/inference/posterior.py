import numpy as np

from ..kernels.stationary import rbf_kernel
from ..kernels.nonstationary import ns_rbf_kernel
from ..utils.numerics import condition_number_jitter
from ..utils.linalg import cholesky_with_jitter, solve_cholesky, solve_triangular, fast_tdot
from ..utils.preprocessing import denormalise


def gp_posterior(x, y, xt, mean_y, lengthscale, signal_variance, noise_variance):
    """
    Compute the non-stationary Gaussian Process posterior mean and covariance.
    
    Parameters
    ----------
    x : ndarray, shape (n, d)
        Training input data.
    y : ndarray, shape (n,)
        Training output data.
    xt : ndarray, shape (m, d)
        Test input data.
    mean_y : float
        Mean of y.
    lengthscale : float
        Lengthscale.
    signal_variance : float
        Signal variance.
    noise_variance : float
        Noise variance.
    
    Returns
    -------
    posterior_mean : ndarray, shape (m,)
        Posterior mean.
    posterior_cov : ndarray, shape (m, m)
        Posterior covariance.
    """
    K_train_train = rbf_kernel(x, x, lengthscale, signal_variance, noise_variance)
    K_train_test = rbf_kernel(x, xt, lengthscale, signal_variance, 0)
    # K_test_test = rbf_kernel(xt, xt, lengthscale, signal_variance, 0)
    
    # K_test_test += 1e-4 * np.eye(K_test_test.shape[0])
    # K_train_train += 1e-6 * np.eye(K_train_train.shape[0])

    # this seems to solve most problems with instability, the gp.tolerance is not enough
    jitter = condition_number_jitter(K_train_train, cond_thresh=1e4)
    # print(f"K_train_train gp jitter: {np.diag(jitter)[0]}")
    K_train_train += jitter  # 1e-6

    # cond_Ktt = np.linalg.cond(K_train_train)
    # print(f"Condition number of Ktt: {cond_Ktt}")

    # cond_Kts = np.linalg.cond(K_train_test)
    # print(f"Condition number of Kts: {cond_Kts}")

    # cond_Kss = np.linalg.cond(K_test_test)
    # print(f"Condition number of Kss: {cond_Kss}")

    # eigenvalues = np.linalg.eigvalsh(K_test_test)
    # print(f"Min eigenvalue of Kss: {eigenvalues.min()}")

    # print(f"Norm ratio: {np.linalg.norm(K_train_test) / np.linalg.norm(K_test_test)}")

    # Cholesky decomposition
    woodbury_chol = cholesky_with_jitter(K_train_train)
    
    # Solve for alpha = K_train_train⁻¹ * (y - mean_y)
    woodbury_vector = solve_cholesky(woodbury_chol, y - mean_y)
    
    # Compute mean prediction: mu_* = K(X_*, X) alpha
    posterior_mean = K_train_test.T @ woodbury_vector + mean_y
    
    # Solve for V = L⁻¹ K_train_test
    # V = solve_triangular(L, K_train_test, lower=True)
    
    # Compute covariance: K(X_*, X_*) - V^T V
    # posterior_cov = K_test_test - V.T @ V
    posterior_cov = None  # not needed for posterior
    
    # Ensure symmetry
    # posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    
    # print(f"Min value in fcov: {posterior_cov.min()}")

    return posterior_mean, posterior_cov


def nsgp_posterior(gp, xt=None):
    """
    Compute the posterior of a non-stationary GP at test points 'xt'.
    
    Parameters
    ----------
    gp : object
        GP model instance.
    xt : ndarray, shape (n,)
        Test input points (optional, default is training points).
    
    Returns
    -------
    posterior_mean : ndarray, shape (n, 1)
        Posterior mean of function f.
    fstd : ndarray, shape (n,)
        Posterior standard deviation of function f.
    posterior_lengthscale : ndarray, shape (n,)
        Posterior mean of lengthscale.
    posterior_signal_variance : ndarray, shape (n,)
        Posterior mean of signal variance.
    posterior_noise_variance : ndarray, shape (n,)
        Posterior mean of noise variance.
    """
    if xt is None:
        xt = gp.normalized_inputs

    # print("posterior lengthscale")
    log_posterior_lengthscale, _ = gp_posterior(
        gp.normalized_inputs, 
        gp.log_lengthscale, 
        xt, 
        gp.mean_log_lengthscale, 
        gp.beta_lengthscale, 
        gp.alpha_lengthscale, 
        gp.tolerance
    )
    # print("posterior signal variance")
    log_posterior_signal_variance, _ = gp_posterior(
        gp.normalized_inputs, 
        gp.log_signal_variance, 
        xt, 
        gp.mean_log_signal_variance, 
        gp.beta_signal_variance, 
        gp.alpha_signal_variance, 
        gp.tolerance
    )
    # print("posterior noise variance")
    log_posterior_noise_variance, _ = gp_posterior(
        gp.normalized_inputs, 
        gp.log_noise_variance, 
        xt, 
        gp.mean_log_noise_variance, 
        gp.beta_noise_variance, 
        gp.alpha_noise_variance, 
        gp.tolerance
    )

    posterior_lengthscale = np.exp(log_posterior_lengthscale)
    posterior_signal_variance = np.exp(log_posterior_signal_variance)
    posterior_noise_variance = np.exp(log_posterior_noise_variance)

    # Compute kernel matrices
    K_train_train = gp.get_ns_rbf_kernel_with_noise

    K_train_test = ns_rbf_kernel(
        gp.normalized_inputs, 
        xt, 
        gp.lengthscale, 
        posterior_lengthscale, 
        gp.signal_variance, 
        posterior_signal_variance, 
        0 # log 0
    ) 

    K_test_test = ns_rbf_kernel(
        xt, 
        xt, 
        posterior_lengthscale, 
        posterior_lengthscale, 
        posterior_signal_variance, 
        posterior_signal_variance, 
        0 # log 0 # problem, Kss is not PSD,
    ) 

    # print("Posterior of non-stationary")

    # K_test_test is only needed for its diagonal elements, no computations
    # jitter = condition_number_jitter(K_test_test, cond_thresh=1e6)
    # print(f"K_test_test jitter: {np.diag(jitter)[0]}")
    # K_test_test += jitter  # 1e-4

    jitter = condition_number_jitter(K_train_train, cond_thresh=1e4)

    if gp.verbose_output:
        print(f"K_train_train nsgp jitter: {np.diag(jitter)[0]}")
        
    K_train_train += jitter  # 1e-6

    # cond_K_train_test = np.linalg.cond(K_train_test)
    # if cond_K_train_test > 1e5:
    #     print(f"Warning: K_train_test is ill-conditioned (cond={cond_K_train_test:.2e})")

    # cond_K_train_train = np.linalg.cond(K_train_train)
    # if cond_K_train_train > 1e5:
    #     print(f"Warning: K_train_train is ill-conditioned (cond={cond_K_train_train:.2e})")

    # cond_K_test_test = np.linalg.cond(K_test_test)
    # if cond_K_test_test > 1e5:
    #     print(f"Warning: K_test_test is ill-conditioned (cond={cond_K_test_test:.2e})")

    # eigenvalues = np.linalg.eigvalsh(K_test_test)
    # print(f"Min eigenvalue of Kss: {eigenvalues.min()}")

    # print(f"Norm ratio: {np.linalg.norm(K_train_test) / np.linalg.norm(K_test_test)}")

    # Cholesky decomposition (K_chol i used when K_train_train is well conditioned and 
    # no noise is needed but it is called woodbury_chol when noise is added for stability.
    # K_chol is the exact cholesky factor)
    woodbury_chol = cholesky_with_jitter(K_train_train)
    
    # Solve for alpha = K_train_train⁻¹ * (y - mean_y) # gp.mean_function = 0 because of normalization? 
    # Same woodbury_vector as in Woodbury_matrix in gp
    woodbury_vector = solve_cholesky(woodbury_chol, gp.normalized_outputs) # - gp.mean_function)

    # Compute mean prediction: mu_* = K(X_*, X) alpha
    posterior_mean = K_train_test.T @ woodbury_vector # + gp.mean_function
    
    # Solve for V = L⁻¹ K_train_test
    # V = solve_triangular(L, K_train_test, lower=True)
    
    # Compute covariance: K(X_*, X_*) - V^T V
    # posterior_cov = K_test_test - V.T @ V

    # Ensure symmetry
    # posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)

    # Use method from scikit-learn (only diagonal posterior_cov):
    # This is nummerically more stable to compute
    # posterior_cov = np.diag(K_test_test).copy()
    # posterior_cov -= np.einsum("ij,ji->i", V.T, V)

    # Use method from GPy:
    # Look deeper into what is happening and try to explain
    # Compute the pseudo-inverse of K_train_train using a numerically stable method (Cholesky)
    # wi, woodbury_chol, LWi, W_logdet = pdinv(K_train_train)

    # Compute the full Woodbury inverse using the Cholesky factorization
    # dpotri computes the inverse of (woodbury_chol @ woodbury_chol.T), which approximates K_train_train⁻¹
    # woodbury_inv, _ = dpotri(woodbury_chol, lower=True)

    # Ensure the inverse matrix is symmetric by explicitly enforcing symmetry
    # woodbury_inv = np.tril(woodbury_inv) + np.tril(woodbury_inv,-1).T

    # Toggle for computing full covariance matrix or just the diagonal
    full_cov = False

    if full_cov:
        # Compute the full posterior covariance matrix
        if woodbury_chol.ndim == 2:  # Standard case (no missing data)
            # Solve the triangular system L * x = K_train_test to get tmp = L⁻¹ K_train_test
            tmp = solve_triangular(woodbury_chol, K_train_test)[0]

            # Compute posterior covariance using the Woodbury identity:
            # posterior_cov = K_test_test - (K_train_test.T @ K_train_train⁻¹ @ K_train_test)
            posterior_cov = K_test_test - fast_tdot(tmp.T)

        elif woodbury_chol.ndim == 3:  # Handles missing data (batch processing case) (probably not needed)
            # Initialize a 3D array to store covariance matrices for different missing data scenarios
            print("Missing data")
            posterior_cov = np.empty((K_test_test.shape[0], K_test_test.shape[1], woodbury_chol.shape[2]))
            for i in range(posterior_cov.shape[2]): # Iterate over different missing data cases
                tmp = solve_triangular(woodbury_chol[:, :, i], K_train_test)[0]
                posterior_cov[:, :, i] = (K_test_test - fast_tdot(tmp.T))

    else:
        # Extract the diagonal elements of K_test_test (needed when full_cov=False)
        K_test_test_diag = np.diag(K_test_test)

        if woodbury_chol.ndim == 2:  # Standard case
            # Solve L * x = K_train_test to get tmp = L⁻¹ K_train_test
            tmp = solve_triangular(woodbury_chol, K_train_test)[0]

            # Compute only the diagonal of the posterior covariance:
            # posterior_variance = diag(K_test_test) - sum of squared elements in tmp
            posterior_cov = (K_test_test_diag - np.square(tmp).sum(0))

        elif woodbury_chol.ndim == 3:  # Handles missing data (batch processing case) (probably not needed)
            # Initialize an array to store diagonal variances for missing data scenarios
            print("Missing data")
            posterior_cov = np.empty((K_test_test_diag.shape[0], woodbury_chol.shape[2]))
            for i in range(posterior_cov.shape[1]): # Iterate over different missing data cases
                tmp = solve_triangular(woodbury_chol[:, :, i], K_train_test)[0]
                posterior_cov[:, i] = (K_test_test_diag - np.square(tmp).sum(0))
        
        # Ensure that posterior_cov is positive
        # posterior_cov = np.clip(posterior_cov, 1e-15, np.inf)

    # Diagonalize posterior_cov if full_cov is calculated
    # posterior_cov = np.diag(posterior_cov)

    if np.all(posterior_cov >= 0): 
        fstd = np.sqrt(posterior_cov) 
    else: 
        fstd = posterior_cov
        print(f"Warning: Negative posterior covariance: {posterior_cov.min()}.")

    _, _, posterior_mean, fstd, posterior_lengthscale, posterior_signal_variance, posterior_noise_variance = denormalise(gp, posterior_mean, fstd, posterior_lengthscale, posterior_signal_variance, posterior_noise_variance)

    return posterior_mean, fstd, posterior_lengthscale, posterior_signal_variance, posterior_noise_variance