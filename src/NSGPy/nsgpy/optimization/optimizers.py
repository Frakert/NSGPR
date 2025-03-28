import numpy as np

from ..metrics.likelihoods import nsgpmll
from ..utils.plotting import plot_nsgp_2d
from .gradients import lengthscale_gradient, signal_variance_gradient, noise_variance_gradient, betas_gradient, alphas_gradient


def nsgpgrad(gp):
    """
    Learn a nonstationary Gaussian Process (NSGP) by optimizing over multiple restarts.

    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    gp : object
        Contains necessary kernel parameters and data.
    mll : float
        The maximum MLL score.
    """
    if gp.verbose_output:
        print(f"Optimizing for {gp.random_restarts} restarts ...")
        print("  model   iter stepsize       mll")

    # Store models and MLL values
    gps = [None] * gp.random_restarts
    mlls = np.zeros(gp.random_restarts)

    # First gradient descent run with existing initial values
    gps[0] = gradient(gp)
    mlls[0], *_ = nsgpmll(gp)

    # Perform additional runs with randomized initial conditions
    for iter in range(1, gp.random_restarts):
        # Set random initial values
        gp.initial_lengthscale = np.random.uniform(0.03, 0.3) # Start with larger lengthscale and slowly lower the min value for more stable starts
        gp.initial_signal_variance = np.random.uniform(0.1, 0.5)
        gp.initial_noise_variance = np.random.uniform(0.01, 0.10)
        gp.initialize_kernels()  # Perform necessary precomputations

        gps[iter] = gradient(gp)
        mlls[iter], *_ = nsgpmll(gp)

    # Select the best model based on maximum MLL
    best_index = np.argmax(mlls)
    gp = gps[best_index]
    mll = mlls[best_index]

    if gp.verbose_output:
        print(f"Best model mll={mll:.2f}")

    return gp, mll


def gradient(gp, nonstationary_functions=None):
    """
    Compute gradient descent over the marginal log likelihood (MLL) of the
    adaptiveGP against the 9 parameters and 3 latent functions.

    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    nonstationary_functions : str, optional
        String containing the non-stationary parameters.
    
    Returns
    -------
    gp : object
        Contains updated kernel parameters and data.
    """
    if nonstationary_functions is None:
        nonstationary_functions = gp.nonstationary_functions

    # Initial step size
    step = 1e-5 

    # MLL of the initial values
    mlls = np.zeros((gp.gradient_iterations, gp.n_targets))
    mlls[0, :], *_ = nsgpmll(gp)

    if gp.verbose_output:
        print(f"{nonstationary_functions:5}gp {1:6d} {np.log10(step):8.2f} {np.mean(mlls[0, :]):9.2f}")

    if gp.plot_iterations:
        plot_nsgp_2d(gp)
    
    # Initialize derivative storage
    dbl = dbs = dbo = 0
    dal = das = dao = 0

    # Gradient steps over all parameters
    for iter in range(1, gp.gradient_iterations):
        # Compute derivatives of latent functions
        dwl_l = lengthscale_gradient(gp)
        dwl_s = signal_variance_gradient(gp)
        dwl_o = noise_variance_gradient(gp)

        # Compute derivatives of betas and alphas if present in nonstationary_functions
        if 'b' in gp.nonstationary_functions:
            dbl, dbs, dbo = betas_gradient(gp)
            
            # Gradient steps for beta
            gp.beta_lengthscale += step * dbl
            gp.beta_signal_variance += step * dbs
            gp.beta_noise_variance += step * dbo

        if 'a' in gp.nonstationary_functions:
            dal, das, dao = alphas_gradient(gp)
                    
            # Gradient steps for alpha
            gp.alpha_lengthscale += step * dal
            gp.alpha_signal_variance += step * das
            gp.alpha_noise_variance += step * dao

        # Save old parameters
        l_cp = gp.whitened_log_lengthscale
        s_cp = gp.whitened_log_signal_variance
        o_cp = gp.whitened_log_noise_variance

        # Gradient steps for latent functions (some type of clamping works)
        gp.whitened_log_lengthscale = np.clip(gp.whitened_log_lengthscale + step * dwl_l, -5, 5) # Start with larger lengthscale and slowly lower the min value for more stable starts
        gp.whitened_log_signal_variance = np.clip(gp.whitened_log_signal_variance + step * dwl_s, -5, 5)
        gp.whitened_log_noise_variance = np.clip(gp.whitened_log_noise_variance + step * dwl_o, -np.inf, 5)

        # reset the latent variables 
        gp.reset_latent_variables()

        # Compute MLL
        mlls[iter, :], *_ = nsgpmll(gp)

        # Update step size
        if np.all(mlls[iter, :] < mlls[iter - 1, :]):  # If overshooting, revert and decrease step
            gp.whitened_log_lengthscale = l_cp
            gp.whitened_log_signal_variance = s_cp
            gp.whitened_log_noise_variance = o_cp
            mlls[iter, :] = mlls[iter - 1, :]
            step *= 0.70  # Reduce step size

            # reset the latent variables 
            gp.reset_latent_variables()
        else:
            step *= 1.10  # Increase step size if improving

        if gp.verbose_output and iter % 100 == 0:
            print(f"{nonstationary_functions:5}gp {iter:6d} {np.log10(step):8.2f} {np.mean(mlls[iter, :]):9.2f}")

        # Convergence criteria
        if (np.log10(step) < -7) or (iter > 50 and np.mean(mlls[iter, :] - mlls[iter - 30, :]) < 0.1):
            if gp.verbose_output:
                print(f"{nonstationary_functions:5}gp {iter:6d} {np.log10(step):8.2f} {np.mean(mlls[iter, :]):9.2f}")
            break

        if gp.plot_iterations and iter % 10 == 0:
            plot_nsgp_2d(gp)

    return gp