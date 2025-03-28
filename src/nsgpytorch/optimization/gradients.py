import torch


def lengthscale_gradient(gp: object) -> torch.Tensor:
    """
    Calculates the derivative of the lengthscale latent function with respect to 
    marginal log-likelihood.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    whitened_lengthscale_derivative : torch.Tensor, shape (batch_size, n_samples)
        Computed derivative.

    Notes
    -----
    The partial derivative for lengthscale is defined as:

    ∂log L / ∂l̃(x) = 0.5 tr((α αᵀ - K⁻¹_y) ∂K_y / ∂l̃(x)) - K⁻¹_l (l̃ - mu_l)

    Detailed component definitions:

    1. Kernel Derivative ∂K_y / ∂l̃(x)
        ∂K_y(x, x') / ∂l̃(x) = S(x, x') E(x, x') / (R(x, x') L(x,x')^3) l(x) l(x') (4 ||x - x'||² l(x)² - l(x)² + l(x')^4)

        Where:
        - K_y : NS-RBF Kernel with noise
        - l̃ : log lenthscale
        - ||x - x'||² : Squared Euclidean distance between x and x'
        - S(x, x') = σ(x) σ(x') 
        - R(x, x') = sqrt(2 l(x) l(x') / L(x, x'))
        - E(x, x') = exp(-||x - x'||² / L(x, x'))
        - L(x, x') = l(x)² + l(x')²
        
        The derivative matrix ∂K_y(x, x') / ∂l(x) becomes a 'plus' matrix where only i'th column and row are nonzero.

    2. Woodbury matrix: α αᵀ - K⁻¹_y
    
        Where:
        - α = K⁻¹_y y : Woodbury vector
        - K⁻¹_y : Inverse NS-RBF Kernel with noise
    """
    # Get Woodbury matrix: α αᵀ - K⁻¹_y
    woodbury_matrix = gp.get_woodbury_matrix
    
    if 'l' not in gp.nonstationary_functions: # Scalar case
        kernel_without_noise = gp.get_ns_rbf_kernel
                
        lengthscale_mean_sqrt = torch.exp((-2) * gp.mean_log_lengthscale)
        kernel_derivative = lengthscale_mean_sqrt[:, None, None] * (gp.distance_matrix * kernel_without_noise)
        
        # Calculate the derivative
        diff = gp.log_lengthscale - gp.mean_log_lengthscale[:, None]
        lengthscale_derivative = 0.5 * (woodbury_matrix * kernel_derivative.transpose(1, 2)).sum(dim=2) \
                                  - torch.cholesky_solve(diff.unsqueeze(2), gp.cholesky_lengthscale, upper=False).squeeze(-1)
        
        # Distribute the total derivative to all points
        lengthscale_derivative = torch.ones(gp.n_samples, device=gp.device) * lengthscale_derivative.sum(dim=-1, keepdim=True)

        # Determine the whitened log lengthscale derivative
        whitened_lengthscale_derivative = torch.linalg.solve_triangular(
            gp.cholesky_signal_variance, 
            lengthscale_derivative.unsqueeze(2), 
            upper=False
        ).squeeze(-1)
        
    else: # Non-scalar case
        # Precompute 
        lengthscale = gp.lengthscale
        lengthscale2 = lengthscale ** 2
        lengthscale4 = lengthscale2 ** 2
        lengthscale_signal = torch.sqrt(lengthscale) * gp.signal_variance
        
        # Compute lengthscale sum for denominators: l(x)² + l(x')²
        squared_lengthscale_sum = lengthscale2[:, :, None] + lengthscale2[:, None, :]

        # Compute: 4 ||x - x'||² l(x)² - l(x)² + l(x')^4
        kernel_derivative = gp.distance_matrix.unsqueeze(0) * (4 * lengthscale2).unsqueeze(1) \
                            - lengthscale4[:, :, None] - lengthscale4[:, None, :]
        
        # Multiply with scaling term: σ(x) σ(x') sqrt(2 l(x) l(x'))
        kernel_derivative *= (lengthscale_signal / (2 ** 0.5))[:, :, None] * lengthscale_signal[:, None, :]

        # Multiply with exponential term: exp(-||x - x'||²/ L(x, x'))
        kernel_derivative *= torch.exp(-gp.distance_matrix / squared_lengthscale_sum)

        # Devide by: (l(x)² + l(x')²)^(2.5)
        kernel_derivative /= squared_lengthscale_sum ** 2 * torch.sqrt(squared_lengthscale_sum)
        
        # Compute: 0.5 tr((α αᵀ - K⁻¹_y) ∂K_y / ∂l̃(x)) 
        K_W = woodbury_matrix * kernel_derivative
        col_contrib = K_W.sum(dim=2)
        row_contrib = K_W.sum(dim=1)
        lengthscale_derivative = 0.5 * (col_contrib + row_contrib)

        # Subtract: [K⁻¹_l (l̃ - mu_l)](x)
        lengthscale_derivative -= torch.cholesky_solve(
            (gp.log_lengthscale - gp.mean_log_lengthscale[:, None]).unsqueeze(2), 
            gp.cholesky_lengthscale, 
            upper=False
        ).squeeze(-1)
        
        # Determine the whitened log lengthscale derivative
        whitened_lengthscale_derivative = lengthscale_derivative @ gp.cholesky_signal_variance
    
    return whitened_lengthscale_derivative


def signal_variance_gradient(gp: object) -> torch.Tensor:
    """
    Derivative of the sigma latent function with respect to MLL.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    whitened_signal_variance_derivative : torch.Tensor, shape (batch_size, n_samples)
        Computed derivative.

    Notes
    -----
    The partial derivative for signal variance is defined as:

    ∂log L / ∂σ(x) = diag((α αᵀ - K⁻¹_y) K_f) - K⁻¹_σ (σ - mu_σ)

    Where:
    - K_y : NS-RBF Kernel with noise
    - K_f : NS-RBF Kernel without noise
    - K_σ : Prior of the signal variance
    - σ : log signal variance
    """
    # Get Woodbury matrix: α αᵀ - K⁻¹_y
    woodbury_matrix = gp.get_woodbury_matrix
    
    # Compute: K⁻¹_σ (σ - mu_σ)
    sol = torch.cholesky_solve(
        (gp.log_signal_variance - gp.mean_log_signal_variance[:, None]).unsqueeze(2), 
        gp.cholesky_signal_variance, 
        upper=False
    ).squeeze(-1)

    # Compute: diag((α αᵀ - K⁻¹_y) K_f) - K⁻¹_σ (σ - mu_σ)
    signal_variance_derivative = 2 * (woodbury_matrix * gp.get_ns_rbf_kernel.transpose(1, 2)).sum(dim=2) - sol
    
    if 's' in gp.nonstationary_functions: # Non-scalar case 
        whitened_signal_variance_derivative = signal_variance_derivative @ gp.cholesky_signal_variance
    else: # Scalar case 
        # Distribute the total derivative to all points
        signal_variance_derivative = torch.ones(gp.n_samples, device=gp.device) * signal_variance_derivative.sum(dim=-1, keepdim=True)
        
        # Determine the whitened log signal variance derivative
        whitened_signal_variance_derivative = torch.linalg.solve_triangular(
            gp.cholesky_signal_variance, 
            signal_variance_derivative.unsqueeze(2), 
            upper=False
        ).squeeze(-1)
    
    return whitened_signal_variance_derivative


def noise_variance_gradient(gp: object) -> torch.Tensor:
    """
    Derivative of the noise latent function with respect to MLL.
    
    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    
    Returns
    -------
    whitened_noise_variance_derivative : torch.Tensor, shape (batch_size, n_samples)
        Computed derivative.
    
    Notes
    -----
    The partial derivative for noise variance is defined as:

    ∂log L / ∂ω(x) = diag((α αᵀ - K⁻¹_y) Ω) - K⁻¹_ω (ω - mu_ω)

    Where:
    - K_y : NS-RBF Kernel with noise
    - Ω = diag(ω^2) : Noise
    - K_ω : Prior of the noise variance
    - ω : log noise variance
    """
    # Get Woodbury matrix: α αᵀ - K⁻¹_y
    woodbury_matrix = gp.get_woodbury_matrix
    
    # Compute: Ω = diag(ω^2)
    omega = torch.exp(2 * gp.log_noise_variance)

    # Compute: K⁻¹_ω (ω - mu_ω)
    sol = torch.cholesky_solve(
        (gp.log_noise_variance - gp.mean_log_noise_variance[:, None]).unsqueeze(2), 
        gp.cholesky_noise_variance, 
        upper=False
    ).squeeze(-1)

    # Compute: diag((α αᵀ - K⁻¹_y) Ω) - K⁻¹_ω (ω - mu_ω)
    noise_variance_derivative = torch.diagonal(woodbury_matrix, dim1=-2, dim2=-1) * omega - sol
    
    if 'o' in gp.nonstationary_functions: # Non-scalar case 
        whitened_noise_variance_derivative = noise_variance_derivative @ gp.cholesky_noise_variance
    else: # Scalar case 
        # Distribute the total derivative to all points
        noise_variance_derivative = torch.ones(gp.n_samples, device=gp.device) * noise_variance_derivative.sum(dim=-1, keepdim=True)
        
        # Determine the whitened log noise variance derivative
        whitened_noise_variance_derivative = torch.linalg.solve_triangular(
            gp.cholesky_noise_variance, 
            noise_variance_derivative.unsqueeze(2), 
            upper=False
        ).squeeze(-1)
    
    return whitened_noise_variance_derivative