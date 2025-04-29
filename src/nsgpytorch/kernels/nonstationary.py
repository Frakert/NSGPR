import torch


def ns_rbf_kernel(
    input_1: torch.Tensor, 
    input_2: torch.Tensor, 
    lengthscale1: torch.Tensor, 
    lengthscale2: torch.Tensor, 
    signal_variance1: torch.Tensor, 
    signal_variance2: torch.Tensor, 
    noise_variance: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Non-Stationary Radial Basis Function (NS-RBF) kernel.

    Parameters
    ----------
    inputs1 : torch.Tensor, shape (n_samples1, n_features)
        First input data.
    inputs2 : torch.Tensor, shape (n_samples2, n_features)
        Second input data.
    lengthscale1 : torch.Tensor, shape (batch_size, n_samples1)
        Length scale parameter for inputs1.
    lengthscale2 : torch.Tensor, shape (batch_size, n_samples2)
        Length scale parameter for inputs2.
    signal_variance1 : torch.Tensor, shape (batch_size, n_samples1)
        Signal variance for inputs1.
    signal_variance2 : torch.Tensor, shape (batch_size, n_samples2)
        Signal variance for inputs2.
    noise_variance : torch.Tensor or float, shape (batch_size, n_samples1) or scalar
        Noise variance (can be a scalar or a tensor).
    
    Returns
    -------
    kernel_matrix : torch.Tensor, shape (batch_size, n_samples1, n_samples2)
        Computed kernel matrix.

    Notes
    -----
    The NS-RBF kernel is defined as:
    
    k(x, x') = σ(x) σ(x') √(2 l(x) l(x') / (l(x)² + l(x')²)) 
               * exp(-||x - x'||² / (l(x)² + l(x')²))
    
    Where:
    - x, x' : Input vectors
    - σ(x), σ(x') : Signal variance at points x and x'
    - l(x), l(x') : Lengthscale at points x and x'
    - ||x - x'||² : Squared Euclidean distance between x and x'
    """
    # Compute lengthscale sum for denominators: l(x)² + l(x')²
    lengthscale_sum = lengthscale1[:, :, None] ** 2 + lengthscale2[:, None, :] ** 2

    # Compute signal variance scaling terms: σ(x) σ(x') √(2 l(x) l(x'))
    length_signal_1 = signal_variance1 * torch.sqrt(lengthscale1 * 2)
    length_signal_2 = signal_variance2 * torch.sqrt(lengthscale2)
    
    # Compute squared Euclidean distance between points: ||x - x'||²
    squared_distances = torch.cdist(input_1, input_2, p=2).pow(2)
    
    # Calculate the kernel matrix
    kernel_matrix = length_signal_1[:, :, None] * length_signal_2[:, None, :]
    kernel_matrix /= torch.sqrt(lengthscale_sum)
    kernel_matrix *= torch.exp(-squared_distances / lengthscale_sum)
    
    # Add noise to diagonal if the kernel matrix is square
    if input_1.shape == input_2.shape:
        if type(noise_variance) == int:
            noise_variance = max(noise_variance, 1e-6) * torch.ones((kernel_matrix.shape[0], kernel_matrix.shape[1]))
        else:
            noise_variance = torch.clamp(noise_variance, min=1e-6)

        idx = torch.arange(lengthscale1.shape[1], device=input_1.device)
        kernel_matrix[:, idx, idx] += noise_variance**2
    
    return kernel_matrix