import torch


def rbf_kernel(
    input_1: torch.Tensor, 
    input_2: torch.Tensor, 
    lengthscale: float = 1.0, 
    signal_variance: float = 1.0, 
    noise_variance: float = 0.0
) -> torch.Tensor:
    """
    Computes the Radial Basis Function (RBF) Kernel, also known as the Squared Exponential Kernel.

    Parameters
    ----------
    inputs1 : torch.Tensor, shape (n_samples1, n_features)
        First input data.
    inputs2 : torch.Tensor, shape (n_samples2, n_features)
        Second input data.
    lengthscale : float, optional
        Lengthscale parameter l controlling kernel smoothness (default is 1.0).
    signal_variance : float, optional
        Signal variance σ² controlling kernel amplitude (default is 1.0).
    noise_variance : float, optional
        Noise variance (default is 0.0).

    Returns
    -------
    kernel_matrix : torch.Tensor, shape (n_samples1, n_samples2)
        Computed kernel matrix.

    Notes
    -----
    The RBF kernel is defined as:
    
    k(x, x') = σ² * exp(-0.5 * ||x - x'||² / l²)
    
    Where:
    - x, x' : Input vectors
    - σ² : Signal variance (kernel amplitude)
    - l : Lengthscale parameter
    - ||x - x'||² : Squared Euclidean distance between x and x'
    """
    # Compute squared Euclidean distance between points: ||x - x'||²
    squared_distances = torch.cdist(input_1, input_2, p=2).pow(2)

    # Compute the RBF kernel.
    kernel_matrix = (signal_variance**2) * torch.exp(-0.5 * squared_distances / (lengthscale**2))
    
    # Add noise to diagonal if the kernel matrix is square
    if input_1.shape == input_2.shape:
        # Ensure a minimum noise level for numerical stability
        noise_variance = max(noise_variance, 1e-6)
        idx = torch.arange(input_1.shape[0], device=input_1.device)

        kernel_matrix[idx, idx] += noise_variance**2 * torch.ones(input_1.shape[0])

    return kernel_matrix