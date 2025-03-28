import torch


def condition_number_jitter(A, cond_thresh=1e5):
    """Computes jitter to ensure condition number does not exceed cond_thresh."""
    # may also be approximated by (np.max(np.diag(L)) / np.min(np.diag(L))) ** 2
    eigvals = torch.linalg.eigvalsh(A)  # Compute all eigenvalues
    if eigvals.ndim == 2:
        lambda_min, lambda_max = eigvals[:,0], eigvals[:,-1]
    else:
        lambda_min, lambda_max = eigvals[0], eigvals[-1]

    # Compute jitter needed to maintain desired condition number
    target_min_lambda = lambda_max / cond_thresh
    jitter = torch.clamp(target_min_lambda - lambda_min, min=0)

    return jitter


def cholesky_with_jitter(A, maxtries=5, jitter_factor=1e-6):
    """
    Computes the Cholesky decomposition with jitter if necessary.

    :param A: (N, N) symmetric matrix (must be SPD or nearly SPD).
    :param maxtries: Maximum jitter attempts if A is not SPD.
    :param jitter_factor: Initial jitter relative to mean diagonal.
    :returns: Cholesky factor L such that A = L @ L.T
    """
    N = A.shape[0]
    diag_idx = torch.arange(N, device=A.device)  # Indices for diagonal elements
    jitter = torch.diagonal(A).mean() * jitter_factor  # Initial jitter based on mean value

    for attempt in range(maxtries):
        L, info = torch.linalg.cholesky_ex(A)  # Attempt Cholesky decomposition

        if info.item() == 0:  # Check if decomposition was successful
            return L

        print("cholesky_with_jitter has non-SPD matrix")

        # If failed, add jitter to the diagonal and retry
        A = A.clone()  # Ensure we don't modify the original matrix in-place
        A[diag_idx, diag_idx] += jitter * torch.ones((A.shape[0]))
        jitter *= 10  # Exponentially increase jitter

    raise ValueError("Matrix is not positive definite, even with jitter.")


def cholesky_with_jitter_batch(A, maxtries=5, jitter_factor=1e-6):
    """
    Computes the batched Cholesky decomposition with jitter only for failed cases.

    :param A: Input matrix (B, N, N), must be symmetric.
    :param maxtries: Max number of jitter attempts if A is not SPD.
    :param jitter_factor: Initial jitter relative to mean diagonal.
    :returns: Cholesky factor L such that A = L @ L.T
    """
    B, N, _ = A.shape  # Batch size and matrix size
    diag_idx = torch.arange(N, device=A.device)  # Indices for diagonal
    jitter = torch.diagonal(A).mean(dim=(0), keepdim=True) * jitter_factor  # Batch-wise jitter initialization

    for attempt in range(maxtries):
        L, info = torch.linalg.cholesky_ex(A)  # Attempt Cholesky decomposition

        if torch.all(info == 0):  # Check if all succeeded
            return L

        print("cholesky_with_jitter_batch has non-SPD matrix")
        print(info)

        # Identify failed cases
        failed_mask = info > 0  # Mask where Cholesky failed
        if not torch.any(failed_mask):  # If all succeeded, return L
            return L

        # Add jitter only to failed matrices
        A[failed_mask][:,diag_idx, diag_idx] += (jitter.squeeze(-1) * torch.ones((A.shape[1])).unsqueeze(-1)).T[failed_mask].squeeze(0)
        jitter *= 10  # Increase jitter exponentially for the next attempt

    raise ValueError("Some matrices remained non-SPD even with jitter.")
