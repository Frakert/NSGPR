import numpy as np
from scipy.linalg import eigvalsh


def condition_number_jitter(A, cond_thresh=1e5):
    """Computes jitter to ensure condition number does not exceed cond_thresh."""
    # may also be approximated by (np.max(np.diag(L)) / np.min(np.diag(L))) ** 2
    eigvals = eigvalsh(A, check_finite=False)  # Compute all eigenvalues
    lambda_min, lambda_max = eigvals[0], eigvals[-1]

    # Compute jitter needed to maintain desired condition number
    target_min_lambda = lambda_max / cond_thresh
    jitter = max(0, target_min_lambda - lambda_min)

    return jitter * np.eye(A.shape[0])