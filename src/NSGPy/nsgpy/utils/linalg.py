import numpy as np
from scipy.linalg import lapack, blas


def ensure_fortran_order(A):
    """
    Ensures A is Fortran-ordered for efficient LAPACK computations.
    """
    if A.flags['F_CONTIGUOUS']:
        return A
    print("why are your arrays not F order?")
    return np.asfortranarray(A)


def solve_triangular(A, B, lower=True, trans=False, unitdiag=False):
    """
    Solves a triangular system of equations: A * X = B or A.T * X = B.

    :param A: Triangular matrix
    :param B: Right-hand side matrix
    :param lower: Whether A is lower-triangular (default: True)
    :param transpose: Solve A.T * X = B if True
    :param unit_diag: If True, A is assumed to have a unit diagonal
    :returns: Solution matrix X
    """
    A = ensure_fortran_order(A)
    #Note: B does not seem to need to be F ordered!
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)


def fast_tdot(matrix, output=None):
    """
    Efficiently computes np.dot(matrix, matrix.T) using BLAS for large matrices.

    :param matrix: Input 2D array
    :param output: Optional output array (must be F-ordered)
    :returns: matrix @ matrix.T
    """
    if matrix.dtype != 'float64' or matrix.ndim != 2:
        return np.dot(matrix, matrix.T)

    n = matrix.shape[0]
    if output is None:
        output = np.zeros((n, n), dtype=np.float64, order='F')

    matrix = np.asfortranarray(matrix)
    output = blas.dsyrk(alpha=1.0, a=matrix, beta=0.0, c=output, overwrite_c=True, trans=True, lower=False)
    
    return np.tril(output) + np.tril(output, -1).T


def chained_dot(*args):
    """
    Computes the matrix product of multiple arguments efficiently.

    Example:
        chained_dot(A, B, C) is equivalent to (A @ B) @ C.
    """
    if len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return np.dot(args[0], args[1])
    else:
        return np.dot(chained_dot(*args[:-1]), args[-1])


def cholesky(A):
    """
    Computes the Cholesky decomposition.

    :param A: Input matrix (must be symmetric positive definite)
    :returns: Cholesky factor L such that A = L @ L.T
    """
    A = np.asfortranarray(A)
    return lapack.dpotrf(A, lower=True)
    

def cholesky_with_jitter(A, maxtries=5):
    """
    Computes the Cholesky decomposition with jitter if necessary.

    :param A: Input matrix (must be symmetric positive definite)
    :param max_attempts: Number of jitter attempts if A is not SPD
    :returns: Cholesky factor L such that A = L @ L.T
    """
    A = np.asfortranarray(A)
    L, info = lapack.dpotrf(A, lower=True)
    if info == 0:
        return L
    
    print("cholesky_with_jitter has non-SPD matrix")

    # Handle non-positive definite case
    diagA = np.diag(A)
    if np.any(diagA <= 0.):
        raise ValueError("Matrix not PD: non-positive diagonal elements")

    jitter = diagA.mean() * 1e-6
    num_tries = 1
    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            A_jittered = A.copy()
            A_jittered.flat[::A.shape[0] + 1] += jitter  # In-place diagonal modification
            L, info = lapack.dpotrf(A_jittered, lower=True)
            if info == 0:
                return L
        except Exception:
            jitter *= 10  # Increase jitter exponentially
        num_tries += 1

    raise ValueError("Matrix not positive definite, even with jitter.")
    

def solve_cholesky(L, B, lower=True):
    """
    Solves Ax = B given the Cholesky factor L (or U) such that A = LLᵀ (or UᵀU).

    Better for small matrices.

    :param L: Positive definite matrix
    :param B: Right-hand side matrix
    :param lower: Whether to use lower-triangular Cholesky factor
    :returns: Solution matrix X
    """
    L = ensure_fortran_order(L)
    return lapack.dpotrs(L, B, lower=lower)[0]


def solve_cholesky_L(L, B, lower=True):
    """
    Solves Ax = B given the Cholesky factor L (or U) such that A = LLᵀ (or UᵀU).

    Better for large matrices.

    :param L: (ndarray) Lower (or upper) triangular Cholesky factor of A.
    :param B: (ndarray) Right-hand side matrix or vector.
    :param lower: (bool) If True, L is lower-triangular. If False, L is upper-triangular.
    :return: (ndarray) Solution matrix X.
    """
    # Solve L y = B for y (forward substitution)
    Y, info = lapack.dtrtrs(L, B, lower=lower, trans=False, unitdiag=False)
    if info != 0:
        raise ValueError("Solving Ly = B failed, possibly due to a singular matrix.")

    # Solve Lᵀ x = y for x (back substitution)
    X, info = lapack.dtrtrs(L, Y, lower=lower, trans=True, unitdiag=False)
    if info != 0:
        raise ValueError("Solving Lᵀx = y failed, possibly due to a singular matrix.")
    
    return X


def invert_triangular(L):
    """
    Computes the inverse of a lower triangular matrix.

    :param L: Lower-triangular matrix (from Cholesky decomposition)
    :returns: Inverse of L
    """
    L = ensure_fortran_order(L)
    return lapack.dtrtri(L, lower=True)[0]


def cholesky_inverse(L, lower=True):
    """
    Computes the inverse of a symmetric positive definite matrix using Cholesky.
    Optimized to be faster than dpotri.

    :param L: Symmetric positive definite matrix
    :param lower: Whether to use lower-triangular Cholesky factor
    :returns: Inverse of A
    """
    # Compute L⁻¹ using dtrtri (in-place inversion of a triangular matrix)
    L_inv, info = lapack.dtrtri(L, lower=lower, unitdiag=False)
    if info != 0:
        raise ValueError("Triangular matrix inversion failed.")

    # Compute A⁻¹ = L⁻ᵀ * L⁻¹
    return L_inv.T @ L_inv


def fast_pdinv(A, *args):
    """
    Computes the inverse of a positive definite matrix efficiently using Cholesky.

    :param A: Positive definite matrix
    :returns:
        - Ai: Inverse of A
        - L: Cholesky decomposition of A
        - Li: Inverse of L
        - logdet: Log determinant of A
    """
    L = cholesky_with_jitter(A, *args)
    logdet = 2.*np.sum(np.log(np.diag(L)))
    Li = invert_triangular(L)
    Ai = cholesky_inverse(L, lower=True)
    Ai = np.tril(Ai) + np.tril(Ai,-1).T
    return Ai, L, Li, logdet