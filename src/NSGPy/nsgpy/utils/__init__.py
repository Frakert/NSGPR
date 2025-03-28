"""
nsgpr: A Non-Stationary Gaussian Process Regression package.
"""

# Versioning
try:
    from .._version import __version__
except ImportError:
    __version__ = "0.1.0"  # Fallback version

# Define the public API (explicitly control what is accessible)
__all__ = [
    "solve_triangular", "fast_tdot", "chained_dot", "cholesky", "cholesky_with_jitter", "ensure_fortran_order",
    "solve_cholesky", "solve_cholesky_L", "invert_triangular", "cholesky_inverse", "fast_pdinv",
    "condition_number_jitter",
    "denormalise",
    "plot_nsgp_1d", "plot_kernel_1d", "plot_nsgp_2d"
]

# Lazy loading to prevent circular imports and speed up module loading
def __getattr__(name):
    if name == "solve_triangular":
        from .linalg import solve_triangular
        return solve_triangular
    elif name == "fast_tdot":
        from .linalg import fast_tdot
        return fast_tdot
    elif name == "chained_dot":
        from .linalg import chained_dot
        return chained_dot
    elif name == "cholesky":
        from .linalg import cholesky
        return cholesky
    elif name == "cholesky_with_jitter":
        from .linalg import cholesky_with_jitter
        return cholesky_with_jitter
    elif name == "ensure_fortran_order":
        from .linalg import ensure_fortran_order
        return ensure_fortran_order
    elif name == "solve_cholesky":
        from .linalg import solve_cholesky
        return solve_cholesky
    elif name == "solve_cholesky_L":
        from .linalg import solve_cholesky_L
        return solve_cholesky_L
    elif name == "invert_triangular":
        from .linalg import invert_triangular
        return invert_triangular
    elif name == "cholesky_inverse":
        from .linalg import cholesky_inverse
        return cholesky_inverse
    elif name == "fast_pdinv":
        from .linalg import fast_pdinv
        return fast_pdinv
    elif name == "condition_number_jitter":
        from .numerics import condition_number_jitter
        return condition_number_jitter
    elif name == "denormalise":
        from .preprocessing import denormalise
        return denormalise
    elif name == "plot_nsgp_1d":
        from .plotting import plot_nsgp_1d
        return plot_nsgp_1d
    elif name == "plot_kernel_1d":
        from .plotting import plot_kernel_1d
        return plot_kernel_1d
    elif name == "plot_nsgp_2d":
        from .plotting import plot_nsgp_2d
        return plot_nsgp_2d
    
    raise AttributeError(f"Module {__name__} has no attribute {name}")