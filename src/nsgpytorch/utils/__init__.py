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
    "condition_number_jitter", "cholesky_with_jitter", "cholesky_with_jitter_batch"
    "denormalise",
    "plot_nsgp_1d", "plot_kernel_1d", "plot_nsgp_2d"
]

# Lazy loading to prevent circular imports and speed up module loading
def __getattr__(name):
    if name == "condition_number_jitter":
        from .numerics import condition_number_jitter
        return condition_number_jitter
    elif name == "cholesky_with_jitter":
        from .numerics import cholesky_with_jitter
        return cholesky_with_jitter
    elif name == "cholesky_with_jitter_batch":
        from .numerics import cholesky_with_jitter_batch
        return cholesky_with_jitter_batch
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