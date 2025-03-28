"""
nsgpr: A Non-Stationary Gaussian Process Regression package.
"""

# Versioning
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"  # Fallback version

# Define the public API (explicitly control what is accessible)
__all__ = ["NSGP", "nsgp", "plot_nsgp_1d", "plot_kernel_1d", "plot_nsgp_2d", "nsgp_posterior"]

# Lazy loading to prevent circular imports and speed up module loading
def __getattr__(name):
    if name == "NSGP":
        from .models.nsgp import NSGP
        return NSGP
    elif name == "nsgp":
        from .models.gpmodel import nsgp
        return nsgp
    elif name == "plot_nsgp_1d":
        from .utils.plotting import plot_nsgp_1d
        return plot_nsgp_1d
    elif name == "plot_kernel_1d":
        from .utils.plotting import plot_kernel_1d
        return plot_kernel_1d
    elif name == "plot_nsgp_2d":
        from .utils.plotting import plot_nsgp_2d
        return plot_nsgp_2d
    elif name == "nsgp_posterior":
        from .inference.posterior import nsgp_posterior
        return nsgp_posterior
    raise AttributeError(f"Module {__name__} has no attribute {name}")