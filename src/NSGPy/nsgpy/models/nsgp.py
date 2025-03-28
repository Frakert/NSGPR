import numpy as np

from .gpmodel import nsgp
from ..inference.posterior import nsgp_posterior


class NSGP:
    def __init__(self):
        """Initialize NSGP model with default parameters."""
        self.gp = None  # Will hold the trained GPModel
        self.X_train = None
        self.Y_train = None
        self.hyperparams = {
            "init_lengthscale": 0.05,
            "init_signal_variance": 0.3,
            "init_noise_variance": 0.05,
            "beta_lengthscale": 0.2,
            "beta_signal_variance": 0.2,
            "beta_noise_variance": 0.3,
            "alpha_lengthscale": 1,
            "alpha_signal_variance": 1,
            "alpha_noise_variance": 1,
        }
        self.nonstationary_functions = "lso"  # Default: Learn all nonstationary functions
        self.optimization_method = "grad"     # Default: Use gradient-based optimization

    def fit(self, X_train, Y_train, nonstationary_functions=None, optimization_method=None, **kwargs):
        """
        Train the NSGP model using the `nsgp` function (low-level).
        """
        self.X_train = X_train
        self.Y_train = Y_train

        # Override default settings if provided
        if nonstationary_functions is not None:
            self.nonstationary_functions = nonstationary_functions
        if optimization_method is not None:
            self.optimization_method = optimization_method

        # Update hyperparameters with any provided values
        self.hyperparams.update(kwargs)

        # Call the low-level nsgp function to train
        self.gp = nsgp(
            X_train, 
            Y_train, 
            nonstationary_functions=self.nonstationary_functions,
            optim=self.optimization_method,
            **self.hyperparams
        )

    def predict(self, X_test, return_std=True, return_cov=False):
        """
        Make predictions using the trained GP model.

        This method computes the predictive mean and uncertainty (standard deviation or covariance) 
        for the given test inputs using the trained Nonstationary Gaussian Process (NSGP).

        Parameters
        ----------
        X_test : ndarray, shape (n, d)
            Test input data, where `n` is the number of test points and `d` is the input dimension.
        return_std : bool, optional (default=True)
            If True, returns the standard deviation of the predictions along with the mean.
        return_cov : bool, optional (default=False)
            If True, returns the covariance matrix of the predictions instead of standard deviation.
            Note: `return_cov` takes precedence over `return_std`.

        Returns
        -------
        mean_pred : ndarray, shape (n,)
            Predicted mean values at the test points.
        std_pred or cov_pred : ndarray
            - If `return_std=True` and `return_cov=False`, returns the standard deviation of predictions (shape: (n,)).
            - If `return_cov=True`, returns the covariance matrix of predictions (shape: (n, n)).
            - If both `return_std` and `return_cov` are False, only the mean predictions are returned.

        Raises
        ------
        ValueError
            If the model has not been trained (i.e., `fit()` has not been called).
        """
        if self.gp is None:
            raise ValueError("Model is not trained yet. Call `fit()` first.")

        # Compute predictive mean and covariance using the posterior function
        mean_pred, std_pred, *_ = nsgp_posterior(self.gp, X_test)

        if return_cov:
            return mean_pred, std_pred
        elif return_std:
            return mean_pred, std_pred
        return mean_pred

    def set_params(self, **kwargs):
        """
        Update hyperparameters before training.
        """
        self.hyperparams.update(kwargs)

    def get_params(self):
        """
        Retrieve current hyperparameter settings.
        """
        return self.hyperparams

    def optimizer(self, optimizer="grad", random_restarts=5, max_iteration=5000):
        """(Optional) Placeholder for an explicit optimizer function."""
        self.optimization_method = optimizer
        self.hyperparams["random_restarts"] = random_restarts
        self.hyperparams["gradient_iterations"] = max_iteration

    def score(self, X_test, Y_test, scoring="r2"):
        """
        Evaluate the model using the specified metric.
        """
        if self.gp is None:
            raise ValueError("Model is not trained yet. Call `fit()` first.")

        Y_pred, Y_std = self.predict(X_test, return_std=True)

        # Placeholder for actual scoring functions (add your own)
        metrics = {
            #"r2": lambda: r2_score(Y_test, Y_pred),
            #"mse": lambda: mean_squared_error(Y_test, Y_pred),
            #"nmse": lambda: mean_squared_error(Y_test, Y_pred) / np.var(Y_test),
            #"nlpd": lambda: self.negative_log_predictive_density(Y_test, Y_pred, Y_std),
            #"log_marginal_likelihood": lambda: self.gp.log_marginal_likelihood()
        }

        if scoring not in metrics:
            raise ValueError(f"Unknown scoring method '{scoring}'. Available: {list(metrics.keys())}")

        return metrics[scoring]()  # Compute the chosen metric