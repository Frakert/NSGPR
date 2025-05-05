import torch

from .gpmodel import nsgp
from ..inference.posterior import nsgp_posterior


class NSGP:
    """
    A class implementing a Nonstationary Gaussian Process (NSGP).

    This model allows for flexible and adaptive kernel learning, supporting
    both gradient-based and alternative optimization methods.
    """

    def __init__(self):
        """
        Initialize the NSGP model with default hyperparameters.

        Attributes
        ----------
        gp : object or None
            Stores the trained Gaussian Process model after calling `fit()`.
        X_train : torch.Tensor, shape (n_samples, n_features) or None
            Training input data.
        Y_train : torch.Tensor, shape (n_samples, n_targets) or None
            Training target data.
        hyperparams : dict
            Dictionary containing model hyperparameters.
        nonstationary_functions : str
            Specifies which nonstationary functions to learn. Default is "lso" (learn all).
        optimization_method : str
            Defines the optimization method, default is "grad" (gradient-based).
        """
        self.gp = None
        self.X_train = None
        self.Y_train = None
        self.device = torch.device("cpu") # Default to cpu
        self.hyperparams = {
            "init_lengthscale": [0.01, 0.05, 0.1, 0.01, 0.05, 0.1],
            "init_signal_variance": [0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
            "init_noise_variance": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "beta_lengthscale": 0.2,
            "beta_signal_variance": 0.2,
            "beta_noise_variance": 0.3,
            "alpha_lengthscale": 1,
            "alpha_signal_variance": 1,
            "alpha_noise_variance": 1,
        }
        self.nonstationary_functions = "lso"
        self.optimization_method = "grad"

    def fit(
        self, 
        X_train: torch.Tensor, 
        Y_train: torch.Tensor, 
        nonstationary_functions: str=None, 
        optimization_method: str=None, 
        batch_size: int=1, 
        **kwargs: dict
    ):
        """
        Train the NSGP model.

        Parameters
        ----------
        X_train : torch.Tensor, shape (n_samples, n_features)
            Training input data.
        Y_train : torch.Tensor, shape (n_samples, n_targets)
            Training target data.
        nonstationary_functions : str, optional
            Specifies which nonstationary functions to learn (default: "lso").
        optimization_method : str, optional
            Optimization method, e.g., "grad" for gradient-based optimization.
        batch_size : int, optional (default: 1)
            Number of samples processed per optimization step.
        **kwargs : dict
            Additional hyperparameters to override the defaults.

        Notes
        -----
        - This function initializes the GP model and trains it using `nsgp()`.
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
            batch_size=batch_size,
            device = self.device,
            **self.hyperparams
        )

    def predict(
        self, 
        X_test: torch.Tensor, 
        return_std: bool=True, 
        return_cov: bool=False
    ) -> torch.Tensor:
        """
        Make predictions using the trained GP model.

        This method computes the predictive mean and uncertainty (standard deviation or covariance) 
        for the given test inputs using the trained Nonstationary Gaussian Process (NSGP).

        Parameters
        ----------
        X_test : torch.Tensor, shape (n_samples, n_features)
            Test input data, where `n` is the number of test points and `d` is the input dimension.
        return_std : bool, optional (default is True)
            If True, returns the standard deviation of the predictions along with the mean.
        return_cov : bool, optional (default is False)
            If True, returns the covariance matrix of the predictions instead of standard deviation.
            Note: `return_cov` takes precedence over `return_std`.

        Returns
        -------
        mean_pred : torch.Tensor, shape (n_samples, n_targets)
            Predicted mean values at the test points.
        std_pred or cov_pred : torch.Tensor, shape (n_targets, n_samples) or (n_targets, n_samples, n_samples)
            - If `return_std=True` and `return_cov=False`, returns the standard deviation of predictions (shape: (n_targets, n_samples)).
            - If `return_cov=True`, returns the covariance matrix of predictions (shape: (n_targets, n_samples, n_samples)).
            - If both `return_std` and `return_cov` are False, only the mean predictions are returned.

        Raises
        ------
        ValueError
            If the model has not been trained (i.e., `fit()` has not been called).

        Notes
        -----
        - Uses `nsgp_posterior()` to compute predictions.
        """
        if self.gp is None:
            raise ValueError("Model is not trained yet. Call `fit()` first.")

        # Compute predictive mean and covariance using the posterior function
        mean_pred, std_pred, *_ = nsgp_posterior(self.gp, X_test, return_cov)

        if return_cov or return_std:
            return mean_pred, std_pred
        
        return mean_pred

    def set_params(self, **kwargs: dict):
        """
        Update hyperparameters before training.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs of hyperparameters to update.
        """
        self.hyperparams.update(kwargs)

    def get_params(self) -> dict:
        """
        Update hyperparameters before training.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs of hyperparameters to update.
        """
        return self.hyperparams

    def optimizer(
        self, 
        optimizer: str="grad", 
        n_restarts: int=4, 
        max_iteration: int=5000
    ):
        """
        Configure optimization settings.

        Parameters
        ----------
        optimizer : str, optional (default is "grad")
            Optimization method (e.g., "grad" for gradient-based optimization).
        n_restarts : int, optional (default is 4)
            Number of restarts for optimization.
        max_iteration : int, optional (default is 5000)
            Maximum number of optimization iterations.

        Notes
        -----
        - Stores optimizer settings but does not run optimization itself.
        """
        self.optimization_method = optimizer
        self.hyperparams["n_restarts"] = n_restarts
        self.hyperparams["gradient_iterations"] = max_iteration

    def score(
        self, 
        X_test: torch.Tensor, 
        Y_test: torch.Tensor, 
        scoring: str="r2"
    ) -> torch.Tensor:
        """
        Evaluate the model using a specified scoring metric.

        Parameters
        ----------
        X_test : torch.Tensor, shape (n_samples, n_features)
            Test input data.
        Y_test : torch.Tensor, shape (n_samples, n_targets)
            Ground truth target values.
        scoring : str, optional (default: "r2")
            Scoring method. Supported options:
            - "r2": R-squared score
            - "mse": Mean Squared Error
            - "nmse": Normalized Mean Squared Error
            - "nlpd": Negative Log Predictive Density
            - "log_marginal_likelihood": Log Marginal Likelihood

        Returns
        -------
        torch.Tensor
            The computed score based on the specified metric.

        Raises
        ------
        ValueError
            If the model has not been trained.
            If the provided scoring method is not recognized.

        Notes
        -----
        - Currently, the scoring functions are placeholders and need to be implemented.
        """
        if self.gp is None:
            raise ValueError("Model is not trained yet. Call `fit()` first.")

        Y_pred, Y_std = self.predict(X_test, return_std=True)

        # Placeholder for actual scoring functions
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
    
    def to(self, device):
        """
        Move torch model to another device type.
        """
        self.device = device
    

    
