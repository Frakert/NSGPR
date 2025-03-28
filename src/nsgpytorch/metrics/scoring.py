import torch

def r_score(y_test: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Function that calculates the Coefficient of Determination (R^2 score).

    Parameters
    ----------
    y_test : torch.Tensor, shape (n_samples)
        Real, observed values
    y_pred : torch.Tensor, shape (n_samples)
        Values predicted by the model.

    Returns
    -------
    r_squared : float
        Coefficient of Determination
    """
    if y_test.shape != y_pred.shape:
        raise ValueError("The test data and prediction data need to be the same shape.")
    
    # Ensure that the tensors are of type float for the calculation
    y_test = y_test.float()
    y_pred = y_pred.float()

    ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
    ss_residual = torch.sum((y_test - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()


def mean_squared_error(y_test: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the mean squared error.

    Parameters
    ----------
    y_test : torch.Tensor, shape (n_samples)
        Real, observed values
    y_pred : torch.Tensor, shape (n_samples)
        Values predicted by the model.

    Returns
    -------
    mse : float
        Mean Squared Error
    """
    if y_test.shape != y_pred.shape:
        raise ValueError("The test data and prediction data need to be the same shape.")
    
    return torch.mean((y_test - y_pred) ** 2).item()

def negative_log_predictive_density(y_test: torch.Tensor, y_pred: torch.Tensor, y_std: torch.Tensor) -> float:
    """
    Calculate the negative log predictive density (NLPD).

    Parameters
    ----------
    y_test : torch.Tensor, shape (n_samples)
        Real, observed values
    y_pred : torch.Tensor, shape (n_samples)
        Values predicted by the model.
    y_std : torch.Tensor, shape (n_samples)
        Standard deviation of predicted values.

    Returns
    -------
    nlpd : float
        Negative Log Predictive Density
    """
    if y_test.shape != y_pred.shape or y_test.shape != y_std.shape:
        raise ValueError("All input tensors must have the same shape.")
    
    squared_error_term = 0.5 * ((y_test - y_pred) / y_std) ** 2
    normalization_term = 0.5 * torch.log(2 * torch.pi * y_std**2)
    nlpd = torch.mean(squared_error_term + normalization_term)
    
    return nlpd.item()
