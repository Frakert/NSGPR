import pytest
import numpy as np
import torch

from src.nsgpytorch.metrics.scoring import *

class Test_scoring():
    def test_r_score(self):
        # Test case 1: Perfect prediction
        y_test = torch.tensor([3.0, 4.0, 5.0])
        y_pred = torch.tensor([3.0, 4.0, 5.0])
        assert r_score(y_test, y_pred) == 1
        
        # Test case 2: Case worse than the average
        y_test = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([4.0, 4.0, 4.0])
        assert r_score(y_test, y_pred) < 0
        
        # Test case 3: Error due to shape mismatch
        y_test = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            r_score(y_test, y_pred)

    # Test for mean_squared_error function
    def test_mean_squared_error(self):
        # Test case 1: Perfect prediction
        y_test = torch.tensor([3.0, 4.0, 5.0])
        y_pred = torch.tensor([3.0, 4.0, 5.0])
        assert torch.isclose(torch.tensor(mean_squared_error(y_test, y_pred)), torch.tensor(0.0), atol=1e-6)
        
        # Test case 2: Some error
        y_test = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([3.0, 2.0, 1.0])
        assert torch.isclose(torch.tensor(mean_squared_error(y_test, y_pred)), torch.tensor(2.66666666), atol=1e-6)
        
        # Test case 3: Error due to shape mismatch
        y_test = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            mean_squared_error(y_test, y_pred)

    # Test for negative_log_predictive_density function
    def test_negative_log_predictive_density(self):
        
        # Test case 2: Some error with std > 0
        y_test = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([3.0, 2.0, 1.0])
        y_std = torch.tensor([1.0, 1.0, 1.0])  # standard deviation
        nlpd_value = negative_log_predictive_density(y_test, y_pred, y_std)
        assert nlpd_value > 0
        
        # Test case 3: Error due to shape mismatch
        y_test = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_std = torch.tensor([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            negative_log_predictive_density(y_test, y_pred, y_std)
    