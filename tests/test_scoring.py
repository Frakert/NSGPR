import pytest
import numpy as np
import torch

from src.nsgpytorch.metrics.scoring import *

class Test_scoring():
    
    def test_r_squared_tensors(self):
        y_true, y_pred = torch.tensor(range(20)), torch.tensor(range(20))
        assert r_score(y_true, y_pred) == 1