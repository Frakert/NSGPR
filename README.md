# NSGPR

NSGPR is a Python library for Non-Stationary Gaussian Process Regression, enabling flexible modeling of data with non-stationary characteristics. It is designed for researchers and practitioners who require advanced Gaussian Process (GP) models beyond the standard stationary assumptions.

## Features

- **Non-Stationary Gaussian Process Regression**: Model data with input-dependent covariance structures.
- **Efficient Inference**: Optimized routines for training and prediction.
- **Integration with Scientific Python Stack**: Compatible with NumPy, SciPy, PyTorch and other common libraries.

## Installation

1. Clone the repository:
    ```bash
    git clone
    cd NSGPR
    ```
2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

```python
from NSGPy import NSGP

# Example data
X = ...
y = ...

# Initialize and fit the model
model = NSGP()
model.fit(X, y, "ls") # The last argument "ls" specifies which nonstationary functions to learn (lengthscale, signal variance)

# Make predictions
y_pred, y_var = model.predict(X_test)
```

See the [examples](examples/) directory for more detailed usage and advanced features.

## Documentation
- [Examples](examples/)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This library is inspired by foundational work in Gaussian Processes and non-stationary modeling. The work presented in "Non-Stationary Gaussian Process Regression with Hamiltonian Monte Carlo" and its MATLAB code was used as a basis for this package. 