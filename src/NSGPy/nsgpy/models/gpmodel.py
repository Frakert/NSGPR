import numpy as np
from scipy.spatial.distance import cdist

from ..kernels.stationary import rbf_kernel
from ..kernels.nonstationary import ns_rbf_kernel
from ..optimization.optimizers import nsgpgrad
from ..utils.linalg import cholesky_with_jitter, solve_triangular, solve_cholesky, cholesky_inverse


class GPModel:
    def __init__(self, inputs, outputs):
        """
        Gaussian Process Model with adaptive parameters.

        Parameters
        ----------
        inputs : ndarray, shape (n_samples, n_features)
            Input training data
        outputs : ndarray, shape (n_samples, n_targets)
            Output training data
        """
        # Kernel parameters                 MATLAB variables
        self.initial_lengthscale = 0.05     # init_ell      0.05
        self.initial_signal_variance = 0.30 # init_sigma    0.30
        self.initial_noise_variance = 0.05  # init_omega    0.05
        self.beta_lengthscale = 0.20        # betaell       0.20 
        self.beta_signal_variance= 0.20     # betasigma     0.20
        self.beta_noise_variance = 0.30     # betaomega     0.30
        self.alpha_lengthscale = 1          # alphaell      1
        self.alpha_signal_variance = 1      # alphasigma    1
        self.alpha_noise_variance = 1       # alphaomega    1
        self.mean_function = 0              # muf           0
        self.tolerance = 1e-3               # tol           1e-3

        # Kernel matrices and Cholesky decompositions
        self.kernel_lengthscale = None          # Kl
        self.kernel_signal_variance = None      # Ks
        self.kernel_noise_variance = None       # Ko
        self.cholesky_lengthscale = None        # Ll
        self.cholesky_signal_variance = None    # Ls
        self.cholesky_noise_variance = None     # Lo

        # White-log vectors
        self.whitened_log_lengthscale = None        # wl_ell
        self.whitened_log_signal_variance = None    # wl_sigma
        self.whitened_log_noise_variance = None     # wl_omega

        # Other parameters
        self.nonstationary_functions = "lso"    # nsfuncs   "lso"
        self.optimization_method = "grad"       # optim     "grad"
        self.random_restarts = 5                # restarts  3
        self.gradient_iterations = 5000         # graditers 5000
        self.plot_iterations = False            # plotiters False
        self.verbose_output = True              # verbose   True

        # Data dimensions
        self.n_samples, self.n_features = inputs.shape  # n, d
        self.n_targets = outputs.shape[1]               # p

        # Normalize inputs to [0,1] and outputs to max(abs(y)) = 1
        self.input_min = np.min(inputs, axis=0)                                 # xbias
        self.input_range = np.max(inputs, axis=0) - np.min(inputs, axis=0)      # xscale
        self.output_mean = np.mean(outputs, axis=0)                             # ybias
        self.output_scale = np.max(np.abs(outputs - self.output_mean), axis=0)  # yscale

        self.normalized_inputs = (inputs - self.input_min) / self.input_range       # xtr
        self.normalized_outputs = (outputs - self.output_mean) / self.output_scale  # ytr
        self.mean_function = np.mean(self.normalized_outputs)                       # muf

        # Squared Euclidean distance matrix
        self.distance_matrix = cdist(self.normalized_inputs, self.normalized_inputs, 'sqeuclidean') # D

        # Initialize kernel parameters
        self.initialize_kernels()   # init

    def initialize_kernels(self):
        """Initialize kernel matrices, perform Cholesky decompositions, and set initial parameters."""
        self.kernel_lengthscale = rbf_kernel(
            self.normalized_inputs, 
            self.normalized_inputs, 
            self.beta_lengthscale, 
            self.alpha_lengthscale, 
            self.tolerance
        )
        self.kernel_signal_variance = rbf_kernel(
            self.normalized_inputs, 
            self.normalized_inputs, 
            self.beta_signal_variance, 
            self.alpha_signal_variance, 
            self.tolerance
        )
        self.kernel_noise_variance = rbf_kernel( 
            self.normalized_inputs, 
            self.normalized_inputs, 
            self.beta_noise_variance, 
            self.alpha_noise_variance, 
            self.tolerance
        )

        # Compute Cholesky decompositions
        self.cholesky_lengthscale = cholesky_with_jitter(self.kernel_lengthscale)
        self.cholesky_signal_variance = cholesky_with_jitter(self.kernel_signal_variance)
        self.cholesky_noise_variance = cholesky_with_jitter(self.kernel_noise_variance)

        # Set initial parameters in white-log domain
        self.whitened_log_lengthscale = solve_triangular(
            self.cholesky_lengthscale, 
            np.log(self.initial_lengthscale) * np.ones(self.n_samples)
        )[0]
        self.whitened_log_signal_variance = solve_triangular(
            self.cholesky_signal_variance, 
            np.log(self.initial_signal_variance) * np.ones(self.n_samples)
        )[0]
        self.whitened_log_noise_variance = solve_triangular(
            self.cholesky_noise_variance, 
            np.log(self.initial_noise_variance) * np.ones(self.n_samples)
        )[0]

        self._woodbury_matrix = None
        self._ns_rbf_kernel = None
        self._ns_rbf_kernel_with_noise = None

        self._log_lengthscale = None
        self._log_signal_variance = None
        self._log_noise_variance = None
        self._lengthscale = None
        self._signal_variance = None
        self._noise_variance = None
        self._mean_log_lengthscale = None
        self._mean_log_signal_variance = None
        self._mean_log_noise_variance = None


    def compute_woodbury_matrix(self):
        """Computes and caches the Woodbury matrix (Ktt_noisy⁻¹)."""
        kernel_with_noise = self.get_ns_rbf_kernel_with_noise

        cholesky_kernel = cholesky_with_jitter(kernel_with_noise)
        alpha_vector = solve_cholesky(cholesky_kernel, self.normalized_outputs)
        return np.outer(alpha_vector, alpha_vector) - np.linalg.inv(kernel_with_noise)

    @property
    def get_woodbury_matrix(self):
        """Returns the cached Woodbury matrix."""
        if self._woodbury_matrix is None:
            self._woodbury_matrix = self.compute_woodbury_matrix()
        return self._woodbury_matrix

    @property
    def get_ns_rbf_kernel(self): # l_ell
        if self._ns_rbf_kernel is None:
            self._ns_rbf_kernel = ns_rbf_kernel(
                self.normalized_inputs, 
                self.normalized_inputs, 
                self.lengthscale, 
                self.lengthscale, 
                self.signal_variance, 
                self.signal_variance, 
                0
            )
        return self._ns_rbf_kernel
    
    @property
    def get_ns_rbf_kernel_with_noise(self): # l_ell
        if self._ns_rbf_kernel_with_noise is None:
            kernel = self.get_ns_rbf_kernel.copy()
            prev_noise = 1e-6
            noise_variance = self.noise_variance
            if type(noise_variance) == int:
                noise_variance = max(noise_variance, 1e-3) * np.ones(kernel.shape[0])
            elif noise_variance.size == 1:  # If single float (inside np.array)
                noise_variance = max(noise_variance.item(), 1e-3) * np.ones(kernel.shape[0])
            else:  # If it's a vector, add it to the diagonal
                noise_variance = np.maximum(noise_variance, 1e-3)
            np.fill_diagonal(kernel, kernel.diagonal() - prev_noise + noise_variance**2)
            self._ns_rbf_kernel_with_noise = kernel
        return self._ns_rbf_kernel_with_noise

    @property
    def log_lengthscale(self): # l_ell
        if self._log_lengthscale is None:
            self._log_lengthscale = self.cholesky_lengthscale @ self.whitened_log_lengthscale
        return self._log_lengthscale

    @property
    def log_signal_variance(self): # l_sigma
        if self._log_signal_variance is None:
            self._log_signal_variance = self.cholesky_signal_variance @ self.whitened_log_signal_variance
        return self._log_signal_variance

    @property
    def log_noise_variance(self): # l_omega
        if self._log_noise_variance is None:
            self._log_noise_variance = self.cholesky_noise_variance @ self.whitened_log_noise_variance
        return self._log_noise_variance

    @property
    def lengthscale(self): # ell
        if self._lengthscale is None:
            self._lengthscale = np.exp(self.log_lengthscale)
        return self._lengthscale

    @property
    def signal_variance(self): # sigma
        if self._signal_variance is None:
            self._signal_variance = np.exp(self.log_signal_variance)
        return self._signal_variance

    @property
    def noise_variance(self): # omega
        if self._noise_variance is None:
            self._noise_variance = np.exp(self.log_noise_variance)
        return self._noise_variance
    
    @property
    def mean_log_lengthscale(self): # l_muell
        if self._mean_log_lengthscale is None:
            self._mean_log_lengthscale = np.mean(self.log_lengthscale)
        return self._mean_log_lengthscale

    @property
    def mean_log_signal_variance(self): # l_musigma
        if self._mean_log_signal_variance is None:
            self._mean_log_signal_variance = np.mean(self.log_signal_variance)
        return self._mean_log_signal_variance

    @property
    def mean_log_noise_variance(self): # l_muomega
        if self._mean_log_noise_variance is None:
            self._mean_log_noise_variance = np.mean(self.log_noise_variance)
        return self._mean_log_noise_variance
    
    def reset_latent_variables(self):
        self._woodbury_matrix = None
        self._ns_rbf_kernel = None
        self._ns_rbf_kernel_with_noise = None

        self._log_lengthscale = None
        self._log_signal_variance = None
        self._log_noise_variance = None
        self._lengthscale = None
        self._signal_variance = None
        self._noise_variance = None
        self._mean_log_lengthscale = None
        self._mean_log_signal_variance = None
        self._mean_log_noise_variance = None
    

def nsgp(x, y, nonstationary_functions='lso', optim='grad', *args, **kwargs):
    """
    Main function to learn the adaptive Gaussian Process (NSGP).

    Parameters
    ----------
    x : ndarray, shape (n, d)
        Input data vector.
    y : ndarray, shape (n, p)
        Output data vector.
    nonstationary_functions : str
        String specifying nonstationary components (default: 'lso').
    optim : str
        Optimization method ('grad' [default] or 'hmc').
    args/kwargs : str
        Optional parameters for customization.

    Returns
    -------
    gp : object
        Learned GP model.
    samples : 
        HMC samples (if 'hmc') or empty (if 'grad').
    mll : float
        Marginal log-likelihood.
    mse : float
        Mean squared error.
    nmse : float
        Normalized mean squared error.
    nlpd : float
        Negative log predictive density.
    """

    # Initialize GP model
    gp = GPModel(x, y) 

    # Set remaining arguments as attributes in the GP model
    for i in range(0, len(args), 2):
        setattr(gp, args[i], args[i+1])

    for key, value in kwargs.items():
        setattr(gp, key, value)

    gp.nonstationary_functions = nonstationary_functions
    gp.optimization_method = optim
    gp.initialize_kernels() # Perform necessary initializations

    # Perform optimization (gradient-based)
    gp, _ = nsgpgrad(gp)

    return gp