import torch

from ..kernels.stationary import rbf_kernel
from ..kernels.nonstationary import ns_rbf_kernel
from ..optimization.optimizers import nsgpgrad
from ..utils.numerics import cholesky_with_jitter, cholesky_with_jitter_batch
from ..metrics import scoring


class GPModel:
    def __init__(self, inputs, outputs, device=None, batch_size=1):
        """
        Gaussian Process Model with adaptive parameters using PyTorch.

        Parameters
        ----------
        inputs : torch.Tensor, shape (n_samples, n_features)
            Input training data
        outputs : torch.Tensor, shape (n_samples, n_targets)
            Output training data
        device : str or torch.device, optional
            Device to use ('cpu' or 'cuda').
        batch_size: int
            The number of concurrent processes
        """

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputs = inputs.to(self.device)
        self.outputs = outputs.to(self.device)
        
        self.n_samples, self.n_features = self.inputs.shape
        self.n_targets = self.outputs.shape[1]
        self.batch_size = batch_size
        self.batch_I = torch.eye(self.n_samples, device=self.device).expand(batch_size, -1, -1)  # Identity matrix for each batch

        self.which_target = torch.empty((self.batch_size,)) # whould be a vector of size (batch_size) and contains information about which target is calculates (for woodbury and mll) and usefull for saving if mll is higher
        self.which_restart = torch.empty((self.batch_size,))
        self.mll = torch.full((self.n_targets,), float('-inf')) # vector of size (n_targets) the list of mll values for each target, if new process has higher mll value it will be replaced

        # Kernel parameters                 MATLAB variables
        self.initial_lengthscale = [0.01, 0.05, 0.1, 0.01, 0.05, 0.1]     # init_ell      0.05 # should be vectors of size (n_gradient_itterations) now is it seen as [0.05, rand, rand, rand, rand]
        self.initial_signal_variance = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5] # init_sigma    0.30
        self.initial_noise_variance = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]  # init_omega    0.05

        self.beta_lengthscale = 0.20        # betaell       0.20 
        self.beta_signal_variance= 0.20     # betasigma     0.20
        self.beta_noise_variance = 0.30     # betaomega     0.30

        self.alpha_lengthscale = 1          # alphaell      1
        self.alpha_signal_variance = 1      # alphasigma    1
        self.alpha_noise_variance = 1       # alphaomega    1

        self.tolerance = 1e-6               # tol           1e-3

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
        self.n_restarts = 5                # restarts  3
        self.gradient_iterations = 500         # graditers 5000
        self.plot_iterations = False            # plotiters False
        self.verbose_output = True              # verbose   True

        # Normalize inputs to [0,1] and outputs to max(abs(y)) = 1
        self.input_min = self.inputs.min(dim=0).values
        self.input_range = self.inputs.max(dim=0).values - self.input_min
        self.output_mean = self.outputs.mean(dim=0)
        self.output_scale = self.outputs.sub(self.output_mean).abs().max(dim=0).values
        
        self.normalized_inputs = (self.inputs - self.input_min) / self.input_range
        self.normalized_outputs = (self.outputs - self.output_mean) / self.output_scale

        self.batch_outputs = torch.empty((self.n_samples, self.batch_size)) # (to use in woodbury matrix and mll) (batch_size, n_samples)

        # Squared Euclidean distance matrix
        self.distance_matrix = torch.cdist(self.normalized_inputs, self.normalized_inputs, p=2).pow(2)

        # Initialize kernel parameters
        self.initialize_kernels()   # init

    def initialize_kernels(self): # split into two functions (kernels + cholesky) only one time, and function for whitend log lengthscale (restarts) (when ab is sellected it is neccesery to recalc kernel just make if statement)
        """Initialize kernel matrices, perform Cholesky decompositions, and set initial parameters."""
        self.kernel_lengthscale = rbf_kernel(       # (n_samples, n_samples) same kernel for all targets (except if ab is selected)
            self.normalized_inputs, 
            self.normalized_inputs, 
            self.beta_lengthscale, 
            self.alpha_lengthscale, 
            self.tolerance
        )
        self.kernel_signal_variance = rbf_kernel(   # (n_samples, n_samples) same kernel for all targets
            self.normalized_inputs, 
            self.normalized_inputs, 
            self.beta_signal_variance, 
            self.alpha_signal_variance, 
            self.tolerance
        )
        self.kernel_noise_variance = rbf_kernel(    # (n_samples, n_samples) same kernel for all targets
            self.normalized_inputs, 
            self.normalized_inputs, 
            self.beta_noise_variance, 
            self.alpha_noise_variance, 
            self.tolerance
        )

        # Compute Cholesky decompositions
        self.cholesky_lengthscale = cholesky_with_jitter(self.kernel_lengthscale)          # (n_samples, n_samples) same kernel for all targets
        self.cholesky_signal_variance = cholesky_with_jitter(self.kernel_signal_variance)  # (n_samples, n_samples) same kernel for all targets
        self.cholesky_noise_variance = cholesky_with_jitter(self.kernel_noise_variance)    # (n_samples, n_samples) same kernel for all targets

        # Set initial parameters in white-log domain
        self.whitened_log_lengthscale = torch.empty((self.batch_size, self.n_samples))     # (n_targets, n_samples) initial is the same for all targets, not for restarts thus could compute one for each initial value and just copy to correct locations in batch
        self.whitened_log_signal_variance = torch.empty((self.batch_size, self.n_samples)) 
        self.whitened_log_noise_variance = torch.empty((self.batch_size, self.n_samples)) 

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


    def compute_woodbury_matrix(self): # (n_targets, n_samples, n_samples)
        """Computes and caches the Woodbury matrix (Ktt_noisy⁻¹)."""
        kernel_with_noise = self.get_ns_rbf_kernel_with_noise # (n_targets, n_samples, n_samples)
        cholesky_kernel, info = torch.linalg.cholesky_ex(kernel_with_noise)
        alpha_vector = torch.cholesky_solve(self.batch_outputs.T.unsqueeze(2), cholesky_kernel, upper=False).squeeze(-1) # (n_targets, n_samples)
        return alpha_vector[:, :, None] * alpha_vector[:, None, :] - torch.cholesky_solve(self.batch_I, cholesky_kernel, upper=False)

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
            kernel = self.get_ns_rbf_kernel.clone()
            prev_noise = 1e-12
            noise_variance = self.noise_variance
            if type(noise_variance) == int:
                noise_variance = max(noise_variance, 1e-6) * torch.ones(kernel.shape[0], kernel.shape[1])
            # elif noise_variance.size == 1:  # If single float (inside np.array)
            #     noise_variance = max(noise_variance.item(), 1e-3) * torch.ones(kernel.shape[0])
            else:  # If it's a vector, add it to the diagonal
                noise_variance = torch.clamp(noise_variance, min=1e-6)
            idx = torch.arange(kernel.shape[1], device=kernel.device)
            kernel[:, idx, idx] += noise_variance**2 - prev_noise * torch.ones(kernel.shape[0], kernel.shape[1])
            self._ns_rbf_kernel_with_noise = kernel
        return self._ns_rbf_kernel_with_noise

    @property
    def log_lengthscale(self): # l_ell
        if self._log_lengthscale is None:
            self._log_lengthscale = self.whitened_log_lengthscale @ self.cholesky_lengthscale.T
        return self._log_lengthscale

    @property
    def log_signal_variance(self): # l_sigma
        if self._log_signal_variance is None:
            self._log_signal_variance = self.whitened_log_signal_variance @ self.cholesky_signal_variance.T
        return self._log_signal_variance

    @property
    def log_noise_variance(self): # l_omega
        if self._log_noise_variance is None:
            self._log_noise_variance = self.whitened_log_noise_variance @ self.cholesky_noise_variance.T
        return self._log_noise_variance

    @property
    def lengthscale(self): # ell
        if self._lengthscale is None:
            self._lengthscale = torch.exp(self.log_lengthscale)
        return self._lengthscale

    @property
    def signal_variance(self): # sigma
        if self._signal_variance is None:
            self._signal_variance = torch.exp(self.log_signal_variance)
        return self._signal_variance

    @property
    def noise_variance(self): # omega
        if self._noise_variance is None:
            self._noise_variance = torch.exp(self.log_noise_variance)
        return self._noise_variance
    
    @property
    def mean_log_lengthscale(self): # l_muell (batch_size)
        if self._mean_log_lengthscale is None:
            self._mean_log_lengthscale = self.log_lengthscale.mean(dim=1)
        return self._mean_log_lengthscale

    @property
    def mean_log_signal_variance(self): # l_musigma
        if self._mean_log_signal_variance is None:
            self._mean_log_signal_variance = self.log_signal_variance.mean(dim=1)
        return self._mean_log_signal_variance

    @property
    def mean_log_noise_variance(self): # l_muomega
        if self._mean_log_noise_variance is None:
            self._mean_log_noise_variance = self.log_noise_variance.mean(dim=1)
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
    

def nsgp(
    x: torch.Tensor, 
    y: torch.Tensor, 
    nonstationary_functions: str='lso', 
    optim: str='grad', 
    batch_size: int=1, 
    *args: dict, 
    **kwargs: dict
) -> object:
    """
    Main function to learn the adaptive Gaussian Process (NSGP).

    Parameters
    ----------
    x : torch.Tensor, shape (n_samples, n_features)
        Input data vector.
    y : torch.Tensor, shape (n_samples, n_targets)
        Output data vector.
    nonstationary_functions : str
        String specifying nonstationary components (default is 'lso').
    batch_size: int
        The number of concurrent processes
    optim : str
        Optimization method (default is 'grad').
    args/kwargs : dict
        Optional parameters for customization.

    Returns
    -------
    gp : object
        Learned GP model.
    """
    # Initialize GP model
    gp = GPModel(x, y, batch_size=batch_size) 

    # Set remaining arguments as attributes in the GP model
    for i in range(0, len(args), 2):
        setattr(gp, args[i], args[i+1])

    for key, value in kwargs.items():
        setattr(gp, key, value)

    gp.nonstationary_functions = nonstationary_functions
    gp.optimization_method = optim
    gp.initialize_kernels() # Perform necessary initializations (probably needs to be placed in nspgrad because is dependend on batch allocation) need if if parameters of kernels are changed

    # Initialize final GP
    gp_final = GPModel(x, y, batch_size=gp.n_targets) 

    # Set same arguments as attributes in the GP model
    for i in range(0, len(args), 2):
        setattr(gp_final, args[i], args[i+1])

    for key, value in kwargs.items():
        setattr(gp_final, key, value)

    gp_final.nonstationary_functions = nonstationary_functions
    gp_final.optimization_method = optim
    gp_final.initialize_kernels()

    # Perform optimization (gradient-based)
    gp, _ = nsgpgrad(gp, gp_final)

    return gp