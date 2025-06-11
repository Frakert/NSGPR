import matplotlib.animation as animation

from NSGPy import nsgp, plot_nsgp_1d, plot_nsgp_2d, nsgp_posterior, NSGP, plot_kernel_1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gmean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class TrueModel:
    def __init__(self, x, f):
        self.x = x
        self.f = f

def f_noisy(x1):
    """Generates a noisy function with added random noise."""
    return np.sin((x1 * 6) ** 4 * 0.01) + (1 - x1) + (np.random.random(x1.shape) - 0.5) * 0.5 * (1 - x1)

def f(x1):
    """Defines a smooth function without noise."""
    return np.sin((x1 * 6) ** 4 * 0.01) + (1 - x1)

def denormalise(gp, ft=0, ftstd=0, lt=0, st=0, ot=0):
    """
    Denormalises the given parameters using the stored normalization factors in 'pars'.
    
    Parameters
    ----------
    gp : object
        GP model instance containing normalisation parameters
    ft : 
        Function values to denormalise (default 0)
    ftstd : 
        Function standard deviation to denormalise (default 0)
    lt : 
        Lengthscale posterior mean (default 0)
    st : 
        Signal variance posterior mean (default 0)
    ot : 
        Noise variance posterior mean (default 0)

    Returns
    -------
    xtr : 
        Denormalised training inputs
    ytr : 
        Denormalised training outputs
    ft : 
        Denormalised function values
    ftstd : 
        Denormalised function standard deviations
    lt : 
        Denormalised lengthscale
    st : 
        Denormalised signal variance
    ot : 
        Denormalised noise variance
    """
    # Denormalisation
    xtr = gp.normalized_inputs * gp.input_range + gp.input_min
    ytr = gp.normalized_outputs * gp.output_scale + gp.output_mean
    ft = ft * gp.output_scale + gp.output_mean
    ftstd = ftstd * gp.output_scale
    lt = lt * gmean(gp.input_range)
    ot = ot * gp.output_scale
    st = st * gp.output_scale

    return xtr, ytr, ft, ftstd, lt, st, ot


#==========================================
# Data Generation
#==========================================
#Xt = np.linspace(0, 1, 19).reshape(-1,1)  # 10 evenly spaced points between 0 and 1
Xt = np.array([0.1, 0.25, 0.98, 0.37, 0.63, 0.81, 0.5, 0.75,  0.9,]).reshape(-1,1)  # Custom points for training
y_pred1D = f(Xt).reshape(-1,1)  # Compute corresponding function values

# Generate test data for true function visualization (optional)
Xt_true = np.linspace(0, 1, 100).reshape(-1,1)  # 100 test points for smooth function visualization
y_true1D = f(Xt_true).reshape(-1,1)  # Compute true function values
truemodel = TrueModel(Xt_true, y_true1D)  # Store true model data for comparison

# ========================================== 
# Gaussian Process Regression Setup
# ==========================================
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# Define hyperparameters for the NSGP model (optional)
hyperparameters = {
    "init_lengthscale": 0.05,      # Initial lengthscale for the kernel
    "init_signal_variance": 0.3,   # Initial signal variance
    "init_noise_variance": 0.05,   # Initial noise variance
    "beta_lengthscale": 0.2,       # Controls variation of lengthscale
    "beta_signal_variance": 0.2,   # Controls variation of signal variance
    "beta_noise_variance": 0.3,    # Controls variation of noise variance
    "alpha_lengthscale": 1,        # Alpha parameter for lengthscale
    "alpha_signal_variance": 1,    # Alpha parameter for signal variance
    "alpha_noise_variance": 1,     # Alpha parameter for noise variance
    "verbose_output": False,
}


# Prepare figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Precompute true function for background
axs[0].plot(Xt_true, y_true1D, '--', color='black', label="True function")
axs[1].plot(truemodel.x, truemodel.f, '--', color='black', label="True function")

# Line and scatter handles
gpr_line, = axs[0].plot([], [], color='steelblue', label='Posterior mean')
gpr_fill = axs[0].fill_between([], [], [], color='steelblue', alpha=0.5)
gpr_scatter = axs[0].scatter([], [], color='red')

nsgp_line, = axs[1].plot([], [], color='steelblue', label='Posterior mean')
nsgp_fill = axs[1].fill_between([], [], [], color='steelblue', alpha=0.45)
nsgp_scatter = axs[1].scatter([], [], color='red')

# A function to initialize the empty frame
def init():
    # First 5 data points
    Xt_subset = Xt[:5]
    y_subset = y_pred1D[:5]

    # Fit GPR
    gaussian_process.fit(Xt_subset, y_subset)
    mean_pred, std_pred = gaussian_process.predict(Xt_true, return_std=True)

    # Fit NSGP
    lgp = NSGP()
    lgp.set_params(**hyperparameters)
    lgp.optimizer(optimizer="grad", random_restarts=3, max_iteration=1000)
    lgp.fit(Xt_subset, y_subset, "ls")
    gp = lgp.gp
    xt = np.linspace(((min(truemodel.x) - gp.input_min) / gp.input_range).item(),
                     ((max(truemodel.x) - gp.input_min) / gp.input_range).item(), 250)[:, None]
    ft, ftstd, lt, st, ot = nsgp_posterior(gp, xt)
    xt = xt * gp.input_range + gp.input_min
    xtr, ytr, *_ = denormalise(gp)

    # Plot initial state
    axs[0].cla()
    axs[0].fill_between(Xt_true.ravel(),
                        (mean_pred - 1.96 * std_pred).ravel(),
                        (mean_pred + 1.96 * std_pred).ravel(),
                        alpha=0.5, color='steelblue')
    gpr_line.set_data(Xt_true.ravel(), mean_pred.ravel())
    axs[0].scatter(Xt_subset.ravel(), y_subset.ravel(), color='red')

    axs[1].cla()
    axs[1].fill_between(xt.ravel(), ft.ravel() - 2*ftstd, ft.ravel() + 2*ftstd,
                        alpha=0.45, color='steelblue')
    nsgp_line.set_data(xt.ravel(), ft.ravel())
    axs[1].scatter(xtr.ravel(), ytr.ravel(), color='red')

    axs[0].plot(Xt_true, y_true1D, '--', color='black', label="True function")
    axs[1].plot(truemodel.x, truemodel.f, '--', color='black', label="True function")

    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(min(y_true1D)-1, max(y_true1D)+1)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(min(y_true1D)-1, max(y_true1D)+1)
    axs[1].grid(True)
    axs[1].legend()

    return gpr_line, nsgp_line

# Animation update function
def update(frame):
    # Start from index 5, include all points up to current frame
    i = frame + 5
    Xt_subset = Xt[:i+1]
    y_subset = y_pred1D[:i+1]

    # Fit GPR
    gaussian_process.fit(Xt_subset, y_subset)
    mean_pred, std_pred = gaussian_process.predict(Xt_true, return_std=True)

    # Fit NSGP
    lgp = NSGP()
    lgp.set_params(**hyperparameters)
    lgp.optimizer(optimizer="grad", random_restarts=3, max_iteration=1000)
    lgp.fit(Xt_subset, y_subset, "ls")
    gp = lgp.gp
    xt = np.linspace(((min(truemodel.x) - gp.input_min) / gp.input_range).item(),
                     ((max(truemodel.x) - gp.input_min) / gp.input_range).item(), 250)[:, None]
    ft, ftstd, lt, st, ot = nsgp_posterior(gp, xt)
    xt = xt * gp.input_range + gp.input_min
    xtr, ytr, *_ = denormalise(gp)

    # Update GPR plot
    axs[0].cla() # Clear fills
    gpr_line.set_data(Xt_true.ravel(), mean_pred.ravel())
    axs[0].fill_between(Xt_true.ravel(), 
                        (mean_pred - 1.96 * std_pred).ravel(), 
                        (mean_pred + 1.96 * std_pred).ravel(), 
                        alpha=0.5, color='steelblue')
    axs[0].scatter(Xt_subset.ravel(), y_subset.ravel(), color='red')

    # Update NSGP plot
    axs[1].cla()  # Clear fills
    nsgp_line.set_data(xt.ravel(), ft.ravel())
    axs[1].fill_between(xt.ravel(), ft.ravel() - 2*ftstd, ft.ravel() + 2*ftstd, alpha=0.45, color='steelblue')
    axs[1].scatter(xtr.ravel(), ytr.ravel(), color='red')

    axs[0].plot(Xt_true, y_true1D, '--', color='black', label="True function")
    axs[1].plot(truemodel.x, truemodel.f, '--', color='black', label="True function")

    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(min(y_true1D)-1, max(y_true1D)+1)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(min(y_true1D)-1, max(y_true1D)+1)
    axs[1].grid(True)
    axs[1].legend()

    return gpr_line, nsgp_line

# Final settings for layout
for ax in axs:
    ax.set_xlim(0, 1)
    ax.set_ylim(min(y_true1D)-1, max(y_true1D)+1)
    ax.legend(fontsize=12)
    ax.grid(True)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(Xt) - 5, init_func=init, blit=False, repeat=False)
# Optional: Save as mp4 or GIF
ani.save('gpr_vs_nsgp.mp4', fps=1, dpi=200)

plt.show()
