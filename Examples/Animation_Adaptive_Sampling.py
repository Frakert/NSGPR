import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from NSGPy import nsgp, plot_nsgp_1d, plot_nsgp_2d, nsgp_posterior, NSGP, plot_kernel_1d
from scipy.stats import gmean

# -----------------------------------
# Define the true underlying function
# -----------------------------------
def f(x1):
    return np.sin((x1 * 6) ** 4 * 0.01) + (1 - x1)

# -----------------------------------
# Denormalise for NSGP outputs
# -----------------------------------
def denormalise(gp, ft=0, ftstd=0, lt=0, st=0, ot=0):
    xtr = gp.normalized_inputs * gp.input_range + gp.input_min
    ytr = gp.normalized_outputs * gp.output_scale + gp.output_mean
    ft = ft * gp.output_scale + gp.output_mean
    ftstd = ftstd * gp.output_scale
    lt = lt * gmean(gp.input_range)
    ot = ot * gp.output_scale
    st = st * gp.output_scale
    return xtr, ytr, ft, ftstd, lt, st, ot

# -----------------------------------
# Sampling utilities
# -----------------------------------
def select_highest_uncertainty(model, X_candidates):
    _, std = model.predict(X_candidates, return_std=True)
    return X_candidates[np.argmax(std)]

def select_highest_uncertainty_nsgp(gp, x_range=(0.0, 1.0)):
    x_norm = np.linspace(((x_range[0] - gp.input_min) / gp.input_range).item(),
                         ((x_range[1] - gp.input_min) / gp.input_range).item(), 500)[:, None]
    ft, ftstd, *_ = nsgp_posterior(gp, x_norm)
    x_denorm = x_norm * gp.input_range + gp.input_min
    return x_denorm[np.argmax(ftstd)]


# -----------------------------------
# Initial data
# -----------------------------------
initial_Xt = np.array([0.3, 0.6, 0.9]).reshape(-1, 1)
initial_yt = f(initial_Xt)

Xt_dynamic_stat = initial_Xt.tolist()
yt_dynamic_stat = initial_yt.tolist()

Xt_dynamic_non_stat = initial_Xt.tolist()
yt_dynamic_non_stat = initial_yt.tolist()

Xt_true = np.linspace(0, 1, 500).reshape(-1, 1)
y_true1D = f(Xt_true)

# -----------------------------------
# GPR and NSGP setup
# -----------------------------------
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

hyperparameters = {
    "init_lengthscale": 0.05,
    "init_signal_variance": 0.3,
    "init_noise_variance": 0.05,
    "beta_lengthscale": 0.2,
    "beta_signal_variance": 0.2,
    "beta_noise_variance": 0.3,
    "alpha_lengthscale": 1,
    "alpha_signal_variance": 1,
    "alpha_noise_variance": 1,
    "verbose_output": False,
}

# -----------------------------------
# Plot setup
# -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
axs[0].set_title("GPR (Stationary)")
axs[1].set_title("NSGP (Non-stationary)")

# -----------------------------------
# Init function
# -----------------------------------
def init():
    axs[0].cla()
    axs[1].cla()
    axs[0].plot(Xt_true, y_true1D, '--', color='black', label="True function")
    axs[1].plot(Xt_true, y_true1D, '--', color='black', label="True function")
    return []

# -----------------------------------
# Animation update function
# -----------------------------------
def update(frame):
    global Xt_dynamic_stat, yt_dynamic_stat
    global Xt_dynamic_non_stat, yt

    Xt_array_stat = np.array(Xt_dynamic_stat).reshape(-1, 1)
    yt_array_stat = np.array(yt_dynamic_stat).reshape(-1, 1)

    Xt_array_non_stat = np.array(Xt_dynamic_non_stat).reshape(-1, 1)
    yt_array_non_stat = np.array(yt_dynamic_non_stat).reshape(-1, 1)
    

    # --- GPR ---
    gaussian_process.fit(Xt_array_stat, yt_array_stat)
    mean_pred, std_pred = gaussian_process.predict(Xt_true, return_std=True)
    next_point_gpr = select_highest_uncertainty(gaussian_process, Xt_true)

    # --- NSGP ---
    lgp = NSGP()
    lgp.set_params(**hyperparameters)
    lgp.optimizer(optimizer="grad", random_restarts=3, max_iteration=1000)
    lgp.fit(Xt_array_non_stat, yt_array_non_stat, "ls")
    gp = lgp.gp
    xt = np.linspace(((Xt_true.min() - gp.input_min) / gp.input_range).item(),
                     ((Xt_true.max() - gp.input_min) / gp.input_range).item(), 250)[:, None]
    ft, ftstd, lt, st, ot = nsgp_posterior(gp, xt)
    xt_denorm = xt * gp.input_range + gp.input_min
    xtr, ytr, *_ = denormalise(gp)
    next_point_nsgp = select_highest_uncertainty_nsgp(gp)

    # Add new point (only GPR for simplicity here; could alternate)
    Xt_dynamic_stat.append([next_point_gpr.item()])
    yt_dynamic_stat.append([f(np.array([[next_point_gpr]])).item()])

    Xt_dynamic_non_stat.append([next_point_nsgp.item()])
    yt_dynamic_non_stat.append([f(np.array([[next_point_nsgp]])).item()])

    # --- Plotting ---
    axs[0].cla()
    axs[1].cla()

    axs[0].fill_between(Xt_true.ravel(), (mean_pred - 1.96 * std_pred).ravel(),
                        (mean_pred + 1.96 * std_pred).ravel(), alpha=0.5, color='steelblue')
    axs[0].plot(Xt_true, mean_pred, color='steelblue', label='Posterior mean')
    axs[0].scatter(Xt_array_stat, yt_array_stat, color='red')
    axs[0].plot(Xt_true, y_true1D, '--', color='black', label="True function")

    axs[1].fill_between(xt_denorm.ravel(), ft.ravel() - 2 * ftstd, ft.ravel() + 2 * ftstd,
                        alpha=0.45, color='steelblue')
    axs[1].plot(xt_denorm.ravel(), ft.ravel(), color='steelblue', label='Posterior mean')
    axs[1].scatter(xtr, ytr, color='red')
    axs[1].plot(Xt_true, y_true1D, '--', color='black', label="True function")

    axs[0].set_title("Stationary GP")
    axs[1].set_title("Non-stationary GP")

    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(min(y_true1D) - 1, max(y_true1D) + 1)
        ax.grid(True)
        ax.legend()

    return []

# -----------------------------------
# Run animation
# -----------------------------------
ani = animation.FuncAnimation(fig, update, frames=15, init_func=init, blit=False, repeat=False)
ani.save("gpr_vs_nsgp_active_learning.mp4", fps=1, dpi=200)
plt.show()