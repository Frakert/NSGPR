import torch
import numpy as np
import matplotlib.pyplot as plt

from ..inference.posterior import nsgp_posterior
from .preprocessing import denormalise
from ..kernels.nonstationary import ns_rbf_kernel
from ..optimization.gradients import lengthscale_gradient, signal_variance_gradient, noise_variance_gradient

def cpu_copy(tens: torch.Tensor) -> torch.tensor:
    """Creates a cpu copy, such that the original remains unchanged."""
    return tens.detach().clone().to("cpu")

def plot_nsgp_2d(gp):
    """
    Plots the GP model.
    
    Parameters
    ----------
    gp : object
        GP model instance.
    """
    # Grid parameters
    ng = 50
    nl = 100  # Number of contour levels

    # Create mesh grid
    x1 = torch.linspace(0, 1, ng).to(gp.device)
    x2 = torch.linspace(0, 1, ng).to(gp.device)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    Xt = torch.column_stack((X1.ravel(), X2.ravel()))

    # Compute posterior values
    ft, ftstd, lt, st, ot = nsgp_posterior(gp, Xt)

    # create cpu variants
    if gp.device != "cpu":
        ft, ftstd, lt, st, ot = map(cpu_copy, (ft, ftstd, lt, st, ot))
        X1, X2 = map(cpu_copy, (X1, X2))
    norm_inputs = cpu_copy(gp.normalized_inputs)


    # Define variables and check their existence in the GP model
    top_row_vars = {
        "E[f]": ft,
        "Var[f]": ftstd
    }
    
    bottom_row_vars = {
        "Lengthscale": lt if "l" in gp.nonstationary_functions else None,
        "Signal variance": st if "s" in gp.nonstationary_functions else None,
        "Noise variance": ot if "o" in gp.nonstationary_functions else None,
    }

    for n in range(ft.shape[0]): # itterate over n_targets

        # Filter out None values
        bottom_row_vars = {name: data for name, data in bottom_row_vars.items() if data is not None}

        # Determine subplot layout
        n_top = len(top_row_vars)
        n_bottom = len(bottom_row_vars)
        n_cols = max(n_top, n_bottom)
        n_rows = 1 + (1 if n_bottom > 0 else 0)

        # Create figure and axes
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
        axs = np.atleast_2d(axs)

        # Plot top row variables (E[f] and Var[f])
        for ax, (name, data) in zip(axs[0, :n_top], top_row_vars.items()):
            cf = ax.contourf(X1, X2, data[n].reshape(ng, ng), nl, cmap='viridis')
            ax.set_title(name)
            fig.colorbar(cf, ax=ax)

        # Plot bottom row variables (ell, sigma, omega) if they exist
        if n_bottom > 0:
            for ax, (name, data) in zip(axs[1, :n_bottom], bottom_row_vars.items()):
                cf = ax.contourf(X1, X2, data[n].reshape(ng, ng), nl, cmap='viridis')
                ax.set_title(name)
                fig.colorbar(cf, ax=ax)

        # Delete axis of top right plot when 3 plots at lowest row
        if n_bottom > 2:
            axs[0, 2].axis("off")
        
        axs[0, 0].scatter(norm_inputs[:, 0], norm_inputs[:, 1], c='k', s=20, label='Training Data')

        plt.tight_layout()
        plt.show()


def plot_nsgp_1d(gp, plotlatent=False, plotderivs=False, truemodel=None):
    """
    Plots the GP model.
    
    Parameters
    ----------
    gp : object
        GP model instance.
    plotlatent : bool, optional
        Plot latent functions (lengthscale, signal variance, noise variance).
    plotderivs : bool, optional
        Plot derivative GP.
    plotkernel : bool, optional
        Plot covariance matrix.
    truemodel : object, optional
        Highlight true function.
    """
    if gp.normalized_inputs.shape[1] == 2:
        plot_nsgp_2d(gp)
        return
    
    squares = 1 + plotlatent + plotderivs
    cols = np.array([[248, 118, 109], [0, 186, 56], [97, 156, 255]]) / 255

    if truemodel:
        xt = torch.linspace(((min(truemodel.x) - gp.input_min) / gp.input_range).item(), ((max(truemodel.x) - gp.input_min) / gp.input_range).item(), 250)[:, None].to(gp.device)
    else:
        xt = torch.linspace(-0.1, 1.1, 250)[:, None].to(gp.device)
    

    ft, ftstd, lt, st, ot = nsgp_posterior(gp, xt)
    xt = xt * gp.input_range + gp.input_min
    
    xtr, ytr, *_ = denormalise(gp)

    # Placed here to get all the computations in 1 place
    if plotderivs:
        dl_l = gp.cholesky_lengthscale @ lengthscale_gradient(gp)
        dl_s = gp.cholesky_signal_variance @ signal_variance_gradient(gp)
        dl_o = gp.cholesky_noise_variance @ noise_variance_gradient(gp)

    # create cpu variants
    if gp.device != "cpu":
        ft, ftstd, lt, st, ot = map(cpu_copy, (ft, ftstd, lt, st, ot))
        xt, xtr, ytr = map(cpu_copy, (xt, xtr, ytr))
        truemodel.to("cpu")

    for n in range(ft.shape[0]): # itterate over n_targets

        plt.figure(figsize=(8, 6))
        
        if squares > 1:
            plt.subplot(squares, 1, 1)

        plt.plot(xt, ft[n], color='steelblue', label='Posterior mean')
        

        plt.fill_between(xt.flatten(), ft[n] - 2 * ftstd[n], ft[n] + 2 * ftstd[n], color='steelblue', alpha=0.45)
        plt.fill_between(xt.flatten(), ft[n] - 2 * torch.sqrt(ftstd[n]**2 + ot[n]**2),
                        ft[n] + 2 * torch.sqrt(ftstd[n]**2 + ot[n]**2), color='steelblue', alpha=0.2)
        
        plt.scatter(xtr, ytr[:, n], color='red', s=20, label='Data')

        if truemodel:
            plt.plot(truemodel.x, truemodel.f[:,n], '--', color='black', label='True Function')
            plt.xlim(min(truemodel.x), max(truemodel.x))
        else:
            plt.xlim(min(xt), max(xt))
        
        if plotlatent:
            plt.subplot(squares, 1, 2)
            plt.plot(xt, lt[n], color=cols[0], label='Lengthscale')
            plt.plot(xt, st[n], color=cols[1], label='Signal variance')
            plt.plot(xt, ot[n], color=cols[2], label='Noise variance')
            plt.xlabel('x')
            plt.ylabel('Value')
            plt.ylim(0, 1.1 * max(torch.max(st[n]), torch.max(lt[n]), torch.max(ot[n])))
            plt.legend()
            plt.title('Parameters')
            plt.grid()
            if truemodel:
                plt.xlim(min(truemodel.x), max(truemodel.x))
            else:
                plt.xlim(min(xt), max(xt))
        
        if plotderivs:
            plt.subplot(squares, 1, 3)
            plt.stem(xtr, dl_o, linefmt=cols[2], markerfmt='o', label='Noise Variance Derivative')
            plt.stem(xtr, dl_s, linefmt=cols[1], markerfmt='o', label='Signal Variance Derivative')
            plt.stem(xtr, dl_l, linefmt=cols[0], markerfmt='o', label='Lengthscale Derivative')
            plt.title('Latent Derivatives')
            plt.legend()
            plt.grid()
            if truemodel:
                plt.xlim(min(truemodel.x), max(truemodel.x))
            else:
                plt.xlim(min(xt), max(xt))

        plt.show()


def plot_kernel_1d(gp):

    xt = torch.linspace(0, 1, 250)[:, None].to(gp.device)
    
    ft, ftstd, lt, st, ot = nsgp_posterior(gp, xt)
    xt = xt * gp.input_range + gp.input_min

    K = ns_rbf_kernel(xt, xt, lt / gp.output_scale.unsqueeze(1), lt / gp.output_scale.unsqueeze(1),
                        st / gp.output_scale.unsqueeze(1), st / gp.output_scale.unsqueeze(1), ot / gp.output_scale.unsqueeze(1))
    K_cpu = cpu_copy(K)
    for n in range(ft.shape[0]): # itterate over n_targets
        plt.imshow(K_cpu[n], cmap='hot', interpolation='nearest')
        plt.title('Kernel Matrix')
        plt.colorbar()
        plt.show()