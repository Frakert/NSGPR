import torch
import numpy as np
from collections import deque

from ..metrics.likelihoods import nsgpmll
from .gradients import (
    lengthscale_gradient, 
    signal_variance_gradient, 
    noise_variance_gradient
)


def gradient_step(
    gp: object, 
    step: torch.Tensor, 
    mlls: torch.Tensor, 
    iterations: torch.Tensor
) -> torch.Tensor:
    """
    Perform a single gradient update on Gaussian Process (GP) hyperparameters.

    This function updates the lengthscale, signal variance, and noise variance 
    parameters using gradient descent, with overshooting prevention and adaptive 
    step size adjustment.

    Parameters
    ----------
    gp : object
        Gaussian Process object containing hyperparameters and optimization state.
    step : torch.Tensor, shape (batch_size)
        Current step sizes for each optimization process.
    mlls : torch.Tensor, shape (n_targets, n_restarts, max_itterations)
        Marginal Log Likelihood (MLL) values from previous iterations.
    iterations : torch.Tensor, shape (batch_size)
        Current iteration count for each optimization process.

    Returns
    -------
    new_mll : torch.Tensor, shape (batch_size)
        Updated Marginal Log Likelihood values after the gradient step.
    """
    # Compute gradients for different hyperparameters
    dwl_l = lengthscale_gradient(gp)
    dwl_s = signal_variance_gradient(gp)
    dwl_o = noise_variance_gradient(gp)

    # Store current parameter values for potential rollback
    l_cp = gp.whitened_log_lengthscale
    s_cp = gp.whitened_log_signal_variance
    o_cp = gp.whitened_log_noise_variance

    # Update latent function parameters with clamping
    gp.whitened_log_lengthscale = torch.clip(gp.whitened_log_lengthscale + step * dwl_l, min=-5, max=5)
    gp.whitened_log_signal_variance = torch.clip(gp.whitened_log_signal_variance + step * dwl_s, min=-5, max=5)
    # uncomment for adaptive noise variance
    # gp.whitened_log_noise_variance = torch.clip(gp.whitened_log_noise_variance + step * dwl_o, min=-np.inf, max=5)
    gp.reset_latent_variables()

    # Calculate new MLL
    new_mll = nsgpmll(gp)     

    # Overshooting Check
    overshoot_mask = new_mll < mlls[gp.which_target.long(), gp.which_restart.long(), iterations.long() - 1]

    # Revert updates where overshooting occurred
    gp.whitened_log_lengthscale[overshoot_mask] = l_cp[overshoot_mask]
    gp.whitened_log_signal_variance[overshoot_mask] = s_cp[overshoot_mask]
    # gp.whitened_log_noise_variance[overshoot_mask] = o_cp[overshoot_mask]
    gp.reset_latent_variables()

    # Restore old MLL values for overshot cases
    new_mll[overshoot_mask] = mlls[gp.which_target.long(), gp.which_restart.long(), iterations.long() - 1][overshoot_mask]

    # Adjust step sizes
    step[overshoot_mask] *= 0.7  # Reduce step for overshooting cases
    step[~overshoot_mask] *= 1.1  # Increase step for improving cases

    return new_mll


def start_new_process(
    process_id: int, 
    gp: object, 
    active_processes: list, 
    available_slots: deque, 
    step: torch.Tensor, 
    iterations: torch.Tensor
):
    """
    Initialize a new optimization process for a specific target and restart.

    Parameters
    ----------
    process_id : int
        Unique identifier for the optimization process.
    gp : object
        Gaussian Process object containing optimization configuration.
    active_processes : list
        List tracking currently active optimization processes.
    available_slots : deque
        Queue of available slots for new processes.
    step : torch.Tensor
        Step sizes for gradient optimization.
    iterations : torch.Tensor
        Iteration counters for each process.
    """
    slot_idx = available_slots.popleft()
    step[slot_idx] = 1e-5
    iterations[slot_idx] = 1
    
    # Determine target and restart indices
    restart, target = divmod(process_id, gp.n_targets)

    # Initialization of hyperparameters
    gp.whitened_log_lengthscale[slot_idx] = torch.linalg.solve_triangular(
        gp.cholesky_lengthscale, 
        np.log(gp.initial_lengthscale[restart]) * torch.ones(1, gp.n_samples).unsqueeze(2).to(gp.cholesky_lengthscale.device), 
        upper=False
    ).squeeze(-1)
    gp.whitened_log_signal_variance[slot_idx] = torch.linalg.solve_triangular(
        gp.cholesky_signal_variance, 
        np.log(gp.initial_signal_variance[restart]) * torch.ones(1, gp.n_samples).unsqueeze(2).to(gp.cholesky_signal_variance.device), 
        upper=False
    ).squeeze(-1)
    gp.whitened_log_noise_variance[slot_idx] = torch.linalg.solve_triangular(
        gp.cholesky_noise_variance, 
        np.log(gp.initial_noise_variance[restart]) * torch.ones(1, gp.n_samples).unsqueeze(2).to(gp.cholesky_noise_variance.device), 
        upper=False
    ).squeeze(-1)

    # Set process-specific configuration
    gp.batch_outputs[:, slot_idx] = gp.normalized_outputs[:, target]
    gp.which_target[slot_idx] = target
    gp.which_restart[slot_idx] = restart
    active_processes[slot_idx] = process_id

    if gp.verbose_output:
        print(f"Starting Process {process_id} in slot {slot_idx}")


def remove_converged_process(
    process_id: int, 
    gp: object, 
    gp_final: object, 
    active_processes: list, 
    available_slots: deque, 
    mlls_final: torch.Tensor, 
    mll: torch.Tensor
):
    """
    Handle the completion of a converged optimization process.

    Updates the final GP model with the best hyperparameters and frees up 
    the optimization slot for a new process.

    Parameters
    ----------
    process_id : int
        Unique identifier for the completed optimization process.
    gp : object
        Current Gaussian Process object.
    gp_final : object
        Final Gaussian Process model to store best hyperparameters.
    active_processes : list
        List tracking currently active optimization processes.
    available_slots : deque
        Queue of available slots for new processes.
    mlls_final : torch.Tensor
        Final Marginal Log Likelihood values for each target.
    mll : torch.Tensor
        Current Marginal Log Likelihood values.
    """
    slot_idx = active_processes.index(process_id)
    target = process_id % gp.n_targets

    # Update final GP if current MLL is better
    if mlls_final[target] < mll[slot_idx]:
        mlls_final[target] = mll[slot_idx]
        gp_final.whitened_log_lengthscale[target] = gp.whitened_log_lengthscale[slot_idx]
        gp_final.whitened_log_signal_variance[target] = gp.whitened_log_signal_variance[slot_idx]
        gp_final.whitened_log_noise_variance[target] = gp.whitened_log_noise_variance[slot_idx]

    # Free up the optimization slot
    active_processes[slot_idx] = None
    available_slots.append(slot_idx)


def nsgpgrad(gp: object, gp_final: object) -> object:
    """
    Learn a nonstationary Gaussian Process (NSGP) using parallelized gradient optimization.

    Parameters
    ----------
    gp : object
        Contains necessary kernel parameters and data.
    gp_final : object
        Stores the final optimized hyperparameters.
    
    Returns
    -------
    gp_final : object
        Optimized NSGP model.
    mll_final : torch.Tensor
        The maximum MLL score.
    """
    if gp.verbose_output:
        print(f"Optimizing {gp.n_targets} targets with {gp.n_restarts} restarts "
              f"using {gp.batch_size} parallel processes...")
    
    # Configuration parameters
    batch_size = gp.batch_size
    max_processes = gp.n_restarts * gp.n_targets
    initial_step = 1e-5
    device = gp.device

    # Initialization of optimization state
    step = torch.full((batch_size, 1), initial_step, device=device)
    iterations = torch.full((batch_size,), 1, device=device)
    active_processes = [None] * batch_size
    available_slots = deque(range(batch_size))
    
    # Track Marginal Log Likelihood (MLL) values
    mlls = torch.full((gp.n_targets, gp.n_restarts, gp.gradient_iterations), -np.inf, device=device)
    mlls_final = torch.full((gp.n_targets,), -np.inf, device=device)
    
    # Start initial batch of processes
    current_process_id = 0
    running_processes = min(batch_size, max_processes)
    for _ in range(running_processes):
        start_new_process(current_process_id, gp, active_processes, available_slots, step, iterations)
        current_process_id += 1

    active_mask = torch.tensor([p is not None for p in active_processes], device=device)
        
    while current_process_id < max_processes or any(active_processes):
        iterations[active_mask] += 1
        mll = gradient_step(gp, step, mlls, iterations)

        # Identify active slots
        active_mask = torch.tensor([p is not None for p in active_processes], device=device)
        active_slots = torch.arange(batch_size, device=device)[active_mask]

        active_process_ids = torch.tensor([p if p is not None else -1 for p in active_processes], device=device)
        restart_ids = torch.div(active_process_ids, gp.n_targets, rounding_mode='floor')
        target_ids = torch.remainder(active_process_ids, gp.n_targets)

        # Update MLL values
        mlls[target_ids[active_mask], restart_ids[active_mask], iterations[active_mask].long()] = mll[active_mask]

        # Compute convergence conditions
        step_too_small = torch.log10(step[active_mask]) < -7
        max_iter_reached = iterations[active_mask] > 50
        improvement_too_small = torch.abs(
            mlls[target_ids[active_mask], restart_ids[active_mask], iterations[active_mask].long()] -
            mlls[target_ids[active_mask], restart_ids[active_mask], (iterations[active_mask] - 30).long()]
        ) < 0.1
        maximum_iter_reached = iterations[active_mask] >= gp.gradient_iterations - 1

        converged_mask = step_too_small.view(-1) | (max_iter_reached & improvement_too_small) | maximum_iter_reached.view(-1)
        converged_slots = active_slots[converged_mask]

        # Process converged slots
        for slot_idx in converged_slots:
            process_id = active_processes[slot_idx]

            remove_converged_process(process_id, gp, gp_final, active_processes, available_slots, mlls_final, mll)

            if gp.verbose_output:
                print(f"Process {process_id} converged after {iterations[slot_idx]} iterations with MLL of {mll[slot_idx]}, freeing slot {slot_idx}")

            # Start a new process if available
            if current_process_id < max_processes:
                start_new_process(current_process_id, gp, active_processes, available_slots, step, iterations)
                current_process_id += 1
            else:
                step[slot_idx] = 0  # Ensure positive semi-definiteness

    if gp.verbose_output:
        print(f"Best model MLL: {mlls_final}")
    
    return gp_final, mlls_final