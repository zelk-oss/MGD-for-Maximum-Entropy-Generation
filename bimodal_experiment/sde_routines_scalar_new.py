import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx, erf, erfinv
from scipy.integrate import trapezoid
from scipy import stats

from potentials_new import *

def sigt(t):
    """Time-dependent noise scaling function: (1-t)^2"""
    return (1 - t)

def compute_eta_t_partial(x0, x1, xt_i, It, t, i, potentials, device='cpu'):

    ## Exact --------------------------

    # Compute the drift term with regularization
    # Original drift components

    It_dot = x1 * .5 * torch.pi * torch.cos(.5 * torch.pi * t[i]) - x0 * .5 * torch.pi * torch.sin(.5 * torch.pi * t[i])

    rhs = torch.zeros(len(potentials), device=xt_i.device)
    for j, potential in enumerate(potentials):
        rhs[j] = torch.mean(It_dot * potential.grad(It))
    
    # Gradient matrix
    grad_mat = gradmat(xt_i, potentials)

    # Solve for eta_t
    eta_t_partial = torch.linalg.solve(grad_mat, rhs)

    return eta_t_partial, rhs

def constraint_correction(xt, It, potentials):
    output = torch.zeros(len(potentials), device=xt.device)
    
    for i, potential in enumerate(potentials):
        output[i] = torch.mean(potential(It)-potential(xt))

    return output

def iteration_step_projection(x0, x1, xt, n1, t, i, sigma, potentials, device='cpu'):

    h = t[i+1]-t[i]

    #It = (1 - t[i]) * x0 + t[i] * x1
    It = torch.cos(.5*torch.pi*t[i]) * x0 +  torch.sin(.5*torch.pi*t[i]) * x1
    
    # SDE update with drift and diffusion
        
    eta_t, dt_phi_It = compute_eta_t_partial(x0, x1, xt, It, t, i, potentials, device=device)

    drift = gradphi(xt, potentials) @ eta_t
    noise_scale = torch.sqrt(torch.tensor(2 * h * sigma))
    noise = noise_scale * torch.randn(n1).to(device)
    
    # First update step
    xt = xt + h * drift + noise
    
    # Update interpolation for next step
    #It = (1 - t[i + 1]) * x0 + t[i + 1] * x1
    It = torch.cos(.5*torch.pi*t[i+1]) * x0 +  torch.sin(.5*torch.pi*t[i+1]) * x1
    
    # Constraint correction
    rhs = constraint_correction(xt, It, potentials)
    
    # Gradient matrix
    grad_mat = gradmat(xt, potentials)


    etat2 = torch.linalg.solve(grad_mat, rhs)
    
    # Apply constraint correction
    xt = xt + gradphi(xt, potentials) @ etat2
    
    dH_t = -(etat2/(sigma*h))@dt_phi_It

    return xt, eta_t, etat2/(sigma*h), grad_mat, dH_t

def solve_sde(x1, n1, t, sigmas, potential_names=['x', 'x_abs', 'x2'], device='cpu', std_init=1, xt=None):

    nt = len(t)-1

    potentials = get_potentials(potential_names, device)
    
    # Initialize with Gaussian noise
    x0 = std_init*torch.randn(n1).to(device)
    
    # Storage for trajectories
    #xt = torch.zeros(n1, nt + 1).to(device)
    #xt[:, 0] = x0.squeeze()

    barphi_e = torch.zeros(nt + 1, len(potential_names))
    barphi_p = torch.zeros(nt + 1, len(potential_names))
    
    if xt==None:
        xt = x0.clone()
        barphi_e[0, :] = barphi(x0, potentials)
    else:
        barphi_e[0, :] = barphi(xt, potentials)   
    
    barphi_p[0, :] = barphi(xt, potentials)

    eta_t_list = []
    eta_t2_list = []

    dH_t_list = []
    ratio = []

    sigma = sigmas[0]
    
    for i in range(nt):
        if i % 200 == 0:
            print(f"Step {i}/{nt}")

        xt, etat_t, etat_t2, H, dH_t = iteration_step_projection(x0, x1, xt, n1, t, i, sigma, potentials, device=device)

        sigma = sigmas[i+1]
        #ratio.append(torch.sqrt((etat_t@H@etat_t)/(etat_t2@H@etat_t2.T)))
        
        eta_t_list.append(etat_t.cpu().detach())
        eta_t2_list.append(etat_t2.cpu().detach())

        dH_t_list.append(dH_t.cpu().detach().numpy())

        # Store statistics
        barphi_e[i + 1, :] = barphi(torch.cos(.5*torch.pi*t[i+1]) * x0 +  torch.sin(.5*torch.pi*t[i+1]) * x1, potentials) # barphi((1 - t[i + 1]) * x0 + t[i + 1] * x1, 0)
        barphi_p[i + 1, :] = barphi(xt, potentials)

    #plt.plot(ratio)
    #plt.show()
    
    return x0, xt, barphi_e, barphi_p, torch.stack([etat_t for etat_t in eta_t_list], dim=0), torch.stack([etat_t2 for etat_t2 in eta_t2_list], dim=0), dH_t_list


def get_potentials(potential_names, device):
    potentials = []

    if 'x' in potential_names:
        potentials.append(Identity())
        
    if 'x_abs' in potential_names:
        potentials.append(Abs())
        
    if 'x2' in potential_names:
        potentials.append(Squared())

    if 'x3' in potential_names:
        potentials.append(Third())

    if 'x3_modulus' in potential_names:
        potentials.append(Third_modulus())
        
    if 'x4' in potential_names:
        potentials.append(Quartic())

    if 'x5' in potential_names:
        potentials.append(Fifth())

    if 'x6' in potential_names:
        potentials.append(Sixth())

    if 'gaussian_mixture' in potential_names:
        potentials.append(Gaussian_mixture(device=device))

    if 'bimodal' in potential_names:
        potentials.append(Bimodal())
    
    return potentials


def plot_SD_results(x0, x1, xt, barphi_e, barphi_p, t, sigma, nt, potential_names):
    print("SDE interpolation complete!")

    # Plotting
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Final comparison (matches figure(1) in MATLAB)
    plt.subplot(1, 2, 1)
    It_final = (1 - t[-2]) * x0 + t[-2] * x1  # Using t[i] from last iteration
    plt.hist(It_final.cpu().numpy(), bins=100, density=True, alpha=0.7, label='Exact (It)', color='blue')
    plt.hist(xt.cpu().numpy(), bins=100, density=True, alpha=0.7, label='SDE Interpolant', color='orange')
    plt.legend()
    plt.title('Final Distributions (SDE)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature evolution (matches figure(3) in MATLAB)

    label_phi = []
    label_SDE = []

    for i in range(len(potential_names)):
        label_phi.append('Exact $\phi_' + str(i+1) + '$')
        label_SDE.append('SDE $\phi_' + str(i+1) + '$')
    
    plt.subplot(1, 2, 2)
    plt.plot(t.numpy(), barphi_e.numpy(), "--", linewidth=1, label=label_phi)
    plt.plot(t.numpy(), barphi_p.numpy(), "-",linewidth=1, label=label_SDE)
    plt.legend()
    plt.title('Feature Evolution (SDE)')
    plt.xlabel('Time t')
    plt.ylabel('Feature Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Additional analysis
    print(f"\nFinal Results:")
    print(f"Final feature error: {torch.norm(barphi_e[-1] - barphi_p[-1]):.6f}")
    print(f"Max feature error during interpolation: {torch.max(torch.norm(barphi_e - barphi_p, dim=1)):.6f}")

    # Show statistics of final distributions
    print(f"\nDistribution Statistics:")
    print(f"Target (x1) - Mean: {torch.mean(x1):.4f}, Std: {torch.std(x1):.4f}")
    print(f"Initial (x0) - Mean: {torch.mean(x0):.4f}, Std: {torch.std(x0):.4f}")
    print(f"Final SDE interpolant - Mean: {torch.mean(xt):.4f}, Std: {torch.std(xt):.4f}")
    
    return torch.norm(barphi_e[-1] - barphi_p[-1])

import torch
import matplotlib.pyplot as plt

def plot_moment_matching(barphi_e, barphi_p, t, threshold):
    # Move everything to CPU once to avoid repetitive .cpu() calls
    barphi_e = barphi_e.cpu()
    barphi_p = barphi_p.cpu()
    t = t.cpu()
    
    # 1. Use PyTorch native boolean masking instead of np.where
    keep_mask = barphi_e[-1] > threshold
    
    # Safety check: If nothing survives the threshold, we can't plot the time series
    if not keep_mask.any():
        print(f"Warning: No moments exceeded the threshold of {threshold}. Plotting fallback histogram.")
        # Calculate error for all moments just to show the fallback histogram
        error_last = (2 * (barphi_e - barphi_p).abs() / (barphi_e.abs() + barphi_p.abs()))[-1]
        plt.hist(error_last, bins=100)
        plt.title('Distribution of moment matching error (All Moments)')
        plt.yscale('log')
        plt.show()
        return

    # Filter tensors
    barphi_e = barphi_e[:, keep_mask]
    barphi_p = barphi_p[:, keep_mask]

    # Calculate the symmetric relative error matrix
    rel_error = 2 * (barphi_e - barphi_p).abs() / (barphi_e.abs() + barphi_p.abs())
    
    try:
        # 2. Fix the slicing mismatch. Let's slice both X and Y identically: from index 2 to the second-to-last index.
        t_sliced = t[2:-1]
        error_mean_sliced = rel_error.mean(dim=1)[2:-1]
        
        plt.plot(t_sliced, error_mean_sliced, marker='.')
        plt.xlabel('t')
        plt.yscale('log')
        plt.title('Relative moment matching error')
        plt.show()
        
    except Exception as e:
        print(f"Time-plot failed due to: {e}. Falling back to histogram.")
    
    # This will now run regardless of whether the first plot succeeded
    plt.hist(rel_error[-1], bins=100)
    plt.title('Distribution of moment matching error')
    plt.yscale('log')
    plt.show()