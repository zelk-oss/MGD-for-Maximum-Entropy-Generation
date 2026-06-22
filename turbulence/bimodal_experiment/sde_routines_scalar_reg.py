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
    
    # Predictor step: walkers after drift + noise (y_k)
    y_k = xt + h * drift + noise
    
    # Update interpolation for next step
    #It = (1 - t[i + 1]) * x0 + t[i + 1] * x1
    It = torch.cos(.5*torch.pi*t[i+1]) * x0 +  torch.sin(.5*torch.pi*t[i+1]) * x1
    
    # Constraint correction (raw moment mismatch b_k = phi_bar(I_{k+1}) - phi_bar(y_k))
    rhs = constraint_correction(y_k, It, potentials)
    
    # Gradient (Gram) matrix at the predicted walker: M_k = G(y_k)
    grad_mat = gradmat(y_k, potentials)


    etat2 = torch.linalg.solve(grad_mat, rhs)
    
    # Apply constraint correction (corrector step) -> x_{k+1}
    xt = y_k + gradphi(y_k, potentials) @ etat2
    
    dH_t = -(etat2/(sigma*h))@dt_phi_It

    # Also return the predicted walker y_k and the raw mismatch b_k so the caller
    # can assemble the time-regularised theta problem (see solve_sde / _solve_regularised).
    return xt, eta_t, etat2/(sigma*h), grad_mat, dH_t, y_k, rhs

def moments_matrix(x, potentials):
    """
    Per-sample potentials stacked column-wise: phi(x) of shape (N, r).

    Scalar analogue of SDE.compute_moments. Each potential is scalar-valued here
    (one coefficient), so potential(x) returns a length-N tensor; we reshape to be
    safe and stack along the potential axis.
    """
    return torch.stack([potential(x).reshape(-1) for potential in potentials], dim=1)


def _solve_regularised(t, M, Gf, bb, cc, lam, num_potentials, device='cpu', regularization=0.0):
    """
    Scalar (d = 1) port of SDE._solve_regularised.

    Assembles and solves the block-tridiagonal-in-time system

        (data)        M[k] Theta[k]
        (smoothness)  + lam/dt^2 * Gf[k] (Theta[k] - Theta[k+1])  (and the symmetric term)
                      = bb[k] + lam/dt * (cc terms)

    as one dense (n*r, n*r) linear solve, with diagonal (Jacobi) preconditioning
    mirroring compute_eta_t_partial / the corrector solve. At lam = 0 this reduces
    exactly to the per-step theta solve.
    """
    t = np.asarray(t, dtype=float)
    n, r, dev = len(t), num_potentials, device
    dt = np.diff(t)
    A = torch.zeros((n, r, n, r)).to(dev)
    f = torch.zeros((n, r)).to(dev)
    for k in range(n):
        A[k, :, k, :] += M[k]
        f[k]          += bb[k]
    for k in range(n - 1):
        w = lam / dt[k] ** 2
        A[k,     :, k,     :] += w * Gf[k]
        A[k + 1, :, k + 1, :] += w * Gf[k]
        A[k,     :, k + 1, :] -= w * Gf[k]
        A[k + 1, :, k,     :] -= w * Gf[k]
        f[k]     -= (lam / dt[k]) * cc[k]
        f[k + 1] += (lam / dt[k]) * cc[k]

    # --- diagonal (Jacobi) preconditioning, mirrors the per-step solves ---
    A_flat = A.reshape(n * r, n * r)
    f_flat = f.reshape(n * r)
    S = torch.diagonal(A_flat).clamp_min(1e-30).sqrt()        # per-(k, potential) scale
    A_flat = A_flat / (S[:, None] * S[None, :])
    A_flat = (A_flat + A_flat.T) / 2
    f_flat = f_flat / S
    if regularization:
        A_flat = A_flat + regularization * torch.eye(n * r, device=dev, dtype=A_flat.dtype)

    z = torch.linalg.solve(A_flat, f_flat)
    return (z / S).reshape(n, r)


def _solve_regularised_thomas(t, M, Gf, bb, cc, lam, num_potentials, device='cpu', eps_reg_theta=1e-6):
    """
    Scalar (d = 1) port of SDE._solve_regularised_thomas.

    Same block-tridiagonal system as _solve_regularised, solved via block-Thomas
    elimination with block-Jacobi preconditioning instead of a dense (n*r, n*r)
    solve. Memory: O(n * r**2) instead of O(n**2 * r**2). At lam = 0 this reduces
    exactly to the per-step preconditioned theta solve.
    """
    t = np.asarray(t, dtype=float)
    n, r, dev = len(t), num_potentials, device
    dt = np.diff(t)
    w  = [lam / dk ** 2 for dk in dt]

    D = [M[k].clone() for k in range(n)]
    for k in range(n - 1):
        D[k]     = D[k]     + w[k] * Gf[k]
        D[k + 1] = D[k + 1] + w[k] * Gf[k]
    U = [-w[k] * Gf[k] for k in range(n - 1)]
    L = [Uk.transpose(0, 1) for Uk in U]

    f = [bb[k].clone() for k in range(n)]
    for k in range(n - 1):
        f[k]     = f[k]     - (lam / dt[k]) * cc[k]
        f[k + 1] = f[k + 1] + (lam / dt[k]) * cc[k]

    # block-Jacobi preconditioning, mirrors the per-step solves
    S = [torch.diagonal(Dk).clamp_min(1e-30).sqrt() for Dk in D]
    for k in range(n):
        D[k] = D[k] / (S[k][:, None] * S[k][None, :])
        D[k] = (D[k] + D[k].T) / 2
        f[k] = f[k] / S[k]
    for k in range(n - 1):
        U[k] = U[k] / (S[k][:, None] * S[k + 1][None, :])
        L[k] = U[k].transpose(0, 1)

    eye = torch.eye(r, device=dev, dtype=D[0].dtype)
    c_prime, d_prime = [None] * max(n - 1, 0), [None] * n

    denom0 = D[0] + eps_reg_theta * D[0].diagonal().abs().mean() * eye
    if n > 1:
        sol0 = torch.linalg.solve(denom0, torch.cat([U[0], f[0][:, None]], dim=1))
        c_prime[0], d_prime[0] = sol0[:, :-1], sol0[:, -1]
    else:
        d_prime[0] = torch.linalg.solve(denom0, f[0])

    for k in range(1, n):
        denom = D[k] - L[k - 1] @ c_prime[k - 1]
        rhs   = f[k] - L[k - 1] @ d_prime[k - 1]
        denom = denom + eps_reg_theta * denom.diagonal().abs().mean() * eye
        if k < n - 1:
            sol = torch.linalg.solve(denom, torch.cat([U[k], rhs[:, None]], dim=1))
            c_prime[k], d_prime[k] = sol[:, :-1], sol[:, -1]
        else:
            d_prime[k] = torch.linalg.solve(denom, rhs)

    Theta_scaled = [None] * n
    Theta_scaled[-1] = d_prime[-1]
    for k in range(n - 2, -1, -1):
        Theta_scaled[k] = d_prime[k] - c_prime[k] @ Theta_scaled[k + 1]

    return torch.stack([Theta_scaled[k] / S[k] for k in range(n)])


def solve_sde(x1, n1, t, sigmas, potential_names=['x', 'x_abs', 'x2'], device='cpu', std_init=1, xt=None,
              lam=1.0, n_subsample=1, regularization=0.0, reg_eps=1e-8):

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

    num_potentials = len(potentials)

    eta_t_list = []
    theta_t_list = []

    dH_t_list = []
    ratio = []

    # --- regularised theta problem: accumulators for the block-tridiagonal system ---
    # Mirrors SDE.forward_regularised. Quantities are collected at the *predicted*
    # walker y_k (target time t[i+1]). With n_subsample > 1 the fine-step ingredients
    # are averaged into coarse blocks.
    M_blocks, Gf_blocks, bb_blocks, cc_blocks, t_used = [], [], [], [], []
    accM = accG = accb = accc = None
    cnt = 0
    adot = 0.5 * np.pi                       # d/dt of the Cos-schedule angle a_t = (pi/2) t

    sigma = sigmas[0]
    
    for i in range(nt):
        if i % 200 == 0:
            print(f"Step {i}/{nt}")

        h = t[i+1] - t[i]
        sigma_i = sigma                      # diffusion coefficient D used at this step

        xt, etat_t, etat_t2, H, dH_t, y_k, b_k = iteration_step_projection(
            x0, x1, xt, n1, t, i, sigma_i, potentials, device=device)

        # --- collect regularised-problem ingredients at the predicted walker y_k ---
        t_node = float(t[i + 1])
        a = adot * t_node
        cos_a, sin_a, tan_a = np.cos(a), np.sin(a), np.tan(a)

        if sin_a > reg_eps:                  # skip t = 0 (sin = 0); X = (X_t - cos a Z)/sin a
            mom  = moments_matrix(y_k, potentials)          # phi(y_k)              (N, r)
            M_k  = H                                        # G(y_k)  (raw Gram, = grad_mat)
            Gf_k = mom.T @ mom / n1                         # moment Gram           (r, r)
            # normalise the mismatch by h * D. Here D = sigma (noise var = 2 h sigma),
            # which is the scalar-file analogue of the class's 1/(h sigma**2).
            bb_k = b_k / (h * sigma_i)

            z2    = x0 ** 2                                  # ||Z||^2 per sample (d = 1)
            X_eff = (y_k - cos_a * x0) / sin_a              # reconstructed data endpoint X
            zx    = x0 * X_eff                              # Z . X per sample
            tau   = -adot * (tan_a * (1.0 - z2) + zx)       # tau_k^i               (N,)
            cc_k  = (mom * tau[:, None]).mean(0)            # E[phi(y_k) tau]       (r,)

            if cnt == 0:
                t_used.append(t_node)                       # coarse node at target time
                accM, accG = M_k.clone(), Gf_k.clone()
                accb, accc = bb_k.clone(), cc_k.clone()
            else:
                accM = accM + M_k; accG = accG + Gf_k
                accb = accb + bb_k; accc = accc + cc_k
            cnt += 1
            if cnt == n_subsample:
                M_blocks.append(accM / cnt);  Gf_blocks.append(accG / cnt)
                bb_blocks.append(accb / cnt); cc_blocks.append(accc / cnt)
                cnt = 0

        sigma = sigmas[i+1]
        #ratio.append(torch.sqrt((etat_t@H@etat_t)/(etat_t2@H@etat_t2.T)))
        
        eta_t_list.append(etat_t.cpu().detach())
        theta_t_list.append(etat_t2.cpu().detach())

        dH_t_list.append(dH_t.cpu().detach().numpy())

        # Store statistics
        barphi_e[i + 1, :] = barphi(torch.cos(.5*torch.pi*t[i+1]) * x0 +  torch.sin(.5*torch.pi*t[i+1]) * x1, potentials) # barphi((1 - t[i + 1]) * x0 + t[i + 1] * x1, 0)
        barphi_p[i + 1, :] = barphi(xt, potentials)

    if cnt > 0:                                             # final partial block
        M_blocks.append(accM / cnt);  Gf_blocks.append(accG / cnt)
        bb_blocks.append(accb / cnt); cc_blocks.append(accc / cnt)

    #plt.plot(ratio)
    #plt.show()

    # --- solve the time-regularised theta problem with both solvers ---
    # (same [1:] front-trimming as SDE.forward_regularised: drop the first coarse
    #  node before the solve, then drop the first row of the solution).
    theta_reg_t = _solve_regularised(
        t_used[1:], M_blocks[1:], Gf_blocks[1:], bb_blocks[1:], cc_blocks[1:],
        lam, num_potentials, device=device, regularization=regularization)
    #theta_reg_thomas_t = _solve_regularised_thomas(t_used[1:], M_blocks[1:], Gf_blocks[1:], bb_blocks[1:], cc_blocks[1:], lam, num_potentials, device=device)

    theta_reg_t        = theta_reg_t[1:].cpu().detach()
    #theta_reg_thomas_t = theta_reg_thomas_t[1:].cpu().detach()

    return (
        x0, xt, barphi_e, barphi_p,
        torch.stack(eta_t_list, dim=0),
        torch.stack(theta_t_list, dim=0),
        dH_t_list,
        theta_reg_t,
        #theta_reg_thomas_t,
    )


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