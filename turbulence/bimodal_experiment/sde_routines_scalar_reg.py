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

# ======================================================================================
# Regularised (temporally-smoothed) corrector  --  transposed from sde_routines.py
# ======================================================================================
# This is the scalar counterpart of `SDE.regularised_theta` / `regularised_theta_thomas`
# in sde_routines.py. It does NOT replace the per-step corrector `theta_t` computed inside
# `iteration_step_projection` (that one is kept exactly as is). Instead it builds, over the
# WHOLE time grid at once, a single block-tridiagonal system
#
#       (M_k + coupling) Theta = b_k + coupling      (section 4 of the MGD paper)
#
# whose solution `theta_regularised_t` is a temporally-SMOOTHED corrector: the weight `lam`
# couples neighbouring time nodes through the moment-Gram blocks `Gf_k`, penalising
# fast variation of Theta in time. `lam -> 0` decouples the system into the per-node
# solves `M_k Theta_k = b_k`; larger `lam` smooths harder.
#
# Everything is evaluated on the INTERPOLANT I_k = cos(a) x0 + sin(a) x1 (Cos schedule),
# so only (x0, x1, t, potentials) are needed -- no walker trajectory required, matching the
# standalone `regularised_theta(lam, t)` in the reference.
#
# Notes / caveats (mirror the reference):
#   * Cos schedule only (uses cos/sin/tan of a = pi t / 2 and adot = pi/2).
#   * The last node t = 1 is dropped (cos = 0 would divide by zero in b_k).
#   * Scalar signal => ambient dimension d = 1, so ||Z||^2 = x0^2 and Z.X = x0*x1.
#   * Normalisation: like the reference's interpolant-based `regularised_theta`, the result
#     is a RATE-like coefficient and is directly comparable to the normalised per-step
#     `theta_t = etat2/(sigma*h)` (NOT to the raw etat2). No sigma/h enters here.
#   * Time alignment: row k of the returned array sits at node t[k] (k = 0 .. n_kept-1),
#     whereas the per-step theta_t[i] targets t[i+1]. Shift by one if you overlay them.
# --------------------------------------------------------------------------------------

def _assemble_regularised_blocks(x0, x1, t, potentials, regularization, device):
    """Build the per-node blocks (M_k, Gf_k, b_k, c_k) of the regularised system.

    Returns the trimmed time grid `t` (numpy, last node dropped) plus four python lists,
    one entry per retained node. All matrices are float64 for a stable solve.

        M_k  : (r, r) gradient Gram of the potentials at I_k  (+ regularization * I)
        Gf_k : (r, r) moment Gram  moments_k^T moments_k / B   (the temporal-coupling block)
        b_k  : (r,)   E_n[ grad phi(I_k) . x0 ] / cos_k        (space target)
        c_k  : (r,)   E_n[ phi(I_k) * tau_k ]                  (time target)
    """
    # --- time grid: accept torch or numpy, drop the t = 1 endpoint (cos = 0) ----------
    t = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.isclose(t[-1], 1.0):
        t = t[:-1]

    r = len(potentials)
    B = x0.shape[0]
    d = int(np.prod(x0.shape[1:])) if x0.dim() > 1 else 1   # scalar => d = 1
    eye = torch.eye(r, dtype=torch.float64, device=device)

    # per-sample geometric quantities (fixed across nodes); Z = x0, X = x1
    z2 = x0.reshape(B, -1).double().pow(2).sum(1)           # ||Z||^2  (B,)
    zx = (x0 * x1).reshape(B, -1).double().sum(1)           # Z . X    (B,)
    adot = np.pi / 2.0                                      # d alpha / dt for the Cos schedule

    M, Gf, bb, cc = [], [], [], []
    for tk in t:
        ak = np.pi * tk / 2.0
        cos_k, tan_k = np.cos(ak), np.tan(ak)

        I_k = np.cos(ak) * x0 + np.sin(ak) * x1                                   # (B,)
        moments = torch.stack([p(I_k)      for p in potentials], dim=1).double()  # (B, r)
        grads   = torch.stack([p.grad(I_k) for p in potentials], dim=1).double()  # (B, r)

        M.append(gradmat(I_k, potentials).double() + regularization * eye)        # (r, r)
        Gf.append(moments.T @ moments / B)                                        # (r, r)
        bb.append((grads * x0.reshape(B, 1).double()).mean(0) / cos_k)            # (r,)

        tau = -adot * (tan_k * (d - z2) + zx)                                     # (B,)
        cc.append((moments * tau.reshape(B, 1)).mean(0))                          # (r,)

    return t, M, Gf, bb, cc


def regularised_theta(x0, x1, t, potentials, lam=1.0, regularization=0.0, device='cpu'):
    lam = 0
    """Dense solve of the block-tridiagonal regularised-corrector system.

    Memory O((n*r)^2): only practical for small grids / verification. For full grids use
    `regularised_theta_thomas` (identical solution, O(n*r^2) memory). Returns (n_kept, r).
    """
    t_arr, M, Gf, bb, cc = _assemble_regularised_blocks(x0, x1, t, potentials, regularization, device)
    n, r = len(t_arr), len(potentials)
    dt = np.diff(t_arr)

    A = torch.zeros((n, r, n, r), dtype=torch.float64, device=device)
    f = torch.zeros((n, r),       dtype=torch.float64, device=device)
    for k in range(n):                              # diagonal blocks + space target
        A[k, :, k, :] += M[k]
        f[k]          += bb[k]
    for k in range(n - 1):                          # nearest-neighbour temporal coupling
        w = lam / dt[k] ** 2
        A[k,     :, k,     :] += w * Gf[k]
        A[k + 1, :, k + 1, :] += w * Gf[k]
        A[k,     :, k + 1, :] -= w * Gf[k]
        A[k + 1, :, k,     :] -= w * Gf[k]
        f[k]     -= (lam / dt[k]) * cc[k]
        f[k + 1] += (lam / dt[k]) * cc[k]

    Theta = torch.linalg.solve(A.reshape(n * r, n * r), f.reshape(n * r)).reshape(n, r)
    return Theta.to(x1.dtype)


def regularised_theta_thomas(x0, x1, t, potentials, lam=1.0, regularization=0.0,
                             eps_reg_theta=1e-6, device='cpu'):
    """Block-tridiagonal (block-Thomas) solve of the same system as `regularised_theta`.

    Memory O(n*r^2) instead of O((n*r)^2), so it runs at full grid resolution. A tiny
    scale-aware jitter `eps_reg_theta * mean|diag|` is added to each pivot for robustness
    against near-singular Gram blocks (set eps_reg_theta=0 to recover the exact dense
    solution -- verified to agree to ~1e-10). Returns (n_kept, r).
    """
    lam = 0
    t_arr, M, Gf, bb, cc = _assemble_regularised_blocks(x0, x1, t, potentials, regularization, device)
    n, r = len(t_arr), len(potentials)
    dt = np.diff(t_arr)
    w  = [lam / dk ** 2 for dk in dt]
    eye = torch.eye(r, dtype=torch.float64, device=device)

    # Block tridiagonal: diagonal D, super-diagonal U, sub-diagonal L = U^T.
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

    # Forward elimination (block Thomas).
    c_prime = [None] * max(n - 1, 0)
    d_prime = [None] * n
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

    # Back substitution.
    Theta = [None] * n
    Theta[-1] = d_prime[-1]
    for k in range(n - 2, -1, -1):
        Theta[k] = d_prime[k] - c_prime[k] @ Theta[k + 1]

    return torch.stack(Theta).to(x1.dtype)


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

def solve_sde(x1, n1, t, sigmas, potential_names=['x', 'x_abs', 'x2'], device='cpu', std_init=1, xt=None,
              compute_regularised=True, lam=1.0, regularization=0.0, eps_reg_theta=1e-6,
              regularised_solver='thomas'):
    """
    ... existing behaviour unchanged ...

    Extra (additive) outputs
    ------------------------
    The per-step corrector `theta_t` (returned as the stacked `eta_t2` array, i.e.
    etat2/(sigma*h)) is kept exactly as before. In ADDITION, when `compute_regularised`
    is True, a temporally-smoothed corrector `theta_regularised_t` is computed over the
    whole grid and appended as the LAST return value (None otherwise).

        compute_regularised : toggle the regularised corrector.
        lam                 : temporal-smoothing weight (lam -> 0 == per-node corrector).
        regularization      : diagonal jitter added to each gradient-Gram block M_k.
        eps_reg_theta       : scale-aware pivot jitter for the Thomas solve.
        regularised_solver  : 'thomas' (memory-safe, default) or 'dense' (verification).

    NOTE: the return tuple now has 8 elements (was 7). Update call sites accordingly.
    """

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

    # --- regularised (temporally-smoothed) corrector over the whole grid -------------
    # Kept ALONGSIDE the per-step theta_t above (not a substitute). Built from the
    # interpolant endpoints x0, x1 and the time grid only.
    theta_regularised_t = None
    if compute_regularised:
        if regularised_solver == 'dense':
            theta_regularised_t = regularised_theta(
                x0, x1, t, potentials, lam=lam, regularization=regularization, device=device)
        else:
            theta_regularised_t = regularised_theta_thomas(
                x0, x1, t, potentials, lam=lam, regularization=regularization,
                eps_reg_theta=eps_reg_theta, device=device)
        theta_regularised_t = theta_regularised_t.cpu().detach()

    return x0, xt, barphi_e, barphi_p, torch.stack([etat_t for etat_t in eta_t_list], dim=0), torch.stack([etat_t2 for etat_t2 in eta_t2_list], dim=0), dH_t_list, theta_regularised_t


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