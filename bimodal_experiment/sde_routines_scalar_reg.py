import resource
import sys
import time
import types
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcx, erf, erfinv
from scipy.integrate import trapezoid
from scipy import stats

from potentials_new import *

# shared regularised solver: codes/sde_routines.py needs the project root, codes/ and
# data/ importable. Appended (not prepended) so this folder's modules win any name clash.
_project_root = Path(__file__).resolve().parent.parent
for _p in (_project_root, _project_root / 'codes', _project_root / 'data'):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

from codes.sde_routines import SDE  # noqa: E402

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


def solve_regularised(system, lam, ridge=0.0):
    """
    Theta_reg for one saved/assembled system, with the shared float64 block-Thomas
    solver SDE._solve_regularised_thomas from codes/sde_routines.py (the same call
    as codes/resolve_theta_reg.py). Like forward_regularised, drops the first row.
    Returns (Theta_reg, t_reg, relative scaled residual).
    """
    solver_self = types.SimpleNamespace(num_potentials=system['num_potentials'])
    t = system['t'].double().numpy()
    Theta = SDE._solve_regularised_thomas(
        solver_self, t, system['M'], system['G'], system['b'], system['c'], lam, ridge=ridge)
    return Theta[1:], torch.as_tensor(t[1:]), solver_self.last_reg_residual


def solve_sde(x1, n1, t, sigmas, potential_names=['x', 'x_abs', 'x2'], device='cpu', std_init=1, xt=None,
              lam=1.0, n_subsample=1, regularization=0.0, reg_eps=1e-8,
              solve_reg=True, return_system=False):
    """
    Scalar MGD run that also assembles Guth et al.'s time-regularised theta system
    (as SDE.forward_regularised) and, if solve_reg, solves it for `lam` with the
    shared float64 block-Thomas solver (solve_regularised).

    Returns (x0, xt, barphi_e, barphi_p, eta_t, theta_t, dH_t, theta_reg_t), with
    theta_reg_t = None when solve_reg=False; return_system=True appends the system
    dict (SDE._save_reg_system format), so lam can be chosen after the run
    (select_lambda.py) -- lam never enters the SDE evolution.

    n_subsample > 1 averages the per-step ingredients over blocks of that many
    steps; each block is labelled at the mean time of its steps.
    """
    if regularization:
        raise ValueError("the dense-solve `regularization` ridge is gone; the shared "
                         "Thomas solver takes an explicit ridge (solve_regularised)")

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

    # per-step outputs preallocated on the device and copied to the host once at
    # the end: no per-step host allocations or device syncs over the nt steps
    buf = lambda *shape: torch.empty((nt,) + shape, device=x0.device, dtype=x0.dtype)
    eta_buf, theta_buf, dH_buf = buf(num_potentials), buf(num_potentials), buf()
    ratio = []

    # --- regularised theta problem: accumulators for the block-tridiagonal system ---
    # Mirrors SDE.forward_regularised. Quantities are collected at the *predicted*
    # walker y_k (target time t[i+1]). With n_subsample > 1 the fine-step ingredients
    # are averaged into coarse blocks.
    M_buf, G_buf = buf(num_potentials, num_potentials), buf(num_potentials, num_potentials)
    b_buf, c_buf = buf(num_potentials), buf(num_potentials)
    t_used, nb = [], 0                       # block times, number of blocks filled
    accM = accG = accb = accc = None
    acct = 0.0
    cnt = 0
    adot = 0.5 * np.pi                       # d/dt of the Cos-schedule angle a_t = (pi/2) t

    sigma = sigmas[0]
    loop_t0 = time.time()

    for i in range(nt):
        if i % 200 == 0:
            # ru_maxrss is in kB on Linux: peak host memory of this process so far
            print(f"Step {i}/{nt}  {time.time() - loop_t0:.0f} s  "
                  f"host maxRSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)

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

            if cos_a > reg_eps:
                z2    = x0 ** 2                              # ||Z||^2 per sample (d = 1)
                X_eff = (y_k - cos_a * x0) / sin_a          # reconstructed data endpoint X
                zx    = x0 * X_eff                          # Z . X per sample
                tau   = -adot * (tan_a * (1.0 - z2) + zx)   # tau_k^i               (N,)
                cc_k  = (mom * tau[:, None]).mean(0)        # E[phi(y_k) tau]       (r,)
            else:
                # t = 1: tan a diverges. The solver never reads the last node's c
                # (c_k couples nodes k and k+1), so the node is kept for its data
                # term M, b and its c is zeroed.
                cc_k  = torch.zeros(num_potentials, device=mom.device, dtype=mom.dtype)

            if cnt == 0:
                accM, accG = M_k.clone(), Gf_k.clone()
                accb, accc = bb_k.clone(), cc_k.clone()
                acct = t_node
            else:
                accM = accM + M_k; accG = accG + Gf_k
                accb = accb + bb_k; accc = accc + cc_k
                acct += t_node
            cnt += 1
            if cnt == n_subsample:
                t_used.append(acct / cnt)                   # block labelled at its mean time
                M_buf[nb], G_buf[nb] = accM / cnt, accG / cnt
                b_buf[nb], c_buf[nb] = accb / cnt, accc / cnt
                nb += 1
                cnt = 0

        sigma = sigmas[i+1]
        #ratio.append(torch.sqrt((etat_t@H@etat_t)/(etat_t2@H@etat_t2.T)))
        
        eta_buf[i], theta_buf[i], dH_buf[i] = etat_t.detach(), etat_t2.detach(), dH_t.detach()

        # Store statistics
        barphi_e[i + 1, :] = barphi(torch.cos(.5*torch.pi*t[i+1]) * x0 +  torch.sin(.5*torch.pi*t[i+1]) * x1, potentials) # barphi((1 - t[i + 1]) * x0 + t[i + 1] * x1, 0)
        barphi_p[i + 1, :] = barphi(xt, potentials)

    if cnt > 0:                                             # final partial block
        t_used.append(acct / cnt)
        M_buf[nb], G_buf[nb] = accM / cnt, accG / cnt
        b_buf[nb], c_buf[nb] = accb / cnt, accc / cnt
        nb += 1

    #plt.plot(ratio)
    #plt.show()

    # --- time-regularised theta system, in the SDE._save_reg_system format ---
    # (same [1:] front-trimming as SDE.forward_regularised: drop the first coarse
    #  node before the solve; solve_regularised then drops the first solution row).
    f32 = lambda X: X.detach().to('cpu', torch.float32)
    system = {
        't': torch.as_tensor(np.asarray(t_used[1:], dtype=np.float64)),
        'M': f32(M_buf[1:nb]),                              # (n, r, r)
        'G': f32(G_buf[1:nb]),                              # (n, r, r)
        'b': f32(b_buf[1:nb]),                              # (n, r)
        'c': f32(c_buf[1:nb]),                              # (n, r)
        'num_potentials': int(num_potentials),
        'meta': {'n_subsample': n_subsample, 'potential_names': list(potential_names)},
    }

    theta_reg_t = solve_regularised(system, lam)[0] if solve_reg else None

    out = (
        x0, xt, barphi_e, barphi_p,
        eta_buf.cpu(),
        theta_buf.cpu(),
        list(dH_buf.cpu().numpy()),
        theta_reg_t,
    )
    return out + (system,) if return_system else out


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