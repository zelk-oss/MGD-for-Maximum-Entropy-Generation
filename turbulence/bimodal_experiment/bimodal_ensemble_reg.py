"""
Run the MGD experiment with scalar bimodal distribution for K repetitions
to compute variance of theta estimates across independent runs — for BOTH
the raw MGD estimator and the regularized estimator (Florentin Guth's paper).

x1 ~ bimodal(n1, beta)          [data draw, redrawn every run — see TEST comment]

Randomness source:
  - x0 ~ N(0, std_init^2)            [SDE initialization, inside solve_sde]
  - x1 ~ bimodal(n1, beta)           [data draw, see TEST comment below]
Both are re-seeded per run via torch.manual_seed(42 + k) / np.random.seed(42 + k).

n1 is FIXED for a given invocation of this script (set via --n1, chosen once
in the launcher). beta is also fixed per invocation — the launcher submits
one job per beta value to sweep over beta while holding n1 fixed.

Regularized estimator: regularised_theta_scalar (regularised_theta_scalar.py)
is a direct scalar port of SDE.regularised_theta from sde_routines.py — the
dense full-resolution solve, no block-Thomas approximation. It's built from
the fixed endpoints x0/x1 (not the walker path), so it runs on (almost) the
full t grid rather than a coarse/subsampled one — there's no n_subsample
parameter for this version.
"""

import torch
import numpy as np
import os
import argparse

from sde_routines_scalar_reg import solve_sde
from utils import bimodal

parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=100)
parser.add_argument("--n1", type=int, default=1000000)
parser.add_argument("--nt", type=int, default=10000)
parser.add_argument("--sigma", type=float, default=10)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--lam", type=float, default=2e-5,
                     help="Smoothing regularization weight for regularised_theta_scalar")
parser.add_argument("--n_subsample", type=int, default=1,
                     help="Dimension of the regularization averaging time window")
parser.add_argument("--outdir", type=str, default="results_bim_theta_beta")
args = parser.parse_args()

K     = args.K
n1    = args.n1    # number of particles  →  complexity = n1 * nt   (fixed across the beta sweep)
nt    = args.nt
sigma = args.sigma
beta  = args.beta
lam   = args.lam
n_subsample = args.n_subsample

# ── Fixed hyperparameters ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t = torch.linspace(0, 1, nt + 1, device=device)

potential_names = ['x', 'x2', 'x3', 'x4']
DENOM           = torch.tensor([1., 2., 3., 4.], device=device)   # divisor for eta → theta
target_theta    = torch.tensor([beta / 2, 5 * beta, 0., -beta])   # linear in beta

complexity = n1 * nt          # fixed for every run

results_dir = args.outdir
os.makedirs(results_dir, exist_ok=True)

# ── Helper ───────────────────────────────────────────────────────────────────

def estimate_theta_full(n1: int, nt: int, seed: int):
    """
    Run one MGD solve and return raw + regularized estimates:
      theta_traj_mgd  : (nt+1, 4) – raw MGD theta estimate at every SDE step
      theta_final_mgd : (4,)      – raw estimate at t=1
      theta_traj_reg  : (n, 4)    – regularized theta estimate, n ≈ nt (dense
                                     full-resolution solve, its own time grid
                                     since the data endpoint is dropped)
      theta_final_reg : (4,)      – regularized estimate at its final time point

    Randomness:
      • x0 ~ N(0,1) inside solve_sde  (torch seed controls this)
      • x1 ~ bimodal(n1, beta)        (np seed controls this, redrawn every call)
    Seed is set by the caller before this function is invoked.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # TEST : x1 redrawn inside the loop (fresh dataset per run, not fixed)
    x1 = torch.from_numpy(bimodal(n1, beta)).to(device)

    x0, _, _, _, _, theta_t_list, _, theta_reg_t = solve_sde(
        x1,
        n1,
        t,
        [sigma for _ in range(nt + 1)],
        potential_names=potential_names,
        device=device,
        std_init=1,
        lam=lam,
        n_subsample=n_subsample,
    )
    theta_t_list = theta_t_list.to(device)
    x0     = x0.to(device)

    # ── raw MGD ───────────────────────────────────────────────────────────────
    theta_traj_mgd  = theta_t_list                 # (nt+1, 4)
    theta_final_mgd = theta_traj_mgd[-1]              # (4,)
    theta_traj_reg = theta_reg_t                  # (nt+1 / subsample, 4)
    theta_final_reg = theta_traj_reg[-1]


    return (
        theta_traj_mgd.cpu(), theta_final_mgd.cpu(),
        theta_traj_reg.cpu(), theta_final_reg.cpu(), 
    )

# energy variance and Cramer-Rao bound 
def get_phi(x_tensor):
    """Evaluates the basis potentials [x, x^2, x^3, x^4] for a given tensor."""
    return torch.stack([
        x_tensor, 
        x_tensor**2, 
        x_tensor**3, 
        x_tensor**4
    ], dim=1)

# ── Main loop ────────────────────────────────────────────────────────────────

all_theta_traj_mgd, all_theta_final_mgd = [], []
all_theta_traj_reg, all_theta_final_reg = [], []

for k in range(K):
    seed = 42 + k
    print(f"Run {k+1}/{K}  (beta={beta}, seed={seed})")

    (theta_traj_mgd, theta_final_mgd, 
     theta_traj_reg, theta_final_reg) = estimate_theta_full(n1, nt, seed)

    all_theta_traj_mgd.append(theta_traj_mgd)
    all_theta_final_mgd.append(theta_final_mgd)

    all_theta_traj_reg.append(theta_traj_reg)
    all_theta_final_reg.append(theta_final_reg)

    # ── Partial save after every run ─────────────────────────────────────────
    torch.save(
        {
            "theta_traj_mgd":  torch.stack(all_theta_traj_mgd),    # (k+1, nt+1, 4)
            "theta_final_mgd": torch.stack(all_theta_final_mgd),   # (k+1, 4)
            "theta_traj_reg":  torch.stack(all_theta_traj_reg),    # (k+1, n, 4)
            "theta_final_reg": torch.stack(all_theta_final_reg),   # (k+1, 4)
            "runs_done":       k + 1,
            "target_theta":    target_theta,
            "complexity":      complexity,
            "config": {
                "beta": beta, "sigma": sigma, "nt": nt, "n1": n1, "K": K,
                "lam": lam, "n_subsample": n_subsample, 
            },
        },
        os.path.join(results_dir, "partial_results.pt"),
    )

# ── Final consolidated save ──────────────────────────────────────────────────
theta_traj_mgd_all  = torch.stack(all_theta_traj_mgd)    # (K, nt+1, 4)
theta_final_mgd_all = torch.stack(all_theta_final_mgd)   # (K, 4)

theta_traj_reg_all  = torch.stack(all_theta_traj_reg)    # (K, n, 4)
theta_final_reg_all = torch.stack(all_theta_final_reg)   # (K, 4)

var_final_mgd = theta_final_mgd_all.var(0, unbiased=True)   # (4,)  
var_final_reg = theta_final_reg_all.var(0, unbiased=True)   # (4,)

# ── Variance of Energy & Cramér-Rao Bounds ───────────────────────────────────

# 1. Generate a large true sample to compute expectations over the dataset
N_ref = n1
x_ref_np = bimodal(N_ref, beta)
x_ref = torch.from_numpy(x_ref_np).to(device, dtype=torch.float32)

# 2. Compute covariance matrix of phi(X)
phi_x = get_phi(x_ref) # Shape: (N_ref, 4)
phi_mean = phi_x.mean(dim=0)
phi_centered = phi_x - phi_mean
# Cov(phi(X))
cov_phi = (phi_centered.T @ phi_centered) / (N_ref - 1) 
inv_cov_phi = torch.linalg.inv(cov_phi)

# 3. Cramér-Rao bound for theta (diagonal represents the variance bounds)
cr_bound_theta_matrix = inv_cov_phi / n1
cr_bound_theta_var = torch.diag(cr_bound_theta_matrix)

# 4. Compute the MSE matrix for theta across the K runs (Expectation over runs)
delta_theta_mgd = theta_final_mgd_all.to(device) - theta_final_mgd_all.mean(0).to(device)
mse_matrix_mgd = (delta_theta_mgd.T @ delta_theta_mgd) / K

delta_theta_reg = theta_final_reg_all.to(device) - theta_final_reg_all.mean(0).to(device)
mse_matrix_reg = (delta_theta_reg.T @ delta_theta_reg) / K

# 5. Compute Variance of the Energy (Outer expectation over the dataset)
# Evaluates E_x [ phi(x)^T * MSE_matrix * phi(x) ] efficiently using batch operations
var_energy_mgd = torch.sum((phi_x @ mse_matrix_mgd) * phi_x, dim=1).mean() 
var_energy_reg = torch.sum((phi_x @ mse_matrix_reg) * phi_x, dim=1).mean()

# 6. Cramér-Rao bound for the Energy
cr_bound_energy = torch.sum((phi_x @ inv_cov_phi) * phi_x, dim=1).mean() / n1


print("\n── Results ──────────────────────────────────────")
print(f"beta              : {beta}")
print(f"Target theta      : {target_theta.tolist()}")
print(f"CR Bound (Theta)  : {cr_bound_theta_var.tolist()}")
print(f"Var theta (MGD)   : {var_final_mgd.tolist()}")
print(f"Var theta (Reg)   : {var_final_reg.tolist()}")
print("─" * 40)
print(f"CR Bound (Energy) : {cr_bound_energy.item():.6e}")
print(f"Var Energy (MGD)  : {var_energy_mgd.item():.6e}")
print(f"Var Energy (Reg)  : {var_energy_reg.item():.6e}")

# Append the new metrics to your torch.save dict
torch.save(
    {
        "theta_traj_mgd":  theta_traj_mgd_all,    
        "theta_traj_reg":  theta_traj_reg_all,    
        "theta_final_mgd": theta_final_mgd_all,   
        "theta_final_reg": theta_final_reg_all,   
        "var_final_mgd":   var_final_mgd,
        "var_final_reg":   var_final_reg,
        # New additions below:
        "var_energy_mgd":  var_energy_mgd.cpu(),
        "var_energy_reg":  var_energy_reg.cpu(),
        "cr_bound_theta":  cr_bound_theta_var.cpu(),
        "cr_bound_energy": cr_bound_energy.cpu(),
        "mse_matrix_mgd":  mse_matrix_mgd.cpu(),
        "mse_matrix_reg":  mse_matrix_reg.cpu(),
        # reference
        "target_theta":    target_theta,
        "complexity":      complexity,        
        "config": {
            "beta": beta, "sigma": sigma, "nt": nt, "n1": n1, "K": K,
            "lam": lam, "n_subsample": n_subsample, 
        },
    },
    os.path.join(results_dir, "experiment_K_runs.pt"),
)

print(f"\nSaved to {results_dir}/experiment_K_runs.pt")