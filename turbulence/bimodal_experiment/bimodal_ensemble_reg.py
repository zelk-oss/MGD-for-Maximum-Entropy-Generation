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

print("\n── Results ──────────────────────────────────────")
print(f"beta              : {beta}")
print(f"Target theta      : {target_theta.tolist()}")
print(f"Mean theta_final (MGD)        : {theta_final_mgd_all.mean(0).tolist()}")
print(f"Var  theta_final (MGD)        : {var_final_mgd.tolist()}")
print(f"Mean theta_final (Regularized): {theta_final_reg_all.mean(0).tolist()}")
print(f"Var  theta_final (Regularized): {var_final_reg.tolist()}")

torch.save(
    {
        # full trajectories: lets you replay / animate convergence
        "theta_traj_mgd":  theta_traj_mgd_all,    # (K, nt+1, 4)
        "theta_traj_reg":  theta_traj_reg_all,    # (K, n, 4)
        # summary statistics
        "theta_final_mgd": theta_final_mgd_all,   # (K, 4)
        "theta_final_reg": theta_final_reg_all,   # (K, 4)
        "var_final_mgd":   var_final_mgd,
        "var_final_reg":   var_final_reg,
        # reference
        "target_theta":    target_theta,
        # experimental cost
        "complexity":      complexity,        # = n1 * nt  (scalar, fixed across the beta sweep)
        "config": {
            "beta": beta, "sigma": sigma, "nt": nt, "n1": n1, "K": K,
            "lam": lam, "n_subsample": n_subsample, 
        },
    },
    os.path.join(results_dir, "experiment_K_runs.pt"),
)

print(f"\nSaved to {results_dir}/experiment_K_runs.pt")