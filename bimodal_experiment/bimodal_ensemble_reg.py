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

Two modes, so the K runs can be spread over many jobs:
  run (default)  runs k = run_start .. run_start+n_runs-1 of the K and writes, per run,
                   <outdir>/mgd/run_<k>.pt         raw MGD theta trajectory / final
                   <outdir>/reg_system/run_<k>.pt  regularised system (SDE._save_reg_system
                                                   format, ~8 MB at nt=5e4)
                 Runs whose two files already exist are skipped (resubmit-safe).
  --aggregate    after all runs: variances, Cramér–Rao bounds and phi statistics of
                 the data -> <outdir>/experiment_K_runs.pt

Regularized estimator: lam is NOT chosen here. select_lambda.py solves every saved
system for a lam grid with the shared float64 block-Thomas solver and picks lam
from the data (lam never enters the SDE evolution). --n_subsample > 1 averages the
per-step ingredients over blocks of that many steps (the June 2026 sweeps used
10); the default 1 solves Guth's system on the full grid.

Basis: the potentials are phi_a(x) = x^a / a (potentials_new.py), so the raw
theta estimates are coefficients of x^a / a. Every variance and bound saved
here is in that same phi basis; target_theta (coefficients of x^a in
log p = -beta x^4 + 5 beta x^2 + beta x / 2) is converted with DENOM.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from bimodal_utils import bimodal
from sde_routines_scalar_reg import solve_sde, get_potentials, moments_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=100, help="total number of runs of the ensemble")
parser.add_argument("--run_start", type=int, default=0, help="first run index done by this job")
parser.add_argument("--n_runs", type=int, default=None, help="runs done by this job (default: all K)")
parser.add_argument("--aggregate", action="store_true",
                    help="don't run: combine the saved runs into experiment_K_runs.pt")
parser.add_argument("--n1", type=int, default=1000000)
parser.add_argument("--nt", type=int, default=10000)
parser.add_argument("--sigma", type=float, default=10)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--n_subsample", type=int, default=1,
                    help="Block size for averaging the regularised system over time steps")
parser.add_argument("--outdir", type=str, default="results_bim_theta_beta")
args = parser.parse_args()

K     = args.K
n1    = args.n1    # number of particles  →  complexity = n1 * nt   (fixed across the beta sweep)
nt    = args.nt
sigma = args.sigma
beta  = args.beta
n_subsample = args.n_subsample

# ── Fixed hyperparameters ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

potential_names = ['x', 'x2', 'x3', 'x4']
DENOM           = torch.tensor([1., 2., 3., 4.])       # phi_a = x^a / a
target_theta    = torch.tensor([beta / 2, 5 * beta, 0., -beta])   # coefficients of x^a, linear in beta
target_theta_phi = target_theta * DENOM                # the same density, coefficients of phi_a

complexity = n1 * nt          # fixed for every run

results_dir = Path(args.outdir)
system_dir  = results_dir / "reg_system"
mgd_dir     = results_dir / "mgd"

config = {
    "beta": beta, "sigma": sigma, "nt": nt, "n1": n1, "K": K,
    "n_subsample": n_subsample, "potential_names": potential_names,
}


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ── Run mode ─────────────────────────────────────────────────────────────────

def estimate_theta_full(seed: int):
    """
    Run one MGD solve and return the raw estimate plus the regularised system:
      theta_traj_mgd  : (nt, 4) – raw MGD theta estimate at every SDE step
      system          : dict    – regularised system, solved later for any lam

    Randomness:
      • x0 ~ N(0,1) inside solve_sde  (torch seed controls this)
      • x1 ~ bimodal(n1, beta)        (np seed controls this, redrawn every call)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # TEST : x1 redrawn inside the loop (fresh dataset per run, not fixed)
    x1 = torch.from_numpy(bimodal(n1, beta)).to(device)
    t = torch.linspace(0, 1, nt + 1, device=device)

    out = solve_sde(
        x1,
        n1,
        t,
        [sigma for _ in range(nt + 1)],
        potential_names=potential_names,
        device=device,
        std_init=1,
        n_subsample=n_subsample,
        solve_reg=False,
        return_system=True,
    )
    return out[5].cpu(), out[8]


def run():
    system_dir.mkdir(parents=True, exist_ok=True)
    mgd_dir.mkdir(parents=True, exist_ok=True)
    stop = K if args.n_runs is None else min(K, args.run_start + args.n_runs)
    for k in range(args.run_start, stop):
        seed = 42 + k
        sys_path, mgd_path = system_dir / f"run_{k:03d}.pt", mgd_dir / f"run_{k:03d}.pt"
        if sys_path.exists() and mgd_path.exists():
            print(f"Run {k+1}/{K} already done, skipping")
            continue
        print(f"Run {k+1}/{K}  (beta={beta}, seed={seed})", flush=True)

        theta_traj_mgd, system = estimate_theta_full(seed)
        system["meta"].update(config, seed=seed, run=k)
        save_atomic(system, sys_path)
        save_atomic({"theta_traj_mgd": theta_traj_mgd,      # (nt, 4)
                     "theta_final_mgd": theta_traj_mgd[-1],  # (4,)
                     "seed": seed, "run": k, "config": config}, mgd_path)


# ── Aggregate mode ───────────────────────────────────────────────────────────

def aggregate():
    files = sorted(mgd_dir.glob("run_*.pt"))
    runs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    missing = sorted(set(range(K)) - {r["run"] for r in runs})
    if missing:
        print(f"WARNING: {len(missing)} of K={K} runs missing: {missing}")
    if len(runs) < 2:
        raise SystemExit("need at least 2 runs to aggregate")

    theta_traj_mgd_all  = torch.stack([r["theta_traj_mgd"] for r in runs])    # (K, nt, 4)
    theta_final_mgd_all = torch.stack([r["theta_final_mgd"] for r in runs])   # (K, 4)
    var_final_mgd = theta_final_mgd_all.var(0, unbiased=True)                 # (4,)

    # ── Reference statistics of phi under the data distribution ──────────────
    # Drawn once here (same seed for every beta) and saved, so select_lambda.py can
    # evaluate the energy variance of any estimator without redrawing.
    np.random.seed(0)
    x_ref = torch.from_numpy(bimodal(n1, beta)).to(device, dtype=torch.float64)
    phi_x = moments_matrix(x_ref, get_potentials(potential_names, device))   # (N_ref, 4), phi basis
    mean_phi = phi_x.mean(0)
    phi_centered = phi_x - mean_phi
    cov_phi = (phi_centered.T @ phi_centered) / (len(x_ref) - 1)       # Cov(phi(X)), Fisher information
    second_moment_phi = (phi_x.T @ phi_x) / len(x_ref)                  # E[phi phi^T]
    inv_cov_phi = torch.linalg.inv(cov_phi)

    # Cramér-Rao bound for theta (n1 samples per run)
    cr_bound_theta_matrix = inv_cov_phi / n1
    cr_bound_theta_var = torch.diag(cr_bound_theta_matrix)

    # Energy E(x) = theta . phi(x). Var over runs, averaged over x ~ data:
    #   E_x[phi^T C phi] = tr(C E[phi phi^T])      (uncentered, as in the June runs)
    #   tr(C Cov(phi))                             (centered: energy up to a constant)
    C_mgd = torch.cov(theta_final_mgd_all.T.double().to(device))
    var_energy_mgd = torch.trace(C_mgd @ second_moment_phi)
    var_energy_mgd_centered = torch.trace(C_mgd @ cov_phi)
    cr_bound_energy = torch.trace(cr_bound_theta_matrix @ second_moment_phi)
    cr_bound_energy_centered = torch.trace(cr_bound_theta_matrix @ cov_phi)   # = r / n1

    print("\n── Results (raw MGD; REG is computed by select_lambda.py) ──")
    print(f"beta              : {beta}   runs: {len(runs)}/{K}")
    print(f"Target theta (phi): {target_theta_phi.tolist()}")
    print(f"Mean theta (MGD)  : {theta_final_mgd_all.mean(0).tolist()}")
    print(f"CR Bound (Theta)  : {cr_bound_theta_var.tolist()}")
    print(f"Var theta (MGD)   : {var_final_mgd.tolist()}")
    print("─" * 40)
    print(f"CR Bound (Energy) : {cr_bound_energy.item():.6e}  (centered {cr_bound_energy_centered.item():.6e})")
    print(f"Var Energy (MGD)  : {var_energy_mgd.item():.6e}  (centered {var_energy_mgd_centered.item():.6e})")

    save_atomic(
        {
            "theta_traj_mgd":  theta_traj_mgd_all,
            "theta_final_mgd": theta_final_mgd_all,
            "runs":            [r["run"] for r in runs],
            "var_final_mgd":   var_final_mgd,
            "var_energy_mgd":  var_energy_mgd.cpu(),
            "var_energy_mgd_centered": var_energy_mgd_centered.cpu(),
            "cr_bound_theta":  cr_bound_theta_var.cpu(),
            "cr_bound_theta_matrix": cr_bound_theta_matrix.cpu(),
            "cr_bound_energy": cr_bound_energy.cpu(),
            "cr_bound_energy_centered": cr_bound_energy_centered.cpu(),
            # phi statistics under the data (phi basis), for select_lambda.py
            "mean_phi":          mean_phi.cpu(),
            "cov_phi":           cov_phi.cpu(),
            "second_moment_phi": second_moment_phi.cpu(),
            # reference
            "target_theta":     target_theta,
            "target_theta_phi": target_theta_phi,
            "complexity":       complexity,
            "config":           config,
        },
        results_dir / "experiment_K_runs.pt",
    )
    print(f"\nSaved to {results_dir}/experiment_K_runs.pt")


if __name__ == "__main__":
    aggregate() if args.aggregate else run()
