"""Choose lam for the regularised estimator of one (or more) bimodal beta runs.

For each results dir written by bimodal_ensemble_reg.py (experiment_K_runs.pt +
reg_system/run_*.pt), solves every run's regularised system for a lam grid with
the batched float64 block-Thomas (codes/lam_selection.py, checked against
SDE._solve_regularised_thomas), then:

- data-driven choice (no ground truth, the rule used for turbulence,
  turbulence/lamtune_select.ipynb): residual R = Theta(lam) - Theta(0), block
  E[z^2] per t-window, averaged over runs; lam_selected = the largest lam with
  E[z^2] <= Z2_MAX in every window.
- oracle check (this experiment knows theta*): per lam, the Fisher-weighted
  energy error E_runs[(theta_1 - theta*)^T Cov(phi) (theta_1 - theta*)]; lam_oracle
  is its minimiser. It only validates the data-driven rule.

Everything is in the phi basis of the solver (phi_a = x^a / a).

Output: <dir>/lam_selection.pt

Usage:
    python select_lambda.py results_bim_theta_beta/<sweep>/beta_0.5000 [...] \
        [--lams 0 1e-8 3e-8 ...] [--chunk 5] [--threads N]
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import sde_routines_scalar_reg  # noqa: F401  (puts codes/ on sys.path)
from lam_selection import (BLOCKS, W_DECIDE, Z2_MAX, block_z2, solve_regularised_thomas_batched,
                           suggest_lam, window_masks)

# same grid as turbulence/launch/resolve_lamsweep.sh: 0 (the reference) + half decades
DEFAULT_LAMS = [0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
N_KEEP_TRAJ = 3        # runs whose full Theta_reg(t) is kept for every lam


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('dirs', nargs='+', type=Path, help='beta results dirs')
    p.add_argument('--lams', nargs='+', type=float, default=DEFAULT_LAMS)
    p.add_argument('--ridge', type=float, default=0.0, help='M_k += ridge * diag(M_k)')
    p.add_argument('--chunk', type=int, default=5, help='runs solved together (memory ~ 0.3 GB/run at n=5e4)')
    p.add_argument('--threads', type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    return p.parse_args()


def energy_forms(delta, S):
    """Mean over runs of delta^T S delta; delta (K, ..., r), S (r, r)."""
    return torch.einsum('k...a,ab,k...b->...', delta, S, delta) / delta.shape[0]


def select(d, lams, ridge, chunk):
    exp = torch.load(d / 'experiment_K_runs.pt', map_location='cpu', weights_only=False)
    # exactly the runs aggregated into experiment_K_runs.pt, so MGD and REG share runs
    files = [d / 'reg_system' / f'run_{k:03d}.pt' for k in exp['runs']]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'[{d}] systems missing for aggregated runs: {missing}')
    if len(files) != exp['config']['K']:
        print(f'[{d}] WARNING: {len(files)} runs of K={exp["config"]["K"]}')
    target = exp['target_theta_phi'].double()
    cov_phi, S2 = exp['cov_phi'].double(), exp['second_moment_phi'].double()

    L, K = len(lams), len(files)
    t_reg = masks = None
    z2_runs, finals, keep_traj, resid = [], [], [], []
    s1 = s2 = None
    t0 = time.time()
    for i0 in range(0, K, chunk):
        systems = [torch.load(f, map_location='cpu', weights_only=False) for f in files[i0:i0 + chunk]]
        t = systems[0]['t'].double()
        for s, f in zip(systems, files[i0:i0 + chunk]):
            assert torch.equal(s['t'].double(), t), f'{f}: time grid differs from run 0'
        stack = lambda key: torch.stack([s[key] for s in systems])
        Th, res = solve_regularised_thomas_batched(t.numpy(), stack('M'), stack('G'), stack('b'),
                                                   stack('c'), lams, ridge=ridge)
        del systems
        Th = Th[:, :, 1:]                       # drop the first row, as forward_regularised
        resid.append(res)
        if t_reg is None:
            t_reg = t[1:].numpy()
            masks = window_masks(t_reg)

        finals.append(Th[:, :, -1])             # (R, L, r), theta at t = 1
        if sum(x.shape[0] for x in keep_traj) < N_KEEP_TRAJ:
            keep_traj.append(Th[:N_KEEP_TRAJ].float())
        s1 = Th.sum(0) if s1 is None else s1 + Th.sum(0)
        s2 = (Th ** 2).sum(0) if s2 is None else s2 + (Th ** 2).sum(0)

        Th = Th.numpy()
        for r_ in range(Th.shape[0]):           # residual vs lam = 0 per run
            ref = Th[r_, lams.index(0.0)]
            z2_runs.append([[[block_z2(Th[r_, i][m] - ref[m], w) for w in BLOCKS] for _, _, m in masks]
                            if lam > 0 else [[np.nan] * len(BLOCKS)] * len(masks)
                            for i, lam in enumerate(lams)])
        print(f'[{d.name}] runs {i0 + 1}-{min(i0 + chunk, K)}/{K} solved ({time.time() - t0:.0f} s)')

    with warnings.catch_warnings():             # all-NaN where a window is shorter than 2 blocks
        warnings.simplefilter('ignore', RuntimeWarning)
        z2 = np.nanmean(np.array(z2_runs, dtype=float), axis=0)          # (L, n_windows, n_blocks)
    resid = torch.cat(resid)                                              # (K, L)

    # across-run variance along t, relative to lam = 0 (median over nodes and coefficients)
    var_t = (s2 - s1 ** 2 / K) / (K - 1)                                  # (L, n, r)
    v, i0 = var_t.numpy(), lams.index(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        vgain = np.array([[np.nanmedian(v[i][m] / np.where(v[i0][m] > 0, v[i0][m], np.nan))
                           for _, _, m in masks] for i in range(L)])

    theta_final_reg = torch.cat(finals)                                   # (K, L, r)
    mgd = exp['theta_final_mgd'].double()                                 # (K, r)
    dev_reg = theta_final_reg - theta_final_reg.mean(0)
    dev_mgd = mgd - mgd.mean(0)
    err_reg, err_mgd = theta_final_reg - target, mgd - target

    out = {
        'lams': list(lams), 'ridge': ridge, 't_reg': torch.as_tensor(t_reg),
        'windows': [(lo, hi, int(m.sum())) for lo, hi, m in masks], 'blocks': BLOCKS,
        'z2': z2, 'variance_gain': vgain, 'solve_residual': resid,
        'theta_final_reg': theta_final_reg,
        'theta_traj_reg_examples': torch.cat(keep_traj)[:N_KEEP_TRAJ],   # (N_KEEP_TRAJ, L, n, r)
        'theta_reg_mean': (s1 / K).float(), 'theta_reg_std': var_t.clamp_min(0).sqrt().float(),
        # per lam (REG) and for raw MGD, phi basis
        'var_theta_reg': theta_final_reg.var(0), 'var_theta_mgd': mgd.var(0),
        'mse_theta_reg': (err_reg ** 2).mean(0), 'mse_theta_mgd': (err_mgd ** 2).mean(0),
        'var_energy_reg': energy_forms(dev_reg, S2) * K / (K - 1),
        'var_energy_reg_centered': energy_forms(dev_reg, cov_phi) * K / (K - 1),
        'var_energy_mgd': energy_forms(dev_mgd, S2) * K / (K - 1),
        'var_energy_mgd_centered': energy_forms(dev_mgd, cov_phi) * K / (K - 1),
        'mse_energy_reg_centered': energy_forms(err_reg, cov_phi),     # oracle criterion
        'mse_energy_mgd_centered': energy_forms(err_mgd, cov_phi),
        'cr_bound_energy': exp['cr_bound_energy'], 'cr_bound_energy_centered': exp['cr_bound_energy_centered'],
        'cr_bound_theta': exp['cr_bound_theta'],
        'target_theta_phi': target, 'config': exp['config'],
    }
    out['lam_selected'] = suggest_lam(lams, z2)
    out['lam_oracle'] = lams[int(torch.argmin(out['mse_energy_reg_centered']))]

    tmp = d / 'lam_selection.pt.tmp'
    torch.save(out, tmp)
    tmp.replace(d / 'lam_selection.pt')
    return out


def report(d, o):
    b = BLOCKS.index(W_DECIDE)
    print(f"\n=== {d}  beta={o['config']['beta']}  K={o['theta_final_reg'].shape[0]}")
    print(f"lam selected (E[z^2] <= {Z2_MAX}, w={W_DECIDE}): {o['lam_selected']}   "
          f"lam oracle (min Fisher-weighted energy MSE): {o['lam_oracle']}   "
          f"max solve residual {o['solve_residual'].max():.1e}")
    hdr = ' '.join(f'[{lo:.2f},{hi:.2f})' for lo, hi, _ in o['windows'])
    print(f"{'lam':>8} | E[z^2] per window {hdr} | var gain (worst) | Var E (centered) | MSE E (centered)")
    for i, lam in enumerate(o['lams']):
        zs = ' '.join(f'{v:13.2f}' for v in o['z2'][i, :, b])
        print(f"{lam:8.0e} | {zs} | {np.nanmax(o['variance_gain'][i]):8.3f} | "
              f"{o['var_energy_reg_centered'][i]:.3e} | {o['mse_energy_reg_centered'][i]:.3e}")
    print(f"{'MGD':>8} | {'':{len(zs)}s} | {'':8s} | {o['var_energy_mgd_centered']:.3e} | "
          f"{o['mse_energy_mgd_centered']:.3e}")
    print(f"{'CR':>8} | {'':{len(zs)}s} | {'':8s} | {o['cr_bound_energy_centered']:.3e} |")


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    lams = sorted(set(float(x) for x in args.lams) | {0.0})     # lam = 0 is the reference
    for d in args.dirs:
        report(d, select(d, lams, args.ridge, args.chunk))


if __name__ == '__main__':
    main()
