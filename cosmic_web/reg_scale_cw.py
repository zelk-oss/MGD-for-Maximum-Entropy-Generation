#!/usr/bin/env python3
"""Smoothing time scale of the time-regularised solve, from a saved regularised system (any experiment).

Moment mode, uniform schedule (codes/sde_routines.reg_energy_weights): per unit t the data term is 1/2 Th^T M Th and the
time term is lam/2 Th'^T C Th' (M = E[grad phi grad phi^T], C = Cov_walkers(phi)). Balancing them with Th' ~ Th / tau gives
    tau(lam) = sqrt(lam) * tau1,   tau1 = median_k sqrt(C_kk / M_kk)     (statistics with both diagonals > 0)
the time over which Theta_reg is smoothed for that lam. It depends on the signal only through C/M, so it is the number
that makes a lam of one experiment comparable with another (e.g. turbulence lam 3e-5 .. 1e-3 against the cosmic web).
Also printed: the conditioning of the Jacobi-scaled blocks of M and C at those nodes, and n_walkers (rank(C) <= n - 1).

Usage (repo root, Jean Zay, CPU, a minute):
    python cosmic_web/reg_scale_cw.py <system.pt> [<system2.pt> ...]  [--t 0.1 0.5 0.9 0.99 0.999] [--lams 3e-5 1e-3 1e-2 1 100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def block_stats(X):
    """Jacobi-scaled min eigenvalue and number of eigenvalues < 1e-10 of a symmetric block (float64)."""
    X = X.double(); X = (X + X.T) / 2
    d = torch.diagonal(X); pos = d > 0
    Xs = X[pos][:, pos] / (d[pos].sqrt()[:, None] * d[pos].sqrt()[None, :])
    ev = torch.linalg.eigvalsh(Xs)
    return float(ev.min()), int((ev < 1e-10).sum()), int(pos.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('systems', nargs='+', type=Path)
    ap.add_argument('--t', nargs='+', type=float, default=[0.1, 0.5, 0.9, 0.99, 0.999])
    ap.add_argument('--lams', nargs='+', type=float, default=[3e-5, 1e-3, 1e-2, 1.0, 100.0])
    a = ap.parse_args()
    for path in a.systems:
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        meta = s.get('meta', {})
        t = s['t'].double().numpy()
        print(f'\n=== {path.name[:110]}\n  mode {meta.get("reg_mode")}, r = {s["num_potentials"]}, nodes {len(t)}, dim {meta.get("dim")}, '
              f'n_walkers {meta.get("n_walkers")}')
        print(f'  {"t":>7} {"node":>5} {"tau1=sqrt(C/M)":>15} {"p10":>9} {"p90":>9} | {"M min eig":>10} {"#<1e-10":>8} | {"C min eig":>10} {"#<1e-10":>8}'
              + ''.join(f' | tau(lam={l:g})' for l in a.lams))
        for tt in a.t:
            k = int(np.argmin(np.abs(t - tt)))
            Mk, Ck = s['M'][k].double(), s['Sigma'][k].double()
            dm, dc = torch.diagonal(Mk), torch.diagonal(Ck)
            ok = (dm > 0) & (dc > 0)
            r = (dc[ok] / dm[ok]).sqrt().numpy()
            tau1 = float(np.median(r))
            (m_min, m_small, _), (c_min, c_small, _) = block_stats(Mk), block_stats(Ck)
            print(f'  {t[k]:7.4f} {k:5d} {tau1:15.3e} {np.percentile(r, 10):9.2e} {np.percentile(r, 90):9.2e} | {m_min:10.2e} {m_small:8d} | '
                  f'{c_min:10.2e} {c_small:8d}' + ''.join(f' | {np.sqrt(l) * tau1:12.2e}' for l in a.lams))


if __name__ == '__main__':
    main()
