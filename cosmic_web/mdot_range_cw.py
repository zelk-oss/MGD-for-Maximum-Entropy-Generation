#!/usr/bin/env python3
"""H3: does mdot lie in the directions the walker covariance C can see? From a saved regularised system (any experiment).

Time term of Theta_reg per cell (codes/sde_routines._solve_regularised_thomas, moment mode):
    lam w dt [ 1/2 Th'^T C Th' - mdot^T Th' ],   C = (Sigma_k + Sigma_k+1)/2,  mdot = (mdot_k + mdot_k+1)/2,
minimised at C Th' = mdot. A component of mdot along directions where C = 0 is not balanced by the quadratic term: there
the time term pushes Th' without bound and only the data term (M) and the ridge stop it.

Per cell: the fraction of mdot outside the range of C,  |P_null mdot| / |mdot|, in the metric of the data term
(statistics scaled by sqrt(diag M), so that every statistic is in the same units). The null space is defined two ways:
  rel  : eigenvalues of the scaled C below --rel_tol x its largest eigenvalue
  rank : everything outside the top n_walkers - 1 eigenvectors (the exact rank bound of one node's covariance)
Live mask: dead statistics (system 'live', if saved) are left out, as the solver pins them.

Usage (repo root, Jean Zay, CPU, minutes per system):
    python cosmic_web/mdot_range_cw.py <system.pt> [<system2.pt> ...] [--t 0.1 0.5 0.9 0.99 0.999 1] [--out DIR]
Output: <out>/mdot_range_<system stem>.npz  (t_cell, frac_rel, frac_rank, n_above_rel, n_walkers)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / 'results' / 'mdot_range'


def cell_fraction(S0, S1, m0, m1, Md, live, n_walkers, rel_tol):
    """(frac_rel, frac_rank, n_above_rel) for one cell; all inputs float64 torch."""
    C = (S0 + S1) / 2
    C = (C + C.T) / 2
    md = (m0 + m1) / 2
    keep = live & (Md > 0)
    s = Md[keep].sqrt()
    Cs = C[keep][:, keep] / (s[:, None] * s[None, :])
    ms = md[keep] / s
    ev, V = torch.linalg.eigh(Cs)                        # ascending
    nm = float(ms.norm())
    if nm == 0:
        return np.nan, np.nan, 0
    proj = V.T @ ms                                      # components of mdot on the eigenvectors
    null_rel = ev < rel_tol * ev[-1]
    n_rank = max(len(ev) - (n_walkers - 1), 0)           # the smallest len - (n-1) eigenvectors
    frac_rel = float(proj[null_rel].norm()) / nm
    frac_rank = float(proj[:n_rank].norm()) / nm
    return frac_rel, frac_rank, int((~null_rel).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('systems', nargs='+', type=Path)
    ap.add_argument('--t', nargs='+', type=float, default=[0.1, 0.5, 0.9, 0.99, 0.999, 1.0],
                    help='print the cell nearest to each t (all cells are saved)')
    ap.add_argument('--rel_tol', type=float, default=1e-8)
    ap.add_argument('--n_walkers', type=int, default=None, help='default: the system meta')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for path in a.systems:
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        meta = s.get('meta', {})
        assert meta.get('reg_mode', 'moment') == 'moment', f'{path.name}: not a moment-mode system'
        nw = a.n_walkers or meta.get('n_walkers')
        assert nw, 'n_walkers not in the system meta: pass --n_walkers'
        t = s['t'].double().numpy()
        n, r = len(t), int(s['num_potentials'])
        live = s.get('live')
        fr, fk, na = np.full(n - 1, np.nan), np.full(n - 1, np.nan), np.zeros(n - 1, int)
        S_prev, m_prev = s['Sigma'][0].double(), s['mdot'][0].double()
        for k in range(n - 1):
            S_next, m_next = s['Sigma'][k + 1].double(), s['mdot'][k + 1].double()
            Md = (torch.diagonal(s['M'][k].double()) + torch.diagonal(s['M'][k + 1].double())) / 2
            lv = torch.ones(r, dtype=torch.bool) if live is None else (live[k].bool() & live[k + 1].bool())
            fr[k], fk[k], na[k] = cell_fraction(S_prev, S_next, m_prev, m_next, Md, lv, nw, a.rel_tol)
            S_prev, m_prev = S_next, m_next
        tc = (t[:-1] + t[1:]) / 2
        print(f'\n=== {path.name[:110]}\n  r = {r}, cells {n - 1}, n_walkers {nw}, live mask {"yes" if live is not None else "no"}')
        print(f'  {"t":>8} {"cell":>6} {"# C eig > tol":>14} {"frac rel":>9} {"frac rank":>10}')
        for tt in a.t:
            k = int(np.argmin(np.abs(tc - tt)))
            print(f'  {tc[k]:8.5f} {k:6d} {na[k]:14d} {fr[k]:9.3f} {fk[k]:10.3f}')
        for lo, hi in [(0, 0.9), (0.9, 0.99), (0.99, 1.0)]:
            sel = (tc >= lo) & (tc < hi)
            if sel.any():
                print(f'  t in [{lo}, {hi}): median frac rel {np.nanmedian(fr[sel]):.3f}, frac rank {np.nanmedian(fk[sel]):.3f} '
                      f'({sel.sum()} cells)')
        f = a.out / f'mdot_range_{path.stem}.npz'
        np.savez_compressed(f, t_cell=tc, frac_rel=fr, frac_rank=fk, n_above_rel=na, n_walkers=nw)
        print(f'  -> {f}')


if __name__ == '__main__':
    main()
