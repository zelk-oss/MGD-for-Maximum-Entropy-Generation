#!/usr/bin/env python3
"""Is the time-coupling matrix C well determined? From a saved regularised system (moment mode, any experiment).

Per cell k of the time-regularised solve (codes/sde_routines._solve_regularised_thomas), C = (Sigma_k + Sigma_k+1)/2,
mdot = (mdot_k + mdot_k+1)/2, and the time term is minimised at C Th' = mdot. Statistics are scaled by sqrt(diag M)
(as cosmic_web/mdot_range_cw.py), so that all of them are in the same units; dead statistics are left out.

Printed per cell, with lam_max / lam_min the largest / smallest eigenvalue of the scaled C:
  rank        number of eigenvalues above 1e-8 lam_max (exact null space: the rest)
  cond        lam_max / lam_min (lam_min <= 0 is printed as inf)
  for eps in 1e-6, 1e-4, 1e-2 (eigenvalues below eps * lam_max, the 'weak' directions):
    n_weak    how many eigenvalues
    mdot_w    share of |mdot|^2 lying on them
    sol_w     share of |Th'|^2 of the solution Th' = C^-1 mdot (eigenvalues > 1e-14 lam_max only) coming from them
If sol_w is large, the solution of the time term is dominated by directions where C is poorly determined.

Usage (repo root, Jean Zay, CPU, a few minutes per cell at r = 2414):
    python turbulence_2d/c_spectrum_t2d.py <system.pt> [<system2.pt> ...] [--out DIR]
Output: <out>/c_spectrum_<system stem>.npz (eigenvalues per cell, metrics) + the table on stdout.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / 'results' / 'c_spectrum'
EPS = (1e-6, 1e-4, 1e-2)


def cell_metrics(S0, S1, m0, m1, Md, live):
    C = (S0 + S1) / 2
    C = (C + C.T) / 2
    md = (m0 + m1) / 2
    keep = live & (Md > 0)
    s = Md[keep].sqrt()
    Cs = C[keep][:, keep] / (s[:, None] * s[None, :])
    ms = md[keep] / s
    ev, V = torch.linalg.eigh(Cs)                        # ascending
    lmax = float(ev[-1])
    proj = V.T @ ms
    nm2 = float((ms ** 2).sum())
    out = {'ev': ev.numpy(), 'r_live': int(keep.sum()), 'lmax': lmax,
           'rank': int((ev > 1e-8 * lmax).sum()),
           'cond': lmax / float(ev[0]) if float(ev[0]) > 0 else float('inf')}
    ok = ev > 1e-14 * lmax
    u2 = torch.zeros_like(ev)
    u2[ok] = (proj[ok] / ev[ok]) ** 2
    tot = float(u2.sum())
    for e in EPS:
        w = ev < e * lmax
        out[f'n_weak_{e:g}'] = int(w.sum())
        out[f'mdot_w_{e:g}'] = float((proj[w] ** 2).sum()) / nm2 if nm2 > 0 else float('nan')
        out[f'sol_w_{e:g}'] = float(u2[w].sum()) / tot if tot > 0 else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('systems', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for path in a.systems:
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        meta = s.get('meta', {})
        assert meta.get('reg_mode', 'moment') == 'moment', f'{path.name}: not a moment-mode system'
        t = s['t'].double().numpy()
        n, r = len(t), int(s['num_potentials'])
        live = s.get('live')
        print(f'\n=== {path.name[:120]}\n  r = {r}, nodes {n}, cells {n - 1}, n_walkers {meta.get("n_walkers")}, '
              f'n_subsample {meta.get("n_subsample")}, t = {np.round(t, 4).tolist()[:12]}')
        if n < 2:
            print('  fewer than 2 nodes: no cell, nothing to measure')
            continue
        head = f'  {"cell":>4} {"t":>8} {"rank":>6} {"cond":>9}' + ''.join(
            f' | eps={e:g}: n_weak mdot_w sol_w' for e in EPS)
        print(head)
        rows, evs = [], []
        S_prev, m_prev = s['Sigma'][0].double(), s['mdot'][0].double()
        for k in range(n - 1):
            S_next, m_next = s['Sigma'][k + 1].double(), s['mdot'][k + 1].double()
            Md = (torch.diagonal(s['M'][k].double()) + torch.diagonal(s['M'][k + 1].double())) / 2
            lv = torch.ones(r, dtype=torch.bool) if live is None else (live[k].bool() & live[k + 1].bool())
            o = cell_metrics(S_prev, S_next, m_prev, m_next, Md, lv)
            S_prev, m_prev = S_next, m_next
            evs.append(o.pop('ev')); rows.append(o)
            line = f'  {k:4d} {(t[k] + t[k + 1]) / 2:8.4f} {o["rank"]:6d} {o["cond"]:9.2e}'
            for e in EPS:
                line += f' | {o[f"n_weak_{e:g}"]:>13d} {o[f"mdot_w_{e:g}"]:6.3f} {o[f"sol_w_{e:g}"]:6.3f}'
            print(line)
        f = a.out / f'c_spectrum_{path.stem}.npz'
        np.savez_compressed(f, t=t, **{f'ev_cell{k}': e for k, e in enumerate(evs)},
                            **{k: np.array([row[k] for row in rows]) for k in rows[0]})
        print(f'  -> {f}')


if __name__ == '__main__':
    main()
